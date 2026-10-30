"""Bale platform adapter — direct HTTP client for the Bale Bot API (tapi.bale.ai).

Bale's bot API is modelled on the Telegram Bot API: a token issued by @botfather inside
the Bale app, updates via ``getUpdates`` long polling (webhook mode is a future
extension), and ``sendMessage`` / ``sendPhoto`` / ``sendChatAction`` methods. All
outbound text is auto-formatted by Bale as markdown — bold needs `` *text* `` and italic
``_text_`` (each requiring surrounding spaces) — so the adapter converts Hermes
markdown to that flavour before sending.

Inbound media: photo/video/document/audio updates carry a ``file_id``; the adapter
calls ``getFile`` for the download path, fetches it from
``https://tapi.bale.ai/file/bot<token>/<path>`` (bots may download ≤20 MB), and
caches it through Hermes' shared media pipeline before the turn runs.
Stickers/animation are not supported yet. Outbound voice notes ride the Telegram-style
``sendVoice`` endpoint: native voice bubbles require Ogg/Opus, so non-Opus audio is
transcoded with ffmpeg when available (the same contract Telegram and Matrix enforce).
Without ffmpeg the original container is uploaded with its honest content type.

config.yaml ``gateway.platforms.bale.extra`` keys: bot_token, allowed_users,
allow_all_users, poll_timeout_s. Env vars override config.yaml: BALE_BOT_TOKEN,
BALE_ALLOWED_USERS, BALE_ALLOW_ALL_USERS, BALE_POLL_TIMEOUT_S, BALE_HOME_CHANNEL.
"""

import asyncio
import contextlib
import hashlib
import logging
import re
import time
from datetime import datetime, UTC
from typing import Any, Dict, Optional

import httpx

from urllib.parse import quote

from agent.ssl_verify import platform_ssl_context
from gateway.platforms._shared import (
    get_scoped_secret as _get_scoped_secret,
    seed_extra_from_env as _seed_extra_from_env,
    send_error,
)
from gateway.platforms.base import (
    BasePlatformAdapter, CachedMedia, SendResult, cache_media_bytes_async,
    transcode_to_ogg_opus)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.config import Platform
from tools.url_safety import is_safe_url

logger = logging.getLogger(__name__)

_BASE_URL = "https://tapi.bale.ai"
_TEXT_LIMIT = 4096          # sendMessage text cap (characters)
_POLL_LIMIT = 100           # getUpdates: max updates per call (1-100, default 100)
_POLL_TIMEOUT_S = 50        # server-side long-poll wait (seconds; Telegram-style, max ~70)

_TRUTHY = {"1", "true", "yes"}

# Inbound Bale media field → (MessageEvent type, default classification kind).
# User-recorded voice notes arrive in the "voice" field (not "audio"); audio is for
# music files uploaded from disk. Both use OGG/Opus containers.
_MEDIA_FIELDS: dict[str, tuple[MessageType, Optional[str]]] = {
    "photo": (MessageType.PHOTO, "image"),
    "video": (MessageType.VIDEO, None),
    "document": (MessageType.DOCUMENT, None),
    "audio": (MessageType.AUDIO, "audio"),
    "voice": (MessageType.VOICE, "audio"),
}
_MEDIA_KIND_TO_TYPE: dict[str, MessageType] = {
    "image": MessageType.PHOTO, "video": MessageType.VIDEO,
    "audio": MessageType.AUDIO, "document": MessageType.DOCUMENT,
}
# Honest multipart content type for outbound voice uploads (post-transcode extension wins).
_VOICE_MIME_BY_EXT: dict[str, str] = {
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".mp3": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
}
# Bale bots may download files up to 20 MB; keep under the budget with headroom.
_MAX_INBOUND_FILE_BYTES = 25 * 1024 * 1024


def _pick_photo_size(payload: Any) -> Optional[dict]:
    """Best photo size from a Bale ``photo`` array (largest that fits the budget,
    else None). Accepts a bare dict for robustness against shape drift."""
    if isinstance(payload, dict):
        return payload if (payload.get("file_id") or "").strip() else None
    candidates = [p for p in payload if isinstance(p, dict) and (p.get("file_id") or "").strip()]
    if not candidates:
        return None
    fits = [p for p in candidates if int(p.get("file_size") or 0) <= _MAX_INBOUND_FILE_BYTES]
    pool = fits or candidates
    return max(pool, key=lambda p: (int(p.get("width") or 0) * int(p.get("height") or 0)))


async def _download_bale_file_bytes(http: httpx.AsyncClient, url: str) -> bytes:
    """Size-capped GET of a Bale file URL; raises on non-2xx / unsafe / oversized payloads.

    ``is_safe_url`` resolves the host and blocks private/internal targets (the token in the
    path is fine — it's a path segment, not userinfo)."""
    if not is_safe_url(url):
        raise ValueError("Blocked unsafe file URL (SSRF protection)")
    async with http.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > _MAX_INBOUND_FILE_BYTES:
                raise ValueError(f"Inbound file too large (>{_MAX_INBOUND_FILE_BYTES} bytes)")
            chunks.append(chunk)
    return b"".join(chunks)


class BaleAPIError(Exception):
    """The Bale API answered ``ok: false`` (or a non-2xx status)."""

    def __init__(self, status: int, code: str, description: str):
        super().__init__(f"Bale API error {status} ({code}): {description}")
        self.status = status
        self.code = code
        self.description = description


def _ms_id() -> str:
    return str(int(time.time() * 1000))


def _env_or_extra(extra: dict, env: str, key: str, default: Any = "") -> Any:
    """Env var overrides config.yaml ``extra``."""
    return _get_scoped_secret(env) or extra.get(key, default)


def _describe_error(error: object) -> str:
    """Secret-safe exception representation for logs and error envelopes (a token in an
    httpx error string must never reach the transcript)."""
    text = "" if error is None else str(error)
    if not text:
        # httpx timeout exceptions stringify to "" — keep the class name so failure lines are non-empty.
        return f"<{type(error).__name__}>" if error is not None else text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        logger.debug("Bale: failed to describe API error", exc_info=True)
        return "<bale error redacted>"


# ── outbound formatting ────────────────────────────────────────────────

def _to_bale_markdown(text: str) -> str:
    """Hermes markdown → Bale's auto-formatted flavour.

    Bale renders EVERY outbound message as markdown: bold is `` *text* `` and italic
    ``_text_`` (each marker pair needs a space before the first and after the last),
    links are native ``[text](url)``, and code has no native form so its markers are
    stripped. Intentional markers are stashed as placeholders during conversion so
    later passes can't re-match them; leftover single ``*``/``_`` characters in plain
    content don't render (Bale requires the surrounding spaces), which is safe to leave.
    """
    if not text:
        return text
    stash: dict[str, str] = {}

    def _hide(value: str) -> str:
        token = f"\x00{len(stash)}\x00"
        stash[token] = value
        return token

    result = re.sub(r"```[^\n]*\n([\s\S]*?)\n?```", lambda m: _hide(m.group(1)), text)  # fenced code → content
    result = re.sub(r"`([^`\n]+)`", lambda m: _hide(m.group(1)), result)             # inline code → content
    result = re.sub(r"\*\*(.+?)\*\*", lambda m: _hide(f" *{m.group(1)}* "), result)  # bold → Bale bold
    result = re.sub(r"__(.+?)__", lambda m: _hide(f" *{m.group(1)}* "), result)      # __bold__ → Bale bold
    result = re.sub(r"\*([^*\n]*?)\*",
                   lambda m: _hide(f"_{m.group(1)}_ "), result)                      # italic → Bale italic
    for token in sorted(stash, reverse=True):
        result = result.replace(token, stash[token])
    return result


def _chunk_text(text: str, limit: int = _TEXT_LIMIT) -> list[str]:
    """Split into ≤``limit``-char chunks at paragraph and space boundaries, keeping the
    whitespace Bale's auto-markdown needs around its markers (a leading/trailing space is part
    of a `` *bold* `` / ``_italic_ `` token); blank lines are dropped."""
    out: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            continue
        work = para
        while len(work) > limit:
            cut = work.rfind(" ", 0, limit)
            if cut <= limit // 3:
                cut = limit
            out.append(work[:cut].rstrip())
            work = work[cut:].lstrip()
        if work.strip():
            out.append(work)
    return out or [""]


# ── adapter ────────────────────────────────────────────────────────────

class BaleAdapter(BasePlatformAdapter):
    """Long-polling HTTP adapter for the Bale Bot API (tapi.bale.ai)."""

    serial_receive_loop = True  # updates are processed one at a time in the poll loop

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("bale"))
        extra = getattr(config, "extra", {}) or {}
        self._token = str(_env_or_extra(extra, "BALE_BOT_TOKEN", "bot_token")).strip()
        allow_all_raw = _env_or_extra(extra, "BALE_ALLOW_ALL_USERS", "allow_all_users", "")
        self._allow_all = bool(allow_all_raw) and str(allow_all_raw).strip().lower() in _TRUTHY
        allowed_raw = extra.get("allowed_users") or []
        if isinstance(allowed_raw, str):
            allowed_raw = [u.strip() for u in allowed_raw.split(",") if u.strip()]
        self._allowed_users: set[str] = {str(u) for u in allowed_raw}
        timeout_raw = _env_or_extra(extra, "BALE_POLL_TIMEOUT_S", "poll_timeout_s", _POLL_TIMEOUT_S)
        try:
            self._poll_timeout_s = max(5, min(int(timeout_raw), 70))
        except (TypeError, ValueError):
            self._poll_timeout_s = _POLL_TIMEOUT_S
        self._bot_id: Optional[int] = None
        self._offset = 0
        self._http: Optional[httpx.AsyncClient] = None
        self._recv_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "Bale"

    def _fail(self, code: str, message: str, *, retryable: bool) -> bool:
        self._set_fatal_error(code, message, retryable=retryable)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Verify the token via getMe and start the long-polling loop."""
        if not self._token:
            return self._fail("config_missing", "BALE_BOT_TOKEN must be set (from @botfather in Bale)",
                              retryable=False)
        fingerprint = hashlib.sha256(self._token.encode()).hexdigest()[:12]
        # One token == one bot; stop a second profile/gateway from polling the same identity.
        if not self._acquire_platform_lock("bale", f"token:{fingerprint}", "Bale bot token"):
            return False
        try:
            self._http = httpx.AsyncClient(timeout=self._poll_timeout_s + 30, verify=platform_ssl_context())
            # getMe puts the user object directly in the body's ``result`` (Telegram shape);
            # _api_call already returns that field, so no second unwrap.
            me = await self._api_call("getMe")
            user = me if isinstance(me, dict) else {}
            self._bot_id = int(user.get("id", -1))
        except BaleAPIError as e:
            if e.status in (401, 403):
                logger.error("Bale: token rejected (%s)", e.code)
                return self._fail("token_invalid", f"Bale bot token was rejected ({e.description})",
                                  retryable=False)
            logger.error("Bale: getMe failed — %s", _describe_error(e))
            await self._close_http()
            return self._fail("connect_failed", _describe_error(e), retryable=True)
        except Exception as e:
            logger.exception("Bale: getMe failed — %s", _describe_error(e))
            await self._close_http()
            return self._fail("connect_failed", _describe_error(e), retryable=True)
        self._recv_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info("Bale: connected as @%s (id %s)", user.get("username") or "?", self._bot_id)
        return True

    async def disconnect(self) -> None:
        with contextlib.suppress(Exception):
            self._release_platform_lock()
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._recv_task
            self._recv_task = None
        await self._close_http()
        self._mark_disconnected()

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if not self.is_connected or self._http is None:
            return SendResult(success=False, error="Not connected")
        chunks = _chunk_text(_to_bale_markdown(content))
        last_id = ""
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if reply_to and index == 0:
                with contextlib.suppress(ValueError):
                    payload["reply_to_message_id"] = int(reply_to)
            # sendMessage returns the Message itself as ``result`` (Telegram shape).
            sent = await self._api_call("sendMessage", payload)
            last_id = str(sent.get("message_id", ""))
        return SendResult(success=True, message_id=last_id or _ms_id())

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if self._http is None:
            return
        with contextlib.suppress(Exception):
            await self._api_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._http is None:
            return SendResult(success=False, error="Not connected")
        payload: dict[str, Any] = {"chat_id": chat_id, "photo": image_url}
        if caption:
            payload["caption"] = _to_bale_markdown(caption)[:_TEXT_LIMIT]
        try:
            sent = await self._api_call("sendPhoto", payload)
            return SendResult(success=True, message_id=str(sent.get("message_id", "")) or _ms_id())
        except BaleAPIError as e:
            return SendResult(success=False, error=_describe_error(e))

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        """Send an audio file as a native Bale voice message bubble.

        Uses the Bale Bot API ``sendVoice`` endpoint (Telegram-style). Native voice bubbles
        require Ogg/Opus: non-Opus audio is transcoded with ffmpeg when available (best
        effort); otherwise the original container is uploaded with its honest content type.
        Caption is supported (up to 4096 chars, markdown-formatted).
        """
        if self._http is None:
            return SendResult(success=False, error="Not connected")
        import os as _os
        if not _os.path.isfile(audio_path):
            return SendResult(success=False, error=f"Audio file not found: {audio_path}")

        try:
            # Voice bubbles render broken from MP3/WAV bytes; transcode to Ogg/Opus first.
            if _os.path.splitext(str(audio_path))[1].lower() not in (".ogg", ".opus"):
                transcoded = await asyncio.to_thread(transcode_to_ogg_opus, audio_path)
                if transcoded:
                    audio_path = transcoded
                else:
                    logger.warning(
                        "Bale: voice transcode unavailable for %s — uploading original container",
                        _os.path.basename(audio_path))
            data: dict[str, Any] = {
                "chat_id": chat_id,
            }
            if reply_to:
                with contextlib.suppress(ValueError):
                    data["reply_to_message_id"] = int(reply_to)
            if caption:
                data["caption"] = _to_bale_markdown(caption)[:_TEXT_LIMIT]

            # Bale sendVoice expects a multipart upload (file field named "voice"); the content
            # type stays honest to the file's real container.
            with open(audio_path, "rb") as audio_file:
                files = {
                    "voice": (_os.path.basename(audio_path), audio_file,
                            _VOICE_MIME_BY_EXT.get(_os.path.splitext(str(audio_path))[1].lower(),
                                                "audio/ogg"))
                }
                try:
                    resp = await self._http.post(
                        f"{_BASE_URL}/bot{self._token}/sendVoice",
                        files=files,
                        data=data)
                    body = resp.json()
                except ValueError as e:
                    return SendResult(success=False, error=f"Non-JSON response ({resp.status_code}) from Bale sendVoice")

                if resp.status_code >= 400 or not body.get("ok"):
                    err_desc = str(body.get("description", body.get("result", "")))
                    logger.warning("Bale: sendVoice failed — %s", _describe_error(BaleAPIError(resp.status_code, "send_voice_failed", err_desc)))
                    return SendResult(success=False, error=err_desc)

                result = body.get("result") or {}
                msg_id = str(result.get("message_id", "")) or _ms_id()
                logger.info("Bale: sent voice message %s to chat %s", msg_id, chat_id)
                return SendResult(success=True, message_id=msg_id)
        except Exception as e:
            logger.warning("Bale: send_voice raised — %s", _describe_error(e), exc_info=True)
            return SendResult(success=False, error=_describe_error(e))

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Auto-TTS playback: send voice bubble (Bale supports this natively)."""
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        # getChat returns ChatFullInfo directly as ``result``.
        info = await self._api_call("getChat", {"chat_id": chat_id}) or {}
        name = info.get("title") or info.get("username") or info.get("first_name") or chat_id
        return {"name": name, "type": "dm" if (info.get("type") or "") == "private" else "group",
                "chat_id": str(info.get("id", chat_id))}

    # ── transport ───────────────────────────────────────────────────────

    async def _api_call(self, method: str, payload: Optional[dict] = None) -> Any:
        """Call ``https://tapi.bale.ai/bot<TOKEN>/<method>``; returns the body's ``result`` field.

        Raises :class:`BaleAPIError` on ``ok: false`` or a non-2xx status."""
        url = f"{_BASE_URL}/bot{self._token}/{method}"
        resp = await (self._http.post(url, json=payload) if payload is not None else self._http.get(url))
        try:
            body = resp.json()
        except ValueError as e:
            raise BaleAPIError(resp.status_code, "bad_response", f"non-JSON response ({resp.status_code})") from e
        if resp.status_code >= 400 or not body.get("ok"):
            raise BaleAPIError(resp.status_code, str(body.get("error_code", "")),
                               str(body.get("description", body.get("result", ""))))
        return body.get("result")

    async def _close_http(self) -> None:
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None

    # ── inbound ────────────────────────────────────────────────────────

    async def _fetch_media_for_message(self, message: dict[str, Any]) -> tuple[Optional[CachedMedia], Optional[str]]:
        """Resolve a Bale media field to cached local media.

        Returns ``(CachedMedia or None, note)`` where *note* is a short user-facing
        fallback line when a present field could not be processed (missing file_id,
        oversized, download failure). ``note`` is ``None`` only when no media field was
        attempted at all.
        """
        for field, (_type, default_kind) in _MEDIA_FIELDS.items():
            payload = message.get(field)
            if not payload:
                continue
            file_id, filename, mime_type, kind_hint = None, "", "", default_kind
            if field == "photo":
                size = _pick_photo_size(payload)
                if not size:
                    return None, f"[photo received but could not be processed]"
                file_id = str(size.get("file_id") or "")
                kind_hint = "image"
            elif isinstance(payload, dict):
                file_id = str(payload.get("file_id") or "")
                filename = str(payload.get("file_name") or "")
                mime_type = str(payload.get("mime_type") or "")
            if not file_id:
                return None, f"[{field} received but could not be processed]"
            try:
                meta = await self._api_call("getFile", {"file_id": file_id})
            except BaleAPIError as e:
                logger.warning("Bale: getFile failed for %s: %s", field, _describe_error(e))
                return None, f"[{field} received but could not be processed]"
            if not isinstance(meta, dict):
                logger.warning("Bale: unexpected getFile response for %s", field)
                return None, f"[{field} received but could not be processed]"
            file_path = str((meta.get("file_path") or "").strip())
            if not file_path:
                logger.warning("Bale: getFile returned no file_path for %s", field)
                return None, f"[{field} received but could not be processed]"
            # The bot token belongs in the path (API shape); keep any path separators intact.
            url = f"{_BASE_URL}/file/bot{self._token}/{quote(file_path, safe='/')}"
            try:
                data = await _download_bale_file_bytes(self._http, url)
            except Exception as e:
                logger.warning("Bale: download failed for %s file: %s", field, _describe_error(e))
                return None, f"[{field} received but could not be processed]"
            try:
                media = await cache_media_bytes_async(
                    data, filename=filename, mime_type=mime_type, default_kind=kind_hint)
            except Exception as e:
                logger.warning("Bale: failed to cache %s file: %s", field, _describe_error(e))
                return None, f"[{field} received but could not be processed]"
            if media is None:
                logger.warning("Bale: cached %s rejected (not a valid image)", field)
                return None, f"[{field} received but could not be processed]"
            logger.info("Bale: inbound %s cached (%d bytes) for chat %s",
                       media.kind, len(data), message.get("chat", {}).get("id"))
            return media, None
        return None, None

    async def _poll_loop(self) -> None:
        """Long-poll getUpdates; the offset advances past each handled update so it is not redelivered."""
        backoff = 1.0
        try:
            while True:
                try:
                    updates = await self._api_call("getUpdates", {
                        "offset": self._offset, "limit": _POLL_LIMIT, "timeout": self._poll_timeout_s})
                    if updates:
                        backoff = 1.0
                        for update in updates:
                            try:
                                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                                await self._handle_update(update)
                            except Exception as e:
                                logger.warning("Bale: error handling update %s — %s",
                                               update.get("update_id"), _describe_error(e), exc_info=True)
                except BaleAPIError as e:
                    if e.status == 409:
                        # Another getUpdates session owns this token (two gateways on one bot).
                        logger.error("Bale: 409 conflict — another poller is using this bot token")
                        self._set_fatal_error("conflict", "Another process is long-polling this Bale bot token",
                                              retryable=False)
                        await self._notify_fatal_error()
                        return
                    logger.warning("Bale: getUpdates API error (%s): %s — retrying", e.code, e.description)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("Bale: getUpdates failed — %s — retrying", _describe_error(e), exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        finally:
            if self.is_connected:
                self._set_fatal_error("connection_lost", "Bale poll loop terminated unexpectedly", retryable=True)
                with contextlib.suppress(Exception):
                    await self._notify_fatal_error()

    def _reply_context(self, message: dict) -> tuple[Optional[str], Optional[str]]:
        """Extract (reply_to_id, reply_to_text) from a Bale message.

        Bale's API mirrors Telegram's — ``reply_to_message`` is an object with
        ``message_id``, ``text``/``caption``. No partial-quote object exists in
        Bale's API (unlike PTB's TextQuote), so we return the full replied-to text."""
        if not message.get("reply_to_message"):
            return None, None
        reply = message["reply_to_message"]
        reply_id = str(reply.get("message_id", ""))
        text = reply.get("text") or reply.get("caption") or None
        # Bale API returns rich media as dict objects (voice, photo, video...) — skip those.
        if isinstance(text, dict):
            return None, None
        return reply_id, text

    async def _handle_update(self, update: dict) -> None:
        """Dispatch one Bale Update (``message`` / ``callback_query`` / ``pre_checkout_query``)."""
        message = update.get("message")
        if not message:
            # Inline-button callbacks: ack so the client's spinner clears. Payments are v2.
            if "callback_query" in update:
                with contextlib.suppress(Exception):
                    await self._api_call("answerCallbackQuery", {
                        "callback_query_id": str(update["callback_query"].get("id")), "text": "✓"})
            return
        sender = message.get("from") or {}
        if sender and int(sender.get("id", -1)) == self._bot_id:
            return  # our own message
        chat = message.get("chat") or {}
        text = (message.get("text") or "").strip()
        caption = (message.get("caption") or "").strip()
        media: Optional[CachedMedia] = None
        media_note: Optional[str] = None
        # Detect message type from original update fields (voice vs audio distinction matters for auto-TTS).
        detected_type = MessageType.TEXT
        for field_name, (_type_hint, _kind) in _MEDIA_FIELDS.items():
            if message.get(field_name):
                detected_type = _type_hint
                break
        if any(message.get(f) for f in _MEDIA_FIELDS):
            media, media_note = await self._fetch_media_for_message(message)
        user_id = str(sender.get("id", ""))
        if self._allowed_users and not self._allow_all and user_id not in self._allowed_users:
            logger.debug("Bale: ignoring message from unauthorized user %s", user_id)
            return
        chat_type = "dm" if (chat.get("type") or "") == "private" else "group"
        source = self.build_source(
            chat_id=str(chat.get("id")),
            chat_name=chat.get("title") or sender.get("first_name") or str(chat.get("id")),
            chat_type=chat_type,
            user_id=user_id,
            user_name=sender.get("username") or sender.get("first_name") or user_id)
        timestamp = datetime.fromtimestamp(int(message.get("date", 0)), tz=UTC)
        event_text = text or caption
        if media is None and not event_text:
            logger.debug("Bale: skipping non-text update from %s", sender.get("id"))
            return
        reply_to_id, reply_to_text = self._reply_context(message)
        await self.handle_message(MessageEvent(
            text=event_text or (media_note or "[message received]"),
            message_type=detected_type,
            source=source, raw_message=message, message_id=str(message.get("message_id")),
            timestamp=timestamp,
            media_urls=[media.path] if media else [],
            media_types=[media.media_type] if media else [],
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text))


# ── plugin registration helpers ────────────────────────────────────────

def check_requirements() -> bool:
    """Bale is configured iff the bot token is present."""
    return bool(_get_scoped_secret("BALE_BOT_TOKEN"))


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_env_or_extra(extra, "BALE_BOT_TOKEN", "bot_token"))


def is_connected(config) -> bool:
    """Check whether Bale is configured (env or config.yaml)."""
    return validate_config(config)


def interactive_setup() -> None:
    """``hermes gateway setup`` flow (lazy hermes_cli imports keep the plugin importable outside the CLI)."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value, print_header, print_info,
        print_warning, print_success)
    from hermes_cli.setup_platforms import declines_reconfigure

    def info(*lines: str) -> None:
        for line in lines:
            print_info(line)

    def _required(label: str, env: str, default: str, what: str) -> bool:
        value = prompt(label, default=default)
        if not value:
            print_warning(f"{what} is required — skipping Bale setup")
            return False
        save_env_value(env, value.strip())
        return True

    print_header("Bale")
    existing_token = get_env_value("BALE_BOT_TOKEN")
    if declines_reconfigure("Bale", "Reconfigure Bale?", "BALE_BOT_TOKEN"):
        return
    info("Connect Hermes to the Iranian messaging app Bale (bale.ai).",
         "   Create your bot by chatting with @botfather inside the Bale app — it hands you a token.",
         "   The Bot API mirrors Telegram's, so no extra packages are needed.")
    print()
    if not _required("Bale bot token (from @botfather in the Bale app)", "BALE_BOT_TOKEN",
                     existing_token or "", "Token"):
        return
    print()
    info("🔒 Access control: restrict who can message the bot")
    if prompt_yes_no("Allow all users to talk to the bot?", False):
        save_env_value("BALE_ALLOW_ALL_USERS", "true")
        save_env_value("BALE_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — any Bale user can command the bot.")
    else:
        save_env_value("BALE_ALLOW_ALL_USERS", "false")
        allowed = prompt("Allowed user IDs (comma-separated, leave empty to deny everyone)",
                         default=get_env_value("BALE_ALLOWED_USERS") or "")
        if allowed:
            save_env_value("BALE_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("BALE_ALLOWED_USERS", "")
            print_info("No users allowed — the bot will ignore all messages until you add user IDs.")
    home = prompt("Home channel ID for cron / notification delivery (leave blank to skip)",
                  default=get_env_value("BALE_HOME_CHANNEL") or "")
    save_env_value("BALE_HOME_CHANNEL", home)
    print()
    print_success("Bale configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def _env_enablement() -> dict | None:
    """``env_enablement_fn``: seed ``PlatformConfig.extra`` from the profile's env BEFORE adapter
    construction; ``None`` when the bot token isn't set. The home channel seeds ``home_channel``
    so cron ``deliver=bale`` has a target without extra config."""
    token = str(_get_scoped_secret("BALE_BOT_TOKEN", "") or "").strip()
    if not token:
        return None
    seed = _seed_extra_from_env((
        ("BALE_ALLOWED_USERS", "allowed_users", None),
        ("BALE_ALLOW_ALL_USERS", "allow_all_users", lambda v: str(v).strip().lower() in _TRUTHY),
        ("BALE_POLL_TIMEOUT_S", "poll_timeout_s", int),
    ), home_env="BALE_HOME_CHANNEL")
    return {"bot_token": token, **seed}


def _sa_error(detail: str) -> dict[str, Any]:
    return {"success": False, **send_error(f"Bale standalone send: {detail}")}


async def _standalone_send(pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
                           media_files: Optional[list[str]] = None, force_document: bool = False) -> dict[str, Any]:
    """Out-of-process cron delivery: fresh client POSTing sendMessage (and sendPhoto per image URL)."""
    extra = getattr(pconfig, "extra", {}) or {}
    token = str(_env_or_extra(extra, "BALE_BOT_TOKEN", "bot_token") or "").strip()
    if not token:
        return _sa_error("BALE_BOT_TOKEN must be configured")
    text = _to_bale_markdown(message)

    async def _post(client: httpx.AsyncClient, method: str, payload: dict):
        resp = await client.post(f"{_BASE_URL}/bot{token}/{method}", json=payload)
        try:
            body = resp.json()
        except ValueError:
            return False, f"non-JSON response ({resp.status_code})"
        if resp.status_code >= 400 or not body.get("ok"):
            return False, str(body.get("description", body.get("result", "")))
        return True, body

    try:
        async with httpx.AsyncClient(timeout=60, verify=platform_ssl_context()) as client:
            last_id = ""
            for chunk in _chunk_text(text):
                ok, detail_or_body = await _post(client, "sendMessage", {"chat_id": chat_id, "text": chunk})
                if not ok:
                    return _sa_error(detail_or_body)
                last_id = str(((detail_or_body or {}).get("result") or {}).get("message_id", ""))
            for url in media_files or []:
                ok, detail_or_body = await _post(client, "sendPhoto", {"chat_id": chat_id, "photo": url})
                if not ok:
                    return _sa_error(detail_or_body)
            return {"success": True, "message_id": last_id or _ms_id()}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Bale standalone send raised", exc_info=True)
        return send_error(f"Bale standalone send failed: {_describe_error(e)}")


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="bale",
        dashboard_description="Talk to Hermes through Bale (the Iranian messenger) via its Telegram-style Bot API.",
        docs_url="https://docs.bale.ai",
        label="Bale",
        adapter_factory=BaleAdapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BALE_BOT_TOKEN"],
        install_hint="No extra packages needed (httpx is a Hermes dependency)",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,  # env-only setups show in gateway status
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,  # cron running separately from the gateway
        allowed_users_env="BALE_ALLOWED_USERS",
        allow_all_env="BALE_ALLOW_ALL_USERS",
        max_message_length=_TEXT_LIMIT,
        emoji="💬",
        pii_safe=True,  # the Bot API does not expose phone numbers
        platform_hint=(
            "You are chatting via Bale (the Iranian messenger). Your messages are rendered as "
            "markdown: **bold**, _italic_ and [links](url) come through natively. Messages are "
            "limited to 4096 characters each (longer replies are split automatically). Keep "
            "responses concise and conversational."))
