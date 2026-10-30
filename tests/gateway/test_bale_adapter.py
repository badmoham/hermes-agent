"""Tests for the Bale platform adapter plugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/bale/adapter.py under a unique module name (plugin_adapter_bale)
# so it cannot collide with other plugin adapters loaded by sibling tests in the same process.
_bale_mod = load_plugin_adapter("bale")

from gateway.platforms.base import CachedMedia  # noqa: F401 - constructed in media tests
from gateway.platforms.event import MessageType

_to_bale_markdown = _bale_mod._to_bale_markdown
_chunk_text = _bale_mod._chunk_text
BaleAdapter = _bale_mod.BaleAdapter
check_requirements = _bale_mod.check_requirements
validate_config = _bale_mod.validate_config
register = _bale_mod.register
_standalone_send = _bale_mod._standalone_send
is_connected = _bale_mod.is_connected
_env_enablement = _bale_mod._env_enablement


class TestBaleMarkdownConversion:

    def test_bold_gains_required_surrounding_spaces(self):
        assert _to_bale_markdown("Hello **world**") == "Hello  *world* "

    def test_double_underscore_bold_converts_to_astrix_style(self):
        assert _to_bale_markdown("__bold__ here") == " *bold*  here"

    def test_italic_converts_to_underscore_style(self):
        assert _to_bale_markdown("a *bit* b") == "a _bit_  b"

    def test_code_fences_are_stripped_content_kept(self):
        assert _to_bale_markdown("x\n```py\nprint(1)\n```\ny") == "x\nprint(1)\ny"

    def test_inline_code_markers_removed(self):
        assert _to_bale_markdown("run `code` now") == "run code now"

    def test_links_pass_through_unchanged(self):
        assert _to_bale_markdown("see [docs](https://bale.ai)") == "see [docs](https://bale.ai)"

    def test_plain_text_with_lone_markers_is_untouched(self):
        # Bale needs surrounding spaces to render markers, so stray single * / _ are inert.
        assert _to_bale_markdown("5 * 3 = 15 and a_b stays") == "5 * 3 = 15 and a_b stays"


class TestBaleChunking:

    def test_short_text_is_single_chunk(self):
        assert _chunk_text("hello world") == ["hello world"]

    def test_empty_text_yields_placeholder(self):
        assert _chunk_text("") == [""]

    def test_long_text_splits_within_limit(self):
        long_text = ("word " * 2000).strip()
        chunks = _chunk_text(long_text)
        assert len(chunks) >= 2
        assert all(len(chunk) <= 4096 for chunk in chunks)

    def test_unsplittable_run_hard_splits_at_limit(self):
        chunks = _chunk_text("a" * 5000)
        assert [len(c) for c in chunks] == [4096, 904]


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


@pytest.fixture
def adapter(monkeypatch):
    for key in ("BALE_BOT_TOKEN", "BALE_ALLOWED_USERS", "BALE_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    from gateway.config import PlatformConfig

    a = BaleAdapter(PlatformConfig(enabled=True, extra={"bot_token": "123:abc"}))
    a._mark_connected()
    http = MagicMock()
    # sendMessage/sendPhoto return the Message itself as ``result`` (Telegram shape).
    http.post = AsyncMock(return_value=_FakeResponse({"ok": True, "result": {"message_id": 42}}))
    a._http = http
    return a


class TestBaleSend:

    @pytest.mark.asyncio
    async def test_send_posts_formatted_text(self, adapter):
        result = await adapter.send("1001", "Hello **world**")
        assert result.success is True
        assert result.message_id == "42"
        args, kwargs = adapter._http.post.call_args
        assert args[0].endswith("/bot123:abc/sendMessage")
        assert kwargs["json"] == {"chat_id": "1001", "text": "Hello  *world* "}

    @pytest.mark.asyncio
    async def test_send_splits_long_content(self, adapter):
        long_text = ("word " * 2000).strip()
        result = await adapter.send("1001", long_text)
        assert result.success is True
        assert adapter._http.post.call_count == len(_chunk_text(long_text))

    @pytest.mark.asyncio
    async def test_send_typing_posts_chat_action(self, adapter):
        await adapter.send_typing("1001")
        args, kwargs = adapter._http.post.call_args
        assert args[0].endswith("/bot123:abc/sendChatAction")
        assert kwargs["json"] == {"chat_id": "1001", "action": "typing"}


class TestBaleInboundUpdates:

    @pytest.mark.asyncio
    async def test_dispatches_authorized_text_message(self, adapter):
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        update = {"update_id": 7, "message": {
            "message_id": 11, "text": "hi", "date": 1_700_000_000,
            "from": {"id": 55, "first_name": "Ali"},
            "chat": {"id": 1001, "type": "private"}}}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.text == "hi"
        assert event.source.user_id == "55"
        assert event.message_id == "11"

    @pytest.mark.asyncio
    async def test_ignores_own_and_unauthorized_messages(self, adapter):
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        await adapter._handle_update({"update_id": 7, "message": {
            "message_id": 11, "text": "self", "from": {"id": 999}, "chat": {"id": 1}}})
        adapter._allowed_users = {"55"}
        await adapter._handle_update({"update_id": 8, "message": {
            "message_id": 12, "text": "stranger", "from": {"id": 66}, "chat": {"id": 1}}})
        assert handle.await_count == 0

    @pytest.mark.asyncio
    async def test_callback_query_is_acked(self, adapter):
        adapter._http.post = AsyncMock(return_value=_FakeResponse({"ok": True, "result": True}))
        await adapter._handle_update({
            "update_id": 9, "callback_query": {"id": "cq-1", "from": {"id": 55}, "message": {}}})
        args, kwargs = adapter._http.post.call_args
        assert args[0].endswith("/bot123:abc/answerCallbackQuery")
        assert kwargs["json"] == {"callback_query_id": "cq-1", "text": "✓"}


    @pytest.mark.asyncio
    async def test_photo_message_caches_and_announces(self, adapter, monkeypatch):
        """A photo update with caption becomes a PHOTO event carrying the cached path."""
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        # getFile returns File{file_path}; the download is intercepted by the patched helper.
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "photos/1/a.jpg"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes",
                            AsyncMock(return_value=b"\xff\xd8\xff\x00fake-jpeg"))
        monkeypatch.setattr(
            _bale_mod, "cache_media_bytes_async",
            AsyncMock(return_value=CachedMedia("/tmp/fake.jpg", "image/jpeg", "image", "a.jpg")))
        update = {"update_id": 20, "message": {
            "message_id": 31, "caption": "sunrise", "date": 1_700_000_500,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "photo": [
                {"file_id": "small", "width": 320, "height": 240, "file_size": 50_000},
                {"file_id": "big", "width": 1600, "height": 1200, "file_size": 900_000},
            ],
        }}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.PHOTO
        assert event.text == "sunrise"
        assert event.media_urls == ["/tmp/fake.jpg"]
        assert event.media_types == ["image/jpeg"]
        # The larger, in-budget size wins over the thumbnail.
        assert adapter._api_call.call_args.args[1] == {"file_id": "big"}

    @pytest.mark.asyncio
    async def test_voice_update_becomes_voicemessage_event(self, adapter, monkeypatch):
        """Inbound voice note (voice field) must arrive as MessageType.VOICE so auto-TTS works."""
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "voice/1/a.ogg"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes", AsyncMock(return_value=b"fake-ogg-data"))
        monkeypatch.setattr(
            _bale_mod, "cache_media_bytes_async",
            AsyncMock(return_value=CachedMedia("/tmp/fake.ogg", "audio/ogg", "audio", "voice.ogg")))
        update = {"update_id": 30, "message": {
            "message_id": 41, "date": 1_700_001_000,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "voice": {"file_id": "v-abc", "duration": 7}}}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.VOICE, f"Expected VOICE type, got {event.message_type}"

    @pytest.mark.asyncio
    async def test_audio_file_is_distinct_from_voice_note(self, adapter, monkeypatch):
        """Audio file uploads (music) are different from user-recorded voice notes."""
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "audio/1/a.mp3"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes", AsyncMock(return_value=b"fake-mp3-data"))
        monkeypatch.setattr(
            _bale_mod, "cache_media_bytes_async",
            AsyncMock(return_value=CachedMedia("/tmp/fake.mp3", "audio/mpeg", "audio", "song.mp3")))
        update_audio = {"update_id": 31, "message": {
            "message_id": 42, "date": 1_700_001_500,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "audio": {"file_id": "a-xyz", "file_name": "song.mp3", "duration": 240}}}
        await adapter._handle_update(update_audio)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.AUDIO, f"Expected AUDIO type for music file, got {event.message_type}"

    @pytest.mark.asyncio
    async def test_voice_with_caption_passes_text(self, adapter, monkeypatch):
        """Voice note with caption carries both the audio and text content."""
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "voice/1/b.ogg"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes", AsyncMock(return_value=b"fake-ogg-data2"))
        monkeypatch.setattr(
            _bale_mod, "cache_media_bytes_async",
            AsyncMock(return_value=CachedMedia("/tmp/fake.ogg2", "audio/ogg", "audio", "voice.ogg")))
        update = {"update_id": 32, "message": {
            "message_id": 43, "caption": "hi there", "date": 1_700_002_000,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "voice": {"file_id": "v-def", "duration": 3}}}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.VOICE
        assert event.text == "hi there"

    @pytest.mark.asyncio
    async def test_photo_download_failure_falls_back_to_note(self, adapter, monkeypatch):
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "photos/1/b.jpg"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        async def boom(http, url):  # noqa: ARG001 - unused in the test stub
            raise RuntimeError("upstream 5xx")
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes", AsyncMock(side_effect=boom))
        update = {"update_id": 21, "message": {
            "message_id": 32, "date": 1_700_000_600,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "photo": [{"file_id": "big", "width": 800, "height": 600, "file_size": 700_000}],
        }}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.TEXT
        assert event.text == "[photo received but could not be processed]"
        assert not event.media_urls

    @pytest.mark.asyncio
    async def test_document_message_uses_file_name_and_type(self, adapter, monkeypatch):
        handle = AsyncMock()
        adapter.handle_message = handle
        adapter._bot_id = 999
        async def fake_api_call(method, payload=None):
            if method == "getFile":
                return {"file_path": "docs/2/report.pdf"}
            raise AssertionError(f"unexpected {method}")
        adapter._api_call = fake_api_call
        monkeypatch.setattr(_bale_mod, "_download_bale_file_bytes",
                            AsyncMock(return_value=b"%PDF-1.7 fake"))
        async def fake_cache(data, **kwargs):
            return CachedMedia("/tmp/fake.pdf", "application/pdf", "document", "report.pdf")
        monkeypatch.setattr(_bale_mod, "cache_media_bytes_async", AsyncMock(return_value=fake_cache))
        update = {"update_id": 22, "message": {
            "message_id": 33, "caption": "see page 4", "date": 1_700_000_700,
            "from": {"id": 55}, "chat": {"id": 1001, "type": "private"},
            "document": {"file_id": "doc-2", "file_name": "report.pdf",
                        "mime_type": "application/pdf", "file_size": 80_000},
        }}
        await adapter._handle_update(update)
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.message_type is MessageType.DOCUMENT
        assert event.text == "see page 4"
        assert event.media_urls == ["/tmp/fake.pdf"]
        assert event.media_types == ["application/pdf"]


class TestBalePollConflict:

    @pytest.mark.asyncio
    async def test_409_conflict_stops_polling_without_retry(self, adapter):
        """Two gateways on one token: the 409 is terminal (not retryable) and marks the platform fatal."""
        adapter._http.post = AsyncMock(
            side_effect=_bale_mod.BaleAPIError(409, "conflict", "terminated by other getUpdates request"))
        notify = AsyncMock()
        adapter._notify_fatal_error = notify
        await adapter._poll_loop()
        assert adapter._fatal_error_code == "conflict"
        assert adapter.is_connected is False
        assert adapter._http.post.call_count == 1


class TestBaleRegistration:

    def test_register_wires_core_hooks(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "bale"
        assert kwargs["required_env"] == ["BALE_BOT_TOKEN"]
        assert kwargs["cron_deliver_env_var"] == "BALE_HOME_CHANNEL"
        assert kwargs["adapter_factory"] is BaleAdapter

    def test_check_requirements_requires_token(self, monkeypatch):
        monkeypatch.setattr(_bale_mod, "_get_scoped_secret", lambda name, default=None, **k: "tok")
        assert check_requirements() is True
        monkeypatch.setattr(_bale_mod, "_get_scoped_secret", lambda name, default=None, **k: "")
        assert check_requirements() is False

    def test_env_enablement_requires_token(self, monkeypatch):
        monkeypatch.setattr(_bale_mod, "_get_scoped_secret", lambda name, default=None, **k: "123:abc")
        seeded = _env_enablement()
        assert seeded["bot_token"] == "123:abc"
        monkeypatch.setattr(_bale_mod, "_get_scoped_secret", lambda name, default=None, **k: "")
        assert _env_enablement() is None

    def test_standalone_send_requires_token(self, monkeypatch):
        import asyncio

        from gateway.config import PlatformConfig

        monkeypatch.setattr(_bale_mod, "_get_scoped_secret", lambda name, default=None, **k: "")
        result = asyncio.run(_standalone_send(
            PlatformConfig(enabled=True, extra={}), "1001", "hello"))
        assert result["success"] is False
