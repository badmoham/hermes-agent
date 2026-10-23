"""Tests for the Bale platform adapter plugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/bale/adapter.py under a unique module name (plugin_adapter_bale)
# so it cannot collide with other plugin adapters loaded by sibling tests in the same process.
_bale_mod = load_plugin_adapter("bale")

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
