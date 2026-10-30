"""Bale ``send_voice`` delivers native voice bubbles only as Ogg/Opus.

The Bale Bot API's ``sendVoice`` endpoint (Telegram-style) renders voice bubbles
broken from MP3/WAV bytes, so the adapter must transcode non-Opus input with
ffmpeg when available — the same contract Telegram and Matrix enforce — and fall
back to uploading the original container with an honest content type otherwise.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/bale/adapter.py under a unique module name (plugin_adapter_bale)
# so it cannot collide with other plugin adapters loaded by sibling tests in the same process.
_bale_mod = load_plugin_adapter("bale")

BaleAdapter = _bale_mod.BaleAdapter


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    for key in ("BALE_BOT_TOKEN", "BALE_ALLOWED_USERS", "BALE_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    from gateway.config import PlatformConfig

    a = BaleAdapter(PlatformConfig(enabled=True, extra={"bot_token": "123:abc"}))
    a._mark_connected()
    http = MagicMock()
    # sendVoice returns the Message itself as ``result`` (Telegram shape).
    http.post = AsyncMock(return_value=_FakeResponse({"ok": True, "result": {"message_id": 42}}))
    a._http = http
    return a


def _uploaded_voice(adapter) -> tuple[str, str]:
    """The multipart ``voice`` file field: (basename, content type)."""
    files = adapter._http.post.call_args.kwargs["files"]
    filename, _file_obj, mime = files["voice"]
    return filename, mime


def test_mp3_voice_attachment_is_transcoded_to_ogg_before_upload(adapter, tmp_path):
    """The dispatcher shape (base.py ``_send_one`` passes is_voice) must upload an
    Ogg/Opus bubble, not raw MP3 bytes."""
    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"\xff\xfbfake-mp3")
    ogg = tmp_path / "clip.ogg"
    ogg.write_bytes(b"OggS fake-opus")

    with patch.object(_bale_mod, "transcode_to_ogg_opus",
                     MagicMock(return_value=str(ogg))) as spy:
        result = asyncio.run(adapter.send_voice(
            chat_id="1001", audio_path=str(mp3), metadata=None, is_voice=True))

    assert result.success is True
    spy.assert_called_once_with(str(mp3))
    filename, mime = _uploaded_voice(adapter)
    assert (filename, mime) == ("clip.ogg", "audio/ogg")


def test_ogg_input_is_uploaded_as_is_without_transcode(adapter, tmp_path):
    ogg = tmp_path / "note.ogg"
    ogg.write_bytes(b"OggS fake-opus")

    with patch.object(_bale_mod, "transcode_to_ogg_opus", MagicMock()) as spy:
        result = asyncio.run(adapter.send_voice(chat_id="1001", audio_path=str(ogg)))

    assert result.success is True
    spy.assert_not_called()
    filename, mime = _uploaded_voice(adapter)
    assert (filename, mime) == ("note.ogg", "audio/ogg")


def test_transcode_failure_falls_back_to_honest_container(adapter, tmp_path):
    """Without ffmpeg the upload still succeeds but carries the file's REAL
    content type — no mislabelled audio/ogg bytes."""
    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"\xff\xfbfake-mp3")

    with patch.object(_bale_mod, "transcode_to_ogg_opus", MagicMock(return_value=None)):
        result = asyncio.run(adapter.send_voice(
            chat_id="1001", audio_path=str(mp3), metadata=None, is_voice=True))

    assert result.success is True
    filename, mime = _uploaded_voice(adapter)
    assert (filename, mime) == ("clip.mp3", "audio/mpeg")


def test_wav_fallback_uses_honest_audio_mime(adapter, tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF fake-wav")

    with patch.object(_bale_mod, "transcode_to_ogg_opus", MagicMock(return_value=None)):
        result = asyncio.run(adapter.send_voice(chat_id="1001", audio_path=str(wav)))

    assert result.success is True
    filename, mime = _uploaded_voice(adapter)
    assert (filename, mime) == ("voice.wav", "audio/wav")


def test_missing_audio_file_is_rejected_before_transcode(adapter, tmp_path):
    with patch.object(_bale_mod, "transcode_to_ogg_opus", MagicMock()) as spy:
        result = asyncio.run(adapter.send_voice(chat_id="1001", audio_path=str(tmp_path / "nope.mp3")))

    assert result.success is False
    spy.assert_not_called()
    adapter._http.post.assert_not_awaited()
