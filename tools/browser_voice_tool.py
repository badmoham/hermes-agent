"""Browser Voice Tools - Web Speech API integration during browser sessions.

Uses the existing browser_evaluate mechanism to inject and run Web Speech API
calls within active browser automation sessions. No separate audio infrastructure
required - speech plays through the browser's current page context.

Tools:
  - browser_speak: Synthesize speech using Web Speech API SpeechSynthesis
  - browser_listen: Capture speech via Web Speech API SpeechRecognition  
  - browser_stop_speaking: Cancel ongoing speech synthesis
"""

import json
import time
from typing import Optional, Dict, Any

from tools.browser_tool import (
    _browser_eval,
)


def browser_speak(text: str, lang: Optional[str] = None, rate: Optional[float] = None, 
                  pitch: Optional[float] = None) -> Dict[str, Any]:
    """Speak text using Web Speech Synthesis API.

    Args:
        text: Text to synthesize into speech. Required.
        lang: BCP-47 language tag (e.g. 'en-US'). Defaults to page locale if omitted.
        rate: Speech rate multiplier 0.1–2.0 (default 1.0). 
        pitch: Speech pitch multiplier 0.0–2.0 (default 1.0).

    Returns:
        Dict with success status and utterance details.
    """
    # Build the JS that constructs and speaks a SpeechSynthesisUtterance
    js = f"""
(function() {{
    const u = new SpeechSynthesisUtterance({text!r});
"""
    if lang:
        js += f'    u.lang = "{lang}";\n'
    if rate is not None:
        js += f"    u.rate = {rate};\n"
    if pitch is not None:
        js += f"    u.pitch = {pitch};\n"
    
    js += """
    speechSynthesis.speak(u);
    return JSON.stringify({ spoken: true, textLength: u.text.length });
})()
"""
    
    result = _browser_eval(js)
    if not isinstance(result, str):
        return {"error": f"speech synthesis failed: {result}"}
    try:
        parsed = json.loads(result)
        return {"success": True, "spoken": True, "textLength": len(text)}
    except Exception:
        # JS eval succeeded but didn't parse cleanly; treat as success anyway
        return {"success": True, "spoken": True}


def browser_stop_speaking() -> Dict[str, Any]:
    """Cancel any ongoing speech synthesis."""
    js = """
(function() {
    speechSynthesis.cancel();
    return JSON.stringify({ stopped: true });
})()
"""
    try:
        result = _browser_eval(js)
        if isinstance(result, str):
            return {"success": True, "stopped": True}
        else:
            return {"success": False, "error": f"Unexpected eval result: {result}"}
    except Exception as e:
        return {"success": False, "error": f"{e}"}


def browser_listen(lang: Optional[str] = None, timeout_sec: float = 15.0) -> Dict[str, Any]:
    """Listen for speech using Web Speech Recognition API.

    Blocks until speech is detected and transcribed, or timeout reached.
    
    Args:
        lang: BCP-47 language tag (e.g. 'en-US'). Defaults to page locale if omitted.
        timeout_sec: Maximum seconds to wait for speech input (default 15).

    Returns:
        Dict with recognized text or error status.
    """
    # Start recognition via evaluate - returns immediately, we poll for result
    js = f"""
(function() {{
    window.__hermes_speech_result = null;
    const rec = new webkitSpeechRecognition();
    if ({lang!r}) rec.lang = "{lang}";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    
    let resolve, reject;
    window.__hermes_speech_promise = new Promise((res, rej) => {{ 
        resolve = res; reject = rej; 
    }});
    
    rec.onresult = (e) => {{
        const transcript = e.results[0][0].transcript;
        window.__hermes_speech_result = {{"type": "success", "text": transcript}};
        resolve(transcript);
    }};
    
    rec.onerror = (e) => {{
        window.__hermes_speech_result = {{"type": "error", "message": e.message}};
        reject(e.message);
    }};
    
    rec.onend = () => {{
        if (!window.__hermes_speech_result) {{
            window.__hermes_speech_result = {{"type": "timeout"}};
        }}
    }};
    
    try {{
        rec.start();
        return JSON.stringify({{ started: true }});
    }} catch (e) {{
        return JSON.stringify({{ error: e.message }});
    }}
}})()
"""
    start_result = _browser_eval(js)
    if not isinstance(start_result, str):
        return {"success": False, "error": f"failed to start recognition: {start_result}"}
    
    try:
        start_parsed = json.loads(start_result)
        if "error" in start_parsed:
            return {"success": False, "error": start_parsed["error"]}
    except:
        pass
    
    # Poll for result up to timeout
    elapsed = 0.0
    while elapsed < timeout_sec:
        time.sleep(0.5)
        elapsed += 0.5
        
        poll_js = """
(function() {
    if (window.__hermes_speech_result) {
        return JSON.stringify(window.__hermes_speech_result);
    }
    return null;
})()
"""
        try:
            poll_result = _browser_eval(poll_js)
            if isinstance(poll_result, str):
                parsed = json.loads(poll_result)
                if "success" in parsed.get("type", "") or parsed.get("type") == "error":
                    return {"text": parsed.get("text"), "status": parsed.get("type")}
        except:
            pass
    
    # Timeout - stop recognition
    browser_stop_speaking()
    return {"success": False, "status": "timeout", "message": f"No speech detected within {int(timeout_sec)} seconds"}


def format_browser_voice_result(result):
    """Format the result of a browser voice tool call."""
    if isinstance(result, dict) and result.get("success"):
        text = result.get("text") or ""
        if result.get("spoken"):
            return "🔊 Speech started"
        elif result.get("stopped"):
            return "🔇 Speech stopped"
        elif result.get("status") == "timeout":
            return "⏱ No speech detected (timeout)"
        else:
            if text:
                return f'🎤 Heard: "{text}"'
    elif isinstance(result, dict) and "error" in result:
        return f"ERROR: {result['error']}"
    return str(result)
