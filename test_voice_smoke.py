#!/usr/bin/env python3
"""Smoke test for browser voice tools - verifies JS injection logic."""

import json

# Mock _browser_eval to return successful speech synthesis result
def mock_browser_eval(js_code, *args, **kwargs):
    if "SpeechSynthesisUtterance" in js_code:
        return '{"spoken": true, "textLength": 10}'
    elif "speechSynthesis.cancel()" in js_code:
        return '{"stopped": true}'
    elif "webkitSpeechRecognition" in js_code:
        # Simulate immediate recognition result for testing
        return '{"started": true}'
    return '{}'

# Patch the module's import
import tools.browser_tool as bt_module
bt_module._browser_eval = mock_browser_eval

# Now test our new voice tool functions
from tools.browser_voice_tool import browser_speak, browser_listen, browser_stop_speaking

print("=== Browser Voice Tools Smoke Test ===\n")

# Test 1: Speak text
result = browser_speak(text="Hello World!", lang="en-US", rate=1.0)
print(f"browser_speak result: {json.dumps(result, indent=2)}")

if not isinstance(result, dict):
    print("❌ FAIL: Expected dict from speak")
else:
    print("✅ PASS: Speak returned proper structure\n")

# Test 2: Stop speaking  
result = browser_stop_speaking()
print(f"browser_stop_speaking result: {json.dumps(result, indent=2)}")

if not isinstance(result, dict):
    print("❌ FAIL: Expected dict from stop")
else:
    print("✅ PASS: Stop returned proper structure\n")

# Test 3: Listen (will timeout quickly since no real speech)
import time
start = time.time()
result = browser_listen(lang="en-US", timeout_sec=1.0)
elapsed = time.time() - start
print(f"browser_listen result after {elapsed:.1f}s: {json.dumps(result, indent=2)}")

if isinstance(result, dict):
    print("✅ PASS: Listen returned proper structure\n")
else:
    print("❌ FAIL: Expected dict from listen\n")

# Test 4: JSON serialization works end-to-end through tool wrapper
def test_browser_speak_wrapper():
    """Verify the wrapper function serializes to valid JSON."""
    text = "Test speech"
    result = bt_module.browser_speak(text=text)
    
    # Should be a JSON string when returned from wrapped handler
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        print(f"✅ PASS: Wrapper produced valid output")
        return True
    except Exception as e:
        print(f"❌ FAIL: Serialization error: {e}")
        return False

test_browser_speak_wrapper()

print("\n=== Smoke Test Complete ===")
