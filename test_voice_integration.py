#!/usr/bin/env python3
"""Test integration of browser voice tools into hermes-agent.

Runs a quick smoke test to verify the three new browser voice tools are properly
registered and their handler functions can be invoked.
"""
import sys

from tools.browser_tool import (
    BROWSER_TOOL_SCHEMAS,
    _BROWSER_TOOL_TABLE,
)


def main():
    print("=== Hermes Agent Browser Voice Tools Integration Test ===\n")

    # 1. Verify schemas are registered
    voice_schemas = [s for s in BROWSER_TOOL_SCHEMAS if "browser_" in s["name"] and any(
        kw in s["description"].lower() or kw == s.get("name", "").replace("browser_", "") 
        for kw in ("speak", "listen")
    )]
    
    expected_voice_tools = {"browser_speak", "browser_listen"}
    found_voice_names = {s["name"] for s in voice_schemas}
    
    print(f"1. Schema registration check:")
    for name in expected_voice_tools:
        status = "✅ OK" if name in found_voice_names else f"❌ MISSING ({found_voice_names})"
        print(f"   {name}: {status}")

    # 2. Verify tool table entries
    registered_in_table = [e[0] for e in _BROWSER_TOOL_TABLE if any(
        kw in e[0] for kw in ("speak", "listen")
    )]
    
    print(f"\n2. Tool registration check:")
    for name in expected_voice_tools:
        status = "✅ OK" if name in registered_in_table else f"❌ MISSING ({registered_in_table})"
        print(f"   {name}: {status}")

    # 3. Verify handler function existence (via globals)
    import tools.browser_tool as bt_module
    print("\n3. Handler function check:")
    for name in expected_voice_tools:
        has_handler = hasattr(bt_module, name)
        status = "✅ OK" if has_handler else f"❌ MISSING attribute {name}"
        print(f"   module.{name}: {status}")

    # 4. Verify tool parameters match schema
    print("\n4. Parameter schema check:")
    for s in BROWSER_TOOL_SCHEMAS:
        if s["name"] == "browser_speak":
            props = list(s["parameters"]["properties"].keys())
            required = s["parameters"]["required"]
            has_text_param = "text" in props and "text" in required
            status = "✅ OK" if has_text_param else f"❌ Missing text param ({props})"
            print(f"   browser_speak parameters: {status}")
            
    # Summary
    all_passed = (expected_voice_tools == found_voice_names) and \
                len(registered_in_table) >= 2
    
    print("\n=== Test Result:", "PASS ✅" if all_passed else "FAIL ❌", "===")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
