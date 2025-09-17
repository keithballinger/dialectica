#!/usr/bin/env python3
"""Simple CLI to test OpenAI GPT-5 streaming without any TUI"""

import os
import sys
from openai import OpenAI

def test_streaming():
    """Test GPT-5 streaming with a simple prompt"""

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Missing OPENAI_API_KEY environment variable")
        return

    print(f"API Key found: {api_key[:10]}...")

    # Create client
    client = OpenAI(api_key=api_key)

    # Simple test prompt
    prompt = "Generate a list of 3 creative names for a coffee shop, in JSON format with 'names' array"

    print(f"\nPrompt: {prompt}")
    print("\n" + "="*50)
    print("Starting stream...")
    print("="*50 + "\n")

    try:
        # Stream the response
        full_text = []
        event_count = 0

        with client.responses.stream(
            model="gpt-5",
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        ) as stream:
            print("Connected! Waiting for events...\n")

            for event in stream:
                event_count += 1

                # Log event type
                print(f"[Event #{event_count}] Type: {event.type}")

                if event.type == "response.output_text.delta":
                    # Print the delta text
                    delta = event.delta
                    print(f"[TEXT]: {repr(delta)}")
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    full_text.append(delta)

                elif event.type == "response.refusal.delta":
                    print(f"\n[REFUSAL]: {event.delta}")

                elif event.type == "response.error":
                    print(f"\n[ERROR]: {event.error}")

                elif event.type == "response.completed":
                    print(f"\n[COMPLETED]")

                else:
                    print(f"[UNKNOWN EVENT]: {event}")

            print("\n\n" + "="*50)
            print("Stream finished!")
            print("="*50)

            # Get final response
            final = stream.get_final_response()
            if final:
                print("\nFinal response available:")
                if hasattr(final, 'output_text'):
                    print(f"output_text: {final.output_text}")
                else:
                    print(f"Full text assembled: {''.join(full_text)}")

    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        # Check for specific error types
        error_str = str(e)
        if "401" in error_str:
            print("\n=> Authentication failed - check your API key")
        elif "404" in error_str:
            print("\n=> Model not found - 'gpt-5' might not be available")
        elif "429" in error_str:
            print("\n=> Rate limit exceeded")
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            print("\n=> Server error from OpenAI")

if __name__ == "__main__":
    print("Testing OpenAI GPT-5 Streaming...")
    print("="*50)
    test_streaming()