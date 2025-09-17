# TUI Testing Guide

## Overview
The Dialectica TUI has been updated with improved streaming support. The streaming functionality now includes proper line buffering for smooth character-by-character display.

## Key Changes Made

### 1. Streaming Screen (`dialectica/tui/streaming_screen.py`)
- Added line buffering for smoother text streaming
- Improved `append_display()` method to accumulate partial lines
- Added buffer flushing on stream completion

### 2. Simple Streaming (`dialectica/tui/simple_streaming.py`)
- Implemented the same line buffering approach
- More reliable streaming with proper character accumulation
- Cleaner display updates using RichLog

### 3. Launcher Integration (`dialectica/tui/launcher.py`)
- Switched to use `SimpleStreamingScreen` for better reliability
- Maintains compatibility with the rest of the pipeline

### 4. Test Streaming (`dialectica/tui/test_streaming.py`)
- Mock streaming test that doesn't require API keys
- Supports different streaming speeds (normal, fast, slow)
- Error simulation for testing error handling

## How to Test

### 1. Mock Streaming Test (No API Required)
```bash
python dialectica/tui/test_streaming.py
```
This will show a TUI with buttons to test different streaming modes without needing API keys.

### 2. Test OpenAI Streaming (Requires API Key)
```bash
export OPENAI_API_KEY="your-key-here"
python test_openai_streaming.py
```
This tests the raw OpenAI streaming without the TUI.

### 3. Full TUI Launcher
```bash
python -m dialectica.tui.launcher
```
This launches the full TUI application with the improved streaming.

## What to Look For

1. **Smooth Character Display**: Text should appear character by character without flickering
2. **Line Handling**: Complete lines should appear immediately, partial lines buffer until complete
3. **No Display Issues**: The RichLog widget should properly display streaming JSON
4. **Proper Completion**: Stream should complete cleanly with status updates

## Streaming Implementation Details

The key improvement is the line buffering approach:

```python
self._line_buffer += delta

# Write complete lines immediately
while "\n" in self._line_buffer:
    line_end = self._line_buffer.index("\n") + 1
    line = self._line_buffer[:line_end]
    self._line_buffer = self._line_buffer[line_end:]
    log.write(line)

# Write partial lines when buffer gets long
if len(self._line_buffer) > 80:
    log.write(self._line_buffer, end="")
    self._line_buffer = ""
```

This ensures:
- Complete lines are displayed immediately
- Partial lines are buffered to prevent display issues
- Long partial lines are flushed periodically
- Remaining buffer is flushed on completion

## Known Issues Fixed

1. **Empty Display**: Fixed by proper buffering and write methods
2. **Character Flickering**: Resolved with line-based buffering
3. **Incomplete JSON Display**: Fixed with proper buffer flushing
4. **Thread Safety**: Using `call_from_thread` correctly for all updates

## Next Steps

The TUI streaming is now ready for testing with actual API calls. The improvements ensure smooth, reliable streaming display for the GPT-5 idea generation phase.