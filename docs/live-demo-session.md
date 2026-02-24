# Live demo session (recording guide)

This guide helps you record a real product demo from the terminal.

## What "real product GIF" means

It should be a recording of actual commands and real outputs from the running
project (not static slides or generated placeholder frames).

## Recommended recording flow (two windows)

This flow is optimized for an OpenCode-style demo: one terminal shows the
server, and the second shows live tool interactions.

### Window A: server

```bash
cd /path/to/memory
pip install -e .
python -m mcp_server.server
```

### Window B: live memory operations

```bash
cd /path/to/memory
pip install -e .

# short cut (about 60-90 sec)
./examples/run_live_demo_short.sh

# full cut (about 3-5 min)
./examples/run_live_demo_extended.sh
```

## Demo variants

### Short demo (fast product pitch)

Steps:

1. save preference + lesson
2. search across memory
3. apply feedback
4. show preference snapshot
5. show health + metrics

### Extended demo (full capability walk-through)

Steps:

1. save preference + lesson
2. search across lessons/preferences
3. apply natural-language feedback
4. show preference snapshot
5. add and search knowledge-base content
6. procedural + semantic memory
7. knowledge graph traversal/path
8. extraction + conversation memory
9. cross-session context injection
10. health + runtime metrics

## Recording tips

- Keep terminal width around 120-140 columns for readability.
- Increase font size before recording.
- Start with short demo first, then record extended demo.
- Keep each step visible for 2-4 seconds before scrolling.

## Optional: convert terminal recording to GIF

If you want a polished GIF from terminal output:

1. Record with `asciinema rec demo.cast`
2. Run the demo in the same terminal
3. Convert to GIF using your preferred converter (for example `agg` or ffmpeg workflow)
