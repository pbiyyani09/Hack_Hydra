# Demo video — how it was made

`../medmemgraph_demo.mp4` (2:00, 1080p) is generated, not screen-recorded. This
directory contains everything needed to reproduce or audit it.

## Everything on screen is real output

The terminal content is stdout captured verbatim from live runs against the
ingested HydraDB graph — no mockups, no edited numbers:

| file | produced by |
|---|---|
| `captured_beat12.txt` | `python -m medmemgraph.demo.agent --patient 10056223 --epsilon 0` |
| `captured_beat3.txt` | `python -m medmemgraph.demo.provenance --patient 10056223 --claim 614261132365461680` |
| `captured_beat4.txt` | the eval results in `results-combined2/`, reformatted to fit 1080p |

The only synthesised elements are the **typing animation** and the **voice-over**
(`edge-tts`, `en-US-AndrewNeural`). Stating that plainly because a polished video
of a terminal invites the question.

## Reproducing

```bash
uv pip install edge-tts pillow          # plus a static ffmpeg on PATH
uv run python demo/video/render_video.py
```

`narration.json` holds the script and each beat's measured audio duration; the
renderer syncs frame counts to those durations so the numbers are on screen
before they are spoken.

## What it claims

The results beat shows `p = 0.20 (not significant)` on screen, and the narration
says "at this sample size the accuracy difference sits inside the noise, so we
call it a match, not a win." That is the claim the data supports — see the
README's Results section for the full statistics.
