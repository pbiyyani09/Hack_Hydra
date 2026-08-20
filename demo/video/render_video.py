"""Render the demo video: real captured terminal output, typed out in sync with
the narration, then muxed with the TTS audio.

Everything on screen is genuine output from `demo/agent.py`, `demo/provenance.py`
and `eval/report.py` captured on the live graph — no mockups, no retouching. The
only thing synthesised is the typing animation and the voice.
"""
import json, pathlib, subprocess, textwrap
from PIL import Image, ImageDraw, ImageFont

CAP = pathlib.Path("/tmp/demo_capture")
FRAMES = CAP / "frames"; FRAMES.mkdir(exist_ok=True)
FFMPEG = "/home/dreadnought/.local/bin/ffmpeg"
W, H, FPS = 1920, 1080, 24
BG, FG = (13, 17, 23), (201, 209, 217)
ACCENT, DIM, WIN = (88, 166, 255), (110, 118, 129), (22, 27, 34)

def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold
              else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if pathlib.Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

MONO, MONO_B = font(21), font(21, True)
BIG, MED = font(64, True), font(30)

def colour_for(line):
    s = line.strip()
    if s.startswith("$"): return ACCENT
    if s.startswith(("answer:", "TIMELINE", "CONTEXT")): return (126, 231, 135)
    if s.startswith(("route=", "path:", "citations:", "tokens:", "latency_ms:")): return DIM
    if s.startswith("[") and "]" in s: return (255, 166, 87)
    if s.strip().startswith("--") and "-->" in s: return (255, 123, 114)
    if "MedMemGraph" in s or "medmemgraph" in s: return (126, 231, 135)
    if "Full-context" in s or "fullctx" in s: return (255, 166, 87)
    if s.startswith("->") or "p = 0.20" in s: return (88, 166, 255)
    return FG

def chrome(d, subtitle):
    d.rectangle([0, 0, W, 54], fill=WIN)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([26 + i*26, 20, 40 + i*26, 34], fill=c)
    d.text((130, 16), subtitle, font=MONO, fill=DIM)

def wrap(lines, width=143):
    out = []
    for ln in lines:
        out.extend(textwrap.wrap(ln, width) or [""])
    return out

def render(lines, path, shown, subtitle):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    chrome(d, subtitle)
    y = 78
    for ln in lines[:shown]:
        if y > H - 40: break
        d.text((34, y), ln[:143], font=MONO, fill=colour_for(ln))
        y += 27
    img.save(path)

def card(path, title, sub, bullets=()):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((110, 300), title, font=BIG, fill=FG)
    d.text((114, 400), sub, font=MED, fill=ACCENT)
    y = 500
    for b in bullets:
        d.text((118, y), b, font=MED, fill=DIM); y += 52
    img.save(path)

spec = json.loads((CAP / "narration.json").read_text())
beat_lines = {
    "beat1": wrap([l for l in (CAP/"beat12.txt").read_text().splitlines()][:9]),
    "beat2": wrap([l for l in (CAP/"beat12.txt").read_text().splitlines()][9:]),
    "beat3": wrap((CAP/"beat3.txt").read_text().splitlines()),
    "beat4": wrap((CAP/"beat4.txt").read_text().splitlines()),
}
SUB = {
    "beat1": "medmemgraph — cross-admission question",
    "beat2": "medmemgraph — question with no answer in the record",
    "beat3": "medmemgraph — provenance walk: how did this fact change?",
    "beat4": "medmemgraph — results, 10 patients / 336 paired items",
}

idx = 0
for b in spec["beats"]:
    n = int(b["actual"] * FPS)
    if b["screen"] == "title":
        for i in range(n):
            card(FRAMES/f"f{idx:05d}.png", "MedMemGraph",
                 "graph-native clinical memory on HydraDB OSS",
                 ["13,782 clinical claims · 28,141 dialogue turns · 20 patients",
                  "every claim linked to the sentence that produced it",
                  "self-hosted — no managed retrieval API"]); idx += 1
    elif b["screen"] == "close":
        for i in range(n):
            card(FRAMES/f"f{idx:05d}.png", "MedMemGraph",
                 "HydraDB OSS 0.1.1 · AGPL-3.0 · self-hosted",
                 ["github.com/pbiyyani09/Hack_Hydra",
                  "0.783 vs 0.757 answerable · 8x fewer tokens"]); idx += 1
    else:
        lines = beat_lines[b["screen"]]
        hold = int(n * 0.55)                       # hold the finished screen longer:
                                                   # the narration quotes numbers, so they
                                                   # must be on screen before they are said
        typing = max(n - hold, 1)
        for i in range(n):
            shown = len(lines) if i >= typing else max(1, int(len(lines) * (i / typing)))
            render(lines, FRAMES/f"f{idx:05d}.png", shown, SUB[b["screen"]]); idx += 1

print(f"  rendered {idx} frames ({idx/FPS:.1f}s at {FPS}fps)")

# concat audio in beat order, then mux with the frame sequence
with open(CAP/"audio_list.txt", "w") as fh:
    for b in spec["beats"]:
        fh.write(f"file '{CAP}/audio/{b['id']}.mp3'\n")
subprocess.run([FFMPEG,"-y","-f","concat","-safe","0","-i",str(CAP/"audio_list.txt"),
                "-c","copy",str(CAP/"voice.mp3")], check=True, capture_output=True)
subprocess.run([FFMPEG,"-y","-framerate",str(FPS),"-i",str(FRAMES/"f%05d.png"),
                "-i",str(CAP/"voice.mp3"),"-c:v","libx264","-pix_fmt","yuv420p",
                "-crf","20","-preset","medium","-c:a","aac","-b:a","192k","-shortest",
                str(CAP/"medmemgraph_demo.mp4")], check=True, capture_output=True)
print("  wrote /tmp/demo_capture/medmemgraph_demo.mp4")
