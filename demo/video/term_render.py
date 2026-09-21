import json, re, os, sys
from PIL import Image, ImageDraw, ImageFont
V = sys.argv[1]; D = float(sys.argv[2]); FPS = 15
rows = [json.loads(l) for l in open(f"{V}/term.jsonl")]
sel = []
for r in rows:
    L = r["line"]
    if L.startswith("ladder:"):
        sel.append((r["t"], "ladder", "ladder   " + L.split(":", 1)[1].strip()))
    elif re.match(r"^\[\d+\] final:", L):
        sel.append((r["t"], "final", "final    " + L.split("final:", 1)[1].strip()))
    elif m := re.match(r"^\[(\d+)\] (k2-[\d.]+b) attempt (\d+): (\S+)\s+\((\d+) tok, (\d+) ms\)", L):
        step, model, att, reason, tok, ms = m.groups()
        kind = "ok" if reason in ("ok", "final") else "bad"
        mark = "✓" if kind == "ok" else "✗"
        sel.append((r["t"], f"{model}:{kind}", f"step {int(step):>2}  {model.replace('k2-',''):>5}  {mark} {reason:<10} {int(ms)/1000:5.1f}s"))
    elif L.strip().startswith("-> "):
        sel.append((r["t"], "tool", "              " + L.strip()[3:][:92]))
    elif m := re.search(r"(\d+ (?:passed|failed)(?:, \d+ (?:passed|failed))?)", L):
        kind = "pass" if "failed" not in m.group(1) else "fail"
        sel.append((r["t"], kind, "              pytest: " + m.group(1)))
# compress time: each real gap -> clamp, then rescale to fit D-2.5s
ts = [s[0] for s in sel]; gaps = [0.0] + [max(0.0, b - a) for a, b in zip(ts, ts[1:])]
vg = [min(max(g * 0.12, 0.35), 2.2) if i else 0 for i, g in enumerate(gaps)]
cum = []; acc = 0.0
for g in vg: acc += g; cum.append(acc)
scale = (D - 2.5) / cum[-1] if cum[-1] else 1
show_at = [0.8 + c * scale for c in cum]
real_end = rows[-1]["t"]
W, H = 1920, 1080
BG = (28, 28, 26); PAPER = (225, 220, 208); MUTED = (140, 136, 126); ACC = (237, 139, 108)
COL = {"ladder": MUTED, "k2-0.9b:ok": PAPER, "k2-0.9b:bad": (201, 119, 106), "k2-3.7b:ok": ACC, "k2-3.7b:bad": (201, 119, 106),
       "k2-375b:ok": (240, 200, 120), "k2-375b:bad": (201, 119, 106), "tool": (143, 179, 201), "pass": (156, 207, 138),
       "fail": (217, 138, 122), "final": (156, 207, 138)}
mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 27)
monob = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 27, index=1)
sans = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
sansb = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34, index=1)
LH = 40; TOP = 190; MAXL = 20
esc_t = next((show_at[i] for i, s in enumerate(sel) if s[1].startswith("k2-3.7b")), None)
os.makedirs(f"{V}/term_frames", exist_ok=True)
n = int(D * FPS)
for f in range(n):
    t = f / FPS
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((96, 70), "K2 CASCADE  ·  LIVE RUN ON A MACBOOK", font=sansb, fill=PAPER)
    d.text((96, 118), "0.9B and 3.7B running locally  ·  375B via the IFM API as a one-word judge  ·  sped up", font=sans, fill=MUTED)
    frac = min(1.0, max(0.0, (t - 0.8) / max(0.1, D - 2.5)))
    real = frac * real_end
    d.text((W - 96, 76), f"real time {int(real)//60}:{int(real)%60:02d}", font=sans, fill=MUTED, anchor="ra")
    d.line([(96, 165), (W - 96, 165)], fill=(60, 60, 56), width=2)
    vis = [s for s, at in zip(sel, show_at) if at <= t][-MAXL:]
    for i, (_, kind, text) in enumerate(vis):
        font = monob if kind.startswith("k2-3.7b") or kind in ("final", "pass") else mono
        d.text((96, TOP + i * LH), text, font=font, fill=COL.get(kind, PAPER))
    if esc_t is not None and esc_t <= t < esc_t + 2.4:
        d.rounded_rectangle([(W - 560, H - 150), (W - 96, H - 84)], radius=10, fill=ACC)
        d.text((W - 328, H - 117), "escalated to 3.7B", font=sansb, fill=BG, anchor="mm")
    im.save(f"{V}/term_frames/{f:05d}.png")
print("frames", n, "lines", len(sel), "escalation at", esc_t)
