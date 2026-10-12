#!/usr/bin/env python3
"""Add beat-synced "crazy" effects and transitions to the finished Navratri edit.

The picture is re-rendered frame by frame; the original audio is copied
untouched (-c:a copy), so the music never changes.

Look is taken from the four reference videos: warm red/orange film grade,
white flash cuts, whip-pan blur, glitch/RGB split, split-screen grids,
light leaks, mirror shots, echo trails, strobes, grain and vignette.

Needs: ffmpeg, python3, numpy, opencv-python-headless
  pip3 install numpy opencv-python-headless

Examples (run from the repo root):
  # full video, 1080p
  python3 tools/crazy_edit.py "ID 01 - Revised .mp4" out_1080.mp4
  # full video in 4K (slow)
  python3 tools/crazy_edit.py "ID 01 - Revised .mp4" out_4k.mp4 --width 3840
  # only a test section
  python3 tools/crazy_edit.py "ID 01 - Revised .mp4" test.mp4 --start 28 --dur 12
"""
import argparse
import json
import math
import os
import random
import subprocess
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Windows (seconds) with no burned-in subtitles, so mirror/grid can't garble text.
SAFE = [(0.0, 1.45), (16.6, 18.8), (24.05, 25.7), (37.05, 54.8)]


def in_any(t, spans):
    return any(a <= t < b for a, b in spans)


class Timeline:
    """Per-frame effect envelopes built from the cut/beat timings."""

    def __init__(self, timing):
        self.fps = timing["fps"]
        self.n = int(round(timing["duration"] * self.fps)) + 1
        self.t = np.arange(self.n) / self.fps
        keys = ["zoom", "rot", "dx", "shake", "rgb", "flash", "blur", "glitch",
                "leak", "bw", "echo", "strobe", "glow", "grid", "mirror", "fade"]
        for k in keys:
            setattr(self, k, np.zeros(self.n, np.float32))
        self.zoom += 1.0
        self.build(timing)

    # envelope helpers --------------------------------------------------
    def decay(self, arr, t0, peak, tau, length=None, mode="add"):
        length = length or tau * 5
        m = (self.t >= t0) & (self.t < t0 + length)
        v = peak * np.exp(-(self.t[m] - t0) / tau)
        if mode == "max":
            arr[m] = np.maximum(arr[m], v)
        else:
            arr[m] += v

    def span(self, arr, a, b, value, mode="max"):
        m = (self.t >= a) & (self.t < b)
        arr[m] = np.maximum(arr[m], value) if mode == "max" else value

    def ramp(self, arr, a, b, v0, v1):
        m = (self.t >= a) & (self.t < b)
        arr[m] = v0 + (v1 - v0) * (self.t[m] - a) / (b - a)

    # the edit ----------------------------------------------------------
    def build(self, timing):
        rnd = random.Random(7)
        T = self.t

        # every beat: small zoom bump (bottom-anchored so subtitles stay)
        for b in timing["beats"]:
            self.decay(self.zoom, b, 0.025, 0.09)
            self.decay(self.glow, b, 0.25, 0.1)
        # strong hits: punch + shake + RGB split
        for s in timing["strong"]:
            self.decay(self.zoom, s, 0.05, 0.1)
            self.decay(self.shake, s, 0.7, 0.08, mode="max")
            self.decay(self.rgb, s, 0.6, 0.06, mode="max")
            self.decay(self.glow, s, 0.6, 0.12)

        # transitions on every cut
        cuts, last = [], -9
        for c in timing["cuts"]:
            if c - last >= 0.12:
                cuts.append(c)
                last = c
        types = ["flash", "whip", "glitch", "punch", "leak", "spin"]
        prev = -9
        for c in cuts:
            if c < 1.5:
                continue
            dense = c - prev < 0.3
            prev = c
            kind = rnd.choice(["glitch", "flash"]) if dense else rnd.choice(types)
            if 30.3 < c < 31.6:
                kind = "glitch"
            self.transition(kind, c, small=dense)

        # --- sections ---------------------------------------------------
        # intro: black & white film, snap to colour on the reveal cut
        self.span(self.bw, 0, 1.6, 1.0)
        self.ramp(self.leak, 0.2, 1.6, 0.0, 0.8)
        self.ramp(self.fade, 0, 0.4, 1.0, 0.0)
        self.decay(self.flash, 1.6, 1.0, 0.08)
        self.decay(self.zoom, 1.6, 0.2, 0.15)
        self.decay(self.shake, 1.6, 1.0, 0.15, mode="max")

        # title card "Ishani Dave na Garba": glitch storm
        for a, b in [(30.70, 30.84), (30.92, 31.02), (31.20, 31.32), (31.60, 31.66)]:
            self.span(self.glitch, a, b, 1.0)
            self.span(self.rgb, a, b + 0.06, 1.0)
        self.span(self.shake, 30.7, 31.6, 0.5)

        # black & white pockets (refs 2 & 4)
        for a, b in [(16.58, 16.84), (43.78, 44.26)]:
            self.span(self.bw, a, b, 1.0)

        # mirror / kaleidoscope on drone + firework shots
        for a, b in [(24.06, 24.85), (25.24, 25.7), (37.08, 37.38), (37.68, 38.45), (49.22, 49.7)]:
            self.span(self.mirror, a, b, 1.0)

        # split-screen grids (ref 1): 3 = 3x3, 2 = 2x2
        for a, b, g in [(17.05, 17.32, 3), (24.52, 24.74, 2), (39.30, 39.56, 3),
                        (43.28, 43.48, 2), (46.72, 46.94, 3), (49.40, 49.66, 3)]:
            self.span(self.grid, a, b, g)

        # echo trails on spinning dancers
        for a, b in [(26.0, 28.3), (41.0, 43.2)]:
            self.span(self.echo, a, b, 0.55)

        # strobe on CO2 jets + finale
        for a, b in [(47.2, 49.0), (50.3, 51.9)]:
            for bt in timing["beats"]:
                if a <= bt < b:
                    self.span(self.strobe, bt + 0.06, bt + 0.14, 1.0)

        # finale fireworks: continuous shake + glow
        self.span(self.shake, 49.2, 52.0, 0.45)
        self.span(self.glow, 49.2, 52.0, 0.8)

        # outro: slow push-in, warm leak, fade out
        self.ramp(self.zoom, 52.0, 54.8, 1.0, 1.07)
        self.ramp(self.leak, 53.6, 54.8, 0.0, 0.9)
        self.ramp(self.fade, 54.35, 54.8, 0.0, 1.0)

        self.zoom = np.clip(self.zoom, 1.0, 1.35)
        self.glow = np.clip(self.glow, 0, 1.2)

    def transition(self, kind, c, small=False):
        k = 0.6 if small else 1.0
        if kind == "flash":
            self.decay(self.flash, c, 0.85 * k, 0.05)
            self.decay(self.zoom, c, 0.12 * k, 0.12)
        elif kind == "whip":
            for a, b in [(c - 0.08, c), (c, c + 0.08)]:
                self.span(self.blur, a, b, 1.0 * k)
            self.ramp(self.dx, c - 0.08, c, 0.0, 0.06 * k)
            self.ramp(self.dx, c, c + 0.08, -0.06 * k, 0.0)
        elif kind == "glitch":
            self.span(self.glitch, c, c + 0.1, k)
            self.decay(self.rgb, c, 1.0, 0.08, mode="max")
        elif kind == "punch":
            self.decay(self.zoom, c, 0.15 * k, 0.15)
            self.decay(self.rgb, c, 0.8, 0.08, mode="max")
            self.decay(self.shake, c, 0.8, 0.1, mode="max")
        elif kind == "leak":
            self.decay(self.leak, c, 1.0 * k, 0.25)
            self.decay(self.flash, c, 0.35 * k, 0.05)
        elif kind == "spin":
            self.ramp(self.rot, c, c + 0.2, 8.0 * k, 0.0)
            self.ramp(self.zoom, c, c + 0.2, 1.0 + 0.25 * k, 1.0)
            self.span(self.blur, c, c + 0.08, 0.6 * k)


class Fx:
    """Image operations; all work on uint8 BGR frames of size W x H."""

    def __init__(self, w, h, seed=3):
        self.w, self.h = w, h
        rng = np.random.default_rng(seed)
        # vignette
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        vig = np.clip(1.0 - 0.38 * np.clip(r - 0.45, 0, None) ** 1.6, 0, 1)
        self.vig = cv2.merge([(vig * 255).astype(np.uint8)] * 3)
        # film grain bank
        self.grain = []
        for _ in range(6):
            g = rng.normal(128, 22, (h // 2, w // 2)).astype(np.float32)
            g = cv2.resize(np.clip(g, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
            self.grain.append(cv2.merge([g] * 3))
        self.white = np.full((h, w, 3), (232, 242, 255), np.uint8)  # warm white (BGR)
        # grade LUTs: contrast + warm (refs 2 & 3)
        x = np.arange(256) / 255.0
        s = np.clip(0.5 + (x - 0.5) * 1.14, 0, 1)
        s = 0.015 + 0.985 * s
        lut_b = np.clip(s ** 1.06 * 0.94, 0, 1)
        lut_g = np.clip(s ** 1.0 * 0.99, 0, 1)
        lut_r = np.clip(s ** 0.93 * 1.04, 0, 1)
        self.lut = np.stack([lut_b, lut_g, lut_r], 1).reshape(256, 1, 3)
        self.lut = (self.lut * 255).astype(np.uint8)
        self.rng = rng

    def geometry(self, f, z, rot, dx, dy):
        if abs(z - 1) < 1e-3 and abs(rot) < 1e-3 and abs(dx) < 0.5 and abs(dy) < 0.5:
            return f
        # anchor low in frame so bottom subtitles stay visible
        M = cv2.getRotationMatrix2D((self.w / 2, self.h * 0.88), rot, z)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(f, M, (self.w, self.h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    def mirror(self, f):
        half = self.w // 2
        f = f.copy()
        f[:, self.w - half:] = f[:, :half][:, ::-1]
        return f

    def grid(self, frames, n):
        tw, th = self.w // n, self.h // n
        out = np.zeros_like(frames[-1])
        for i in range(n * n):
            src = frames[-1 - min(len(frames) - 1, (i * 3) % len(frames))]
            tile = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
            if i % 2:
                tile = tile[:, ::-1]
            y, x = (i // n) * th, (i % n) * tw
            out[y:y + th, x:x + tw] = tile
        lw = max(2, self.w // 400)
        for i in range(1, n):
            out[i * th - lw // 2:i * th + lw // 2 + 1] = 0
            out[:, i * tw - lw // 2:i * tw + lw // 2 + 1] = 0
        return out

    def whip(self, f, amt):
        k = int(amt * self.w * 0.045) | 1
        return cv2.blur(f, (k, 1)) if k > 2 else f

    def rgb_split(self, f, amt):
        d = int(amt * self.w * 0.012)
        if d < 1:
            return f
        b, g, r = cv2.split(f)
        return cv2.merge([np.roll(b, -d, 1), g, np.roll(r, d, 1)])

    def glitch(self, f, amt, fi):
        rng = np.random.default_rng(fi * 31 + 1)
        f = f.copy()
        for _ in range(rng.integers(5, 12)):
            y = int(rng.integers(0, self.h))
            hh = int(rng.integers(self.h // 80, self.h // 12))
            sh = int(rng.normal(0, self.w * 0.06 * amt))
            f[y:y + hh] = np.roll(f[y:y + hh], sh, 1)
            if rng.random() < 0.3:
                f[y:y + hh] = f[y:y + hh][:, :, [2, 0, 1]]
        return f

    def grade(self, f, bw):
        f = cv2.LUT(f, self.lut)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.merge([gray] * 3)
        if bw > 0.01:
            hard = cv2.convertScaleAbs(gray3, alpha=1.25, beta=-25)
            return cv2.addWeighted(f, 1 - bw, hard, bw, 0)
        return cv2.addWeighted(f, 1.22, gray3, -0.22, 0)  # +22% saturation

    def glow(self, f, amt):
        if amt < 0.02:
            return f
        sw, sh = self.w // 8, self.h // 8
        small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.subtract(small, (140, 140, 140, 0))
        small = cv2.GaussianBlur(small, (0, 0), sw / 60)
        small = cv2.multiply(small, (0.55, 0.8, 1.25, 0))  # red-orange halation
        big = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(f, 1.0, big, 1.6 * amt, 0)

    def leak(self, f, amt, t):
        if amt < 0.02:
            return f
        sw, sh = 160, 90
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        cx = sw * (0.15 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.7)))
        cy = sh * (0.3 + 0.2 * math.cos(t * 1.1))
        d = ((xx - cx) / (sw * 0.45)) ** 2 + ((yy - cy) / (sh * 0.7)) ** 2
        a = np.exp(-d)[..., None]
        col = np.array([40, 120, 255], np.float32) * (1 - xx[..., None] / sw) + \
            np.array([150, 40, 255], np.float32) * (xx[..., None] / sw)
        l = np.clip(a * col * amt, 0, 255).astype(np.uint8)
        l = cv2.resize(l, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        inv = cv2.bitwise_not(f)
        return cv2.add(f, cv2.multiply(l, inv, scale=1 / 255))  # screen blend

    def finish(self, f, fi, flick):
        f = cv2.multiply(f, self.vig, scale=flick / 255)
        g = self.grain[fi % len(self.grain)]
        return cv2.addWeighted(f, 1.0, g, 0.09, -128 * 0.09)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--timing", default=os.path.join(HERE, "navratri_timing.json"))
    a = ap.parse_args()

    timing = json.load(open(a.timing))
    tl = Timeline(timing)
    fps = tl.fps
    W = a.width
    H = int(round(W * 9 / 16 / 2)) * 2
    dur = a.dur if a.dur else timing["duration"] - a.start
    nframes = int(round(dur * fps))
    i0 = int(round(a.start * fps))

    trim = ["-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}"]
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", *trim, "-i", a.src, "-map", "0:v:0",
         "-vf", f"scale={W}:{H}:flags=lanczos", "-r", str(fps),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *trim, "-i", a.src,
         "-map", "0:v", "-map", "1:a:0", "-c:v", "libx264", "-preset", a.preset,
         "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)

    fx = Fx(W, H)
    hist = deque(maxlen=24)
    prev = None
    fsize = W * H * 3
    for k in range(nframes):
        buf = reader.stdout.read(fsize)
        if len(buf) < fsize:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        i = min(i0 + k, tl.n - 1)
        t = i / fps

        sh = tl.shake[i]
        dx = tl.dx[i] * W + sh * W * 0.012 * (math.sin(i * 1.9) + 0.5 * math.sin(i * 4.3))
        dy = sh * H * 0.010 * (math.cos(i * 2.3) + 0.5 * math.sin(i * 5.1))
        z = tl.zoom[i] + (0.03 * sh if sh > 0 else 0)
        f = fx.geometry(f, z, tl.rot[i], dx, dy)
        if tl.mirror[i] > 0 and in_any(t, SAFE):
            f = fx.mirror(f)
        hist.append(f)
        if tl.grid[i] >= 2 and in_any(t, SAFE):
            f = fx.grid(list(hist), int(tl.grid[i]))
        if tl.echo[i] > 0 and prev is not None:
            f = cv2.addWeighted(f, 1 - tl.echo[i] * 0.6, prev, tl.echo[i] * 0.6, 0)
        prev = f
        if tl.blur[i] > 0:
            f = fx.whip(f, tl.blur[i])
        if tl.glitch[i] > 0:
            f = fx.glitch(f, tl.glitch[i], i)
        if tl.rgb[i] > 0.02:
            f = fx.rgb_split(f, tl.rgb[i])
        f = fx.grade(f, float(tl.bw[i]))
        f = fx.glow(f, 0.35 + float(tl.glow[i]))
        f = fx.leak(f, float(tl.leak[i]), t)
        if tl.flash[i] > 0.01:
            f = cv2.addWeighted(f, 1 - min(1, tl.flash[i]), fx.white, min(1, tl.flash[i]), 0)
        if tl.strobe[i] > 0 and (i // 2) % 2 == 0:
            f = cv2.convertScaleAbs(f, alpha=0.2)
        flick = 1.0 + 0.025 * math.sin(i * 7.7) * math.sin(i * 1.3)
        if tl.bw[i] > 0.5:
            flick += 0.06 * math.sin(i * 11.1)
        f = fx.finish(f, i, flick)
        if tl.fade[i] > 0:
            f = cv2.convertScaleAbs(f, alpha=1 - float(tl.fade[i]))
        writer.stdin.write(f.tobytes())
        if k % 100 == 0:
            print(f"frame {k}/{nframes}  t={t:.2f}s", flush=True)

    writer.stdin.close()
    writer.wait()
    reader.wait()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
