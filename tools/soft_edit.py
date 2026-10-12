#!/usr/bin/env python3
"""Soft, minimal re-finish of the Navratri edit with new typography.

- Removes the burned-in Gujarati subtitles, lyric words and title card
  (inpainting; the title card is replaced by an AI slow-motion stretch of
  the clean shot before it).
- AI morph transitions (RIFE frame interpolation) at a few section cuts.
- Soft filmic grade: matte blacks, warm highlights, mist bloom, fine grain.
- New typography: editorial Gujarati lines with English translation,
  3D extruded lyric words, a 3D title placed behind the singer (AI person
  segmentation), and quiet intro/outro cards.
- No shake, no beat zooms, no glitches. Original audio is copied untouched.

Needs: ffmpeg, numpy, opencv-python-headless, pillow (with raqm), torch,
onnxruntime, and two model files:
  flownet_v4.22.pkl     https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.22.pkl
  u2net_human_seg.onnx  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx

  python3 tools/soft_edit.py "ID 01 - Revised .mp4" out.mp4 --models ./models
"""
import argparse
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
FPS = 50
DURATION = 54.74

# ---------------------------------------------------------------- timings
# Old burned-in text: (start, end) seconds and the band it sits in (y0, y1 as frame fraction)
SUB_SPANS = [(0.95, 3.45), (5.15, 8.65), (11.7, 16.4), (18.7, 22.9), (25.7, 30.0)]
LYRIC_SPANS = [(32.25, 33.9), (35.35, 37.12)]
# Title card frames replaced by slow motion of the clean shot before it
TITLE_SRC = (1499, 1536)   # clean singer shot
TITLE_DST = (1499, 1580)   # frames to fill (inclusive)
# AI morph transitions: first frame of the incoming shot, half length in frames
MORPHS = [(682, 12), (941, 12), (1288, 12), (1855, 10), (2114, 12), (2343, 12), (2460, 12)]

LINES = [  # (segments [(start, text)], end, english)
    ([(1.70, "કંઈક તો છે,"), (2.55, " આ નવ રાતોમાં…")], 4.30,
     "There is something about these nine nights…"),
    ([(5.30, "એની સાથે જાગે છે"), (6.80, " એક જૂની પરંપરા…")], 8.55,
     "With them awakens an age-old tradition…"),
    ([(11.80, "વર્ષોથી ચાલતી આવતી"), (13.55, " શ્રદ્ધા,"), (14.80, " સંગીત"), (15.80, " અને ગરબા…")], 16.45,
     "A faith carried through the years — music, and garba…"),
    ([(18.80, "એક એવી લાગણી,"), (20.30, " જેની રાહ આખું વર્ષ જોવાય છે…")], 22.75,
     "A feeling the whole year waits for…"),
    ([(25.80, "અને"), (26.80, " હવે ફરી એ જ નવરાત્રીની"), (28.80, " શરૂઆત")], 29.95,
     "And now, once again, Navratri begins."),
]
WORDS = [(32.30, "માંડ"), (32.62, "જાણવે"), (33.00, "શૂન્ય"), (33.40, "શૂન્ય"), (33.84, None),
         (35.40, "ઓઢે"), (35.72, "ઓઢણીને"), (36.10, "જાતો"), (36.42, "જોગાણી"), (37.08, None)]
BEHIND_PERSON = [(1499, 1580), (1819, 1854)]  # frames where typography sits behind people


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def font(name, size, weight=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), size, layout_engine=ImageFont.Layout.RAQM)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight] if "Anek" not in name else [100, weight])
        except Exception:
            pass
    return f


# ---------------------------------------------------------------- video io
def read_frames(src, W, H, i0=0, n=None):
    args = ["ffmpeg", "-v", "error", "-ss", f"{i0 / FPS:.4f}", "-i", src, "-map", "0:v:0",
            "-vf", f"scale={W}:{H}:flags=lanczos"]
    if n:
        args += ["-frames:v", str(n)]
    p = subprocess.Popen(args + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = W * H * 3
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
    p.wait()


# ---------------------------------------------------------------- text removal
def in_spans(t, spans):
    return any(a <= t < b for a, b in spans)


def remove_text(f, t):
    h, w = f.shape[:2]
    bands = []
    if in_spans(t, SUB_SPANS):
        bands.append((0.885, 0.99, 0.08, 0.92, 140, 90))
    if in_spans(t, LYRIC_SPANS):
        bands.append((0.42, 0.62, 0.0, 1.0, 175, 70))
    if not bands:
        return f
    mask = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1, vmin, smax in bands:
        ys, ye, xs, xe = int(h * y0), int(h * y1), int(w * x0), int(w * x1)
        hsv = cv2.cvtColor(f[ys:ye, xs:xe], cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] > vmin) & (hsv[..., 1] < smax)).astype(np.uint8) * 255
        mask[ys:ye, xs:xe] = cv2.dilate(m, np.ones((7, 7), np.uint8))
    if mask.any():
        f = cv2.inpaint(f, mask, 6, cv2.INPAINT_TELEA)
    return f


# ---------------------------------------------------------------- AI models
class Segmenter:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def mask(self, f):
        h, w = f.shape[:2]
        x = cv2.resize(f, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        out = self.s.run(None, {"input.1": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        out = (out - out.min()) / (out.max() - out.min() + 1e-6)
        m = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.clip((m - 0.35) / 0.3, 0, 1)
        return cv2.GaussianBlur(m, (0, 0), 2)[..., None]


def precompute(src, W, H, models, cache):
    """AI slow motion for the title card and AI morph frames. Returns {frame: image}."""
    sys.path.insert(0, HERE)
    from rife.interp import Rife
    rife = Rife(os.path.join(models, "flownet_v4.22.pkl"), scale=0.5)
    os.makedirs(cache, exist_ok=True)
    rep = {}

    def cached(i, make):
        p = os.path.join(cache, f"{W}_{i}.npy")
        if os.path.exists(p):
            return np.load(p)
        img = make()
        np.save(p, img)
        return img

    # title slow motion
    s0, s1 = TITLE_SRC
    d0, d1 = TITLE_DST
    need = [i for i in range(d0, d1 + 1) if not os.path.exists(os.path.join(cache, f"{W}_{i}.npy"))]
    if need:
        clean = [remove_text(f, (s0 + k) / FPS) for k, f in enumerate(read_frames(src, W, H, s0, s1 - s0 + 1))]
    for i in range(d0, d1 + 1):
        pos = (i - d0) * (s1 - s0) / (d1 - d0)
        a = int(math.floor(pos))
        fr = pos - a

        def make(a=a, fr=fr):
            if fr < 0.02 or a + 1 >= len(clean):
                return clean[min(a, len(clean) - 1)]
            return rife.between(clean[a], clean[a + 1], [fr])[0]
        rep[i] = cached(i, make)
        print(f"title slow-mo {i}", flush=True)

    # morphs between shots
    for c, half in MORPHS:
        ia, ib = c - half - 1, c + half
        need = [i for i in range(c - half, c + half) if not os.path.exists(os.path.join(cache, f"{W}_{i}.npy"))]
        if need:
            a = rep.get(ia)
            if a is None:
                a = remove_text(next(read_frames(src, W, H, ia, 1)), ia / FPS)
            b = rep.get(ib)
            if b is None:
                b = remove_text(next(read_frames(src, W, H, ib, 1)), ib / FPS)
            n = 2 * half
            ts = [ease_inout((k + 1) / (n + 1)) for k in range(n)]
            frames = rife.between(a, b, ts)
        for k, i in enumerate(range(c - half, c + half)):
            rep[i] = cached(i, lambda k=k: frames[k])
        print(f"morph at {c}", flush=True)
    return rep


def ease_inout(x):
    return 0.5 - 0.5 * math.cos(math.pi * x)


# ---------------------------------------------------------------- look
class Look:
    def __init__(self, W, H):
        self.W, self.H = W, H
        x = np.arange(256) / 255.0
        s = 0.5 + (x - 0.5) * 1.06
        s = 0.045 + 0.915 * np.clip(s, 0, 1)                  # matte blacks, soft highlights
        b = np.clip(s * 0.97 + 0.02 * (1 - x), 0, 1)           # cool shadows, warm highs
        g = np.clip(s * 0.995, 0, 1)
        r = np.clip(s ** 0.97 * 1.015, 0, 1)
        self.lut = (np.stack([b, g, r], 1).reshape(256, 1, 3) * 255).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        vig = np.clip(1 - 0.22 * np.clip(rr - 0.55, 0, None) ** 1.5, 0, 1)
        self.vig = cv2.merge([(vig * 255).astype(np.uint8)] * 3)
        rng = np.random.default_rng(5)
        self.grain = []
        for _ in range(8):
            g1 = rng.normal(128, 18, (H, W)).astype(np.float32)
            g1 = cv2.GaussianBlur(g1, (0, 0), 0.7)
            self.grain.append(cv2.merge([np.clip(g1, 0, 255).astype(np.uint8)] * 3))

    def grade(self, f):
        f = cv2.LUT(f, self.lut)
        gray = cv2.merge([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)] * 3)
        f = cv2.addWeighted(f, 0.94, gray, 0.06, 0)            # slightly calmer colour
        small = cv2.resize(f, (self.W // 6, self.H // 6), interpolation=cv2.INTER_AREA)
        mist = cv2.GaussianBlur(small, (0, 0), self.W / 160)
        hal = cv2.multiply(cv2.subtract(small, (170, 170, 170, 0)), (0.5, 0.75, 1.3, 0))
        hal = cv2.GaussianBlur(hal, (0, 0), self.W / 240)
        mist = cv2.resize(mist, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        hal = cv2.resize(hal, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        inv = cv2.bitwise_not(f)
        f = cv2.add(f, cv2.multiply(mist, inv, scale=0.14 / 255))   # pro-mist bloom (screen)
        return cv2.addWeighted(f, 1.0, hal, 0.9, 0)

    def finish(self, f, i):
        f = cv2.multiply(f, self.vig, scale=1 / 255)
        return cv2.addWeighted(f, 1.0, self.grain[i % 8], 0.035, -128 * 0.035)


# ---------------------------------------------------------------- typography
IVORY = (246, 238, 224)
GOLD = (214, 176, 112)
GUJ_DIGITS = "૦૧૨૩૪૫૬૭૮૯"


def text_rgba(txt, fnt, fill, pad=12, stroke=0):
    l, t, r, b = fnt.getbbox(txt)
    im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - l, pad - t), txt, font=fnt, fill=fill + (255,))
    return im


def soft_shadow(im, radius, opacity):
    a = im.split()[3].filter(ImageFilter.GaussianBlur(radius))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sh.putalpha(a.point(lambda v: int(v * opacity)))
    return sh


class Typo:
    def __init__(self, W, H):
        self.W, self.H = W, H
        k = W / 1920
        self.k = k
        self.f_guj = font("NotoSerifGujarati.ttf", int(50 * k), 420)
        self.f_eng = font("PlayfairDisplay-Italic.ttf", int(26 * k), 400)
        self.f_num = font("NotoSerifGujarati.ttf", int(22 * k), 500)
        self.f_word = font("NotoSerifGujarati.ttf", int(290 * k), 600)
        self.f_cap = font("Cinzel.ttf", int(26 * k), 500)
        self.f_big = font("NotoSerifGujarati.ttf", int(96 * k), 350)
        self.f_gtitle = font("NotoSerifGujarati.ttf", int(34 * k), 400)
        self.cache = {}
        logo = Image.open(os.path.join(HERE, "..", "assets", "logo.png")).convert("RGBA") \
            if os.path.exists(os.path.join(HERE, "..", "assets", "logo.png")) else None
        self.logo = logo

    # layer helpers
    def blank(self):
        return Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))

    def paste(self, layer, im, x, y, alpha=1.0, blur=0.0):
        if alpha <= 0.003:
            return
        if blur > 0.3:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        if alpha < 0.999:
            a = im.split()[3].point(lambda v: int(v * alpha))
            im = im.copy()
            im.putalpha(a)
        layer.alpha_composite(im, (int(x), int(y)))

    # editorial subtitle lines
    def lines(self, layer, t):
        k = self.k
        x0, y0 = int(150 * k), int(self.H - 205 * k)
        for n, (segs, end, eng) in enumerate(LINES):
            start = segs[0][0]
            if not (start - 0.05 <= t < end + 0.6):
                continue
            out = ease((end + 0.6 - t) / 0.6) if t > end else 1.0
            # chapter mark: gold hairline + Gujarati numeral
            a = ease((t - start) / 0.8) * out
            num = text_rgba(GUJ_DIGITS[0] + GUJ_DIGITS[n + 1], self.f_num, GOLD)
            line_w = int(46 * k * ease((t - start) / 0.9))
            if line_w > 1:
                bar = Image.new("RGBA", (line_w, max(1, int(2 * k))), GOLD + (int(255 * a),))
                layer.alpha_composite(bar, (x0, y0 + int(16 * k)))
            self.paste(layer, num, x0 + int(58 * k), y0, a)
            # gujarati, segment by segment
            x = x0
            for s, txt in segs:
                key = ("seg", txt)
                if key not in self.cache:
                    im = text_rgba(txt, self.f_guj, IVORY)
                    self.cache[key] = (im, soft_shadow(im, 10 * k, 0.55))
                im, sh = self.cache[key]
                p = ease((t - s) / 0.7)
                if t >= s:
                    blur = (1 - p) * 8 * k
                    dy = (1 - p) * 14 * k
                    self.paste(layer, sh, x, y0 + 40 * k + dy, p * out)
                    self.paste(layer, im, x, y0 + 40 * k + dy, p * out, blur + (1 - out) * 6 * k)
                x += self.f_guj.getlength(txt)
            # english translation
            key = ("eng", n)
            if key not in self.cache:
                im = text_rgba(eng, self.f_eng, IVORY)
                self.cache[key] = (im, soft_shadow(im, 8 * k, 0.5))
            im, sh = self.cache[key]
            p = ease((t - start - 0.35) / 1.0)
            self.paste(layer, sh, x0, y0 + 122 * k, p * out * 0.8)
            self.paste(layer, im, x0, y0 + 122 * k, p * out * 0.78, (1 - p) * 5 * k)

    # 3D extruded word with perspective flip
    def word3d(self, txt):
        key = ("w3d", txt)
        if key in self.cache:
            return self.cache[key]
        k = self.k
        face = text_rgba(txt, self.f_word, IVORY, pad=int(60 * k))
        depth = int(22 * k)
        im = Image.new("RGBA", face.size, (0, 0, 0, 0))
        alpha = face.split()[3]
        for d in range(depth, 0, -1):
            shade = 0.35 + 0.45 * (1 - d / depth)
            col = tuple(int(c * shade) for c in GOLD)
            layer = Image.new("RGBA", face.size, col + (0,))
            layer.putalpha(alpha)
            im.alpha_composite(layer.transform(face.size, Image.AFFINE, (1, 0, -d * 0.5, 0, 1, -d), Image.BILINEAR))
        im.alpha_composite(face)
        arr = np.array(im)
        self.cache[key] = arr
        return arr

    def flip(self, arr, ang, scale=1.0):
        """Rotate an RGBA image about its horizontal axis with perspective."""
        h, w = arr.shape[:2]
        a = math.radians(ang)
        f = w * 1.6
        pts = []
        for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
            Y, Z = y * math.cos(a), y * math.sin(a)
            s = f / (f + Z)
            pts.append((w / 2 + x * s * scale, h / 2 + Y * s * scale))
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), np.float32(pts))
        return cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    def words(self, t):
        """Returns an RGBA numpy layer or None."""
        active = []
        for n in range(len(WORDS) - 1):
            s, txt = WORDS[n]
            e = WORDS[n + 1][0]
            if txt and s - 0.01 <= t < e + 0.16:
                active.append((s, e, txt))
        if not active:
            return None
        out = np.zeros((self.H, self.W, 4), np.float32)
        for s, e, txt in active:
            arr = self.word3d(txt)
            pin = ease((t - s) / 0.32)
            pout = ease((t - e) / 0.15) if t > e else 0.0
            ang = 78 * (1 - pin) - 70 * pout
            alpha = pin * (1 - pout) ** 2
            drift = 1.0 + 0.03 * min(1.0, (t - s) / 1.5)
            img = self.flip(arr, ang, drift).astype(np.float32)
            img[..., 3] *= alpha
            h, w = img.shape[:2]
            x, y = (self.W - w) // 2, int(self.H * 0.47 - h / 2)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(self.W, x + w), min(self.H, y + h)
            src = img[y0 - y:y1 - y, x0 - x:x1 - x]
            dst = out[y0:y1, x0:x1]
            a = src[..., 3:4] / 255
            dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
            dst[..., 3:4] = a * 255 + dst[..., 3:4] * (1 - a)
        return out

    # 3D title (logo) with slow rotation + light sweep
    def title(self, t):
        t0, t1 = TITLE_DST[0] / FPS, (TITLE_DST[1] + 1) / FPS
        if not (t0 <= t < t1) or self.logo is None:
            return None
        k = self.k
        key = ("title",)
        if key not in self.cache:
            lg = self.logo
            tw = int(self.W * 0.44)
            lg = lg.resize((tw, int(lg.height * tw / lg.width)), Image.LANCZOS)
            bb = lg.getbbox()
            lg = lg.crop(bb)
            pad = int(40 * k)
            a = lg.split()[3]
            base = Image.new("RGBA", (lg.width + 2 * pad, lg.height + 2 * pad), (0, 0, 0, 0))
            alpha = Image.new("L", base.size, 0)
            alpha.paste(a, (pad, pad))
            depth = int(14 * k)
            for d in range(depth, 0, -1):
                shade = 0.3 + 0.5 * (1 - d / depth)
                lay = Image.new("RGBA", base.size, tuple(int(c * shade) for c in GOLD) + (0,))
                lay.putalpha(alpha)
                base.alpha_composite(lay.transform(base.size, Image.AFFINE, (1, 0, -d * 0.4, 0, 1, -d), Image.BILINEAR))
            face = Image.new("RGBA", base.size, IVORY + (0,))
            face.putalpha(alpha)
            base.alpha_composite(face)
            self.cache[key] = (np.array(base).astype(np.float32), np.array(alpha).astype(np.float32) / 255)
            g = text_rgba("ઈશાની દવે ના ગરબા", self.f_gtitle, GOLD)
            self.cache[("gt",)] = g
        arr, face_a = self.cache[key]
        u = (t - t0) / (t1 - t0)
        pin = ease(u / 0.22)
        pout = ease((u - 0.85) / 0.15) if u > 0.85 else 0
        img = arr.copy()
        # light sweep across the face
        h, w = img.shape[:2]
        xs = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.5
        cx = (-0.3 + 1.6 * u) * w
        sweep = np.exp(-((xs - cx) / (w * 0.06)) ** 2) * face_a * 0.55
        img[..., :3] += (255 - img[..., :3]) * sweep[..., None]
        # gentle Y rotation via perspective, slow push
        ang = math.radians(16 * (1 - u) - 4)
        f = w * 1.8
        pts = []
        for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
            X, Z = x * math.cos(ang), x * math.sin(ang)
            s = f / (f + Z) * (0.94 + 0.06 * u)
            pts.append((w / 2 + X * s, h / 2 + y * s))
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), np.float32(pts))
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        img[..., 3] *= pin * (1 - pout)
        if pout > 0:
            img = cv2.GaussianBlur(img, (0, 0), 0.1 + 10 * k * pout)
        out = np.zeros((self.H, self.W, 4), np.float32)
        x, y = int(self.W * 0.40 - w / 2), int(self.H * 0.40 - h / 2)
        out[y:y + h, x:x + w] = img
        g = self.cache[("gt",)]
        gl = self.blank()
        self.paste(gl, g, x + (w - g.width) // 2, y + h + 6 * k, ease((u - 0.25) / 0.3) * (1 - pout))
        ga = np.array(gl).astype(np.float32)
        a = ga[..., 3:4] / 255
        out[..., :3] = ga[..., :3] * a + out[..., :3] * (1 - a)
        out[..., 3:4] = ga[..., 3:4] + out[..., 3:4] * (1 - a)
        return out

    def intro_outro(self, layer, t):
        k = self.k
        if t < 1.62:
            p = ease((t - 0.15) / 0.6) * (1 - ease((t - 1.2) / 0.4))
            key = ("intro",)
            if key not in self.cache:
                self.cache[key] = (text_rgba("નવરાત્રી", self.f_big, IVORY),
                                   text_rgba("N  A  V  R  A  T  R  I", self.f_cap, GOLD))
            g, c = self.cache[key]
            self.paste(layer, g, (self.W - g.width) // 2, self.H * 0.40 - g.height / 2 + (1 - p) * 10 * k, p, (1 - p) * 6 * k)
            self.paste(layer, c, (self.W - c.width) // 2, self.H * 0.40 + g.height / 2 + 4 * k, p * 0.9)
        if t >= 53.0 and self.logo is not None:
            key = ("outro",)
            if key not in self.cache:
                tw = int(self.W * 0.26)
                lg = self.logo.resize((tw, int(self.logo.height * tw / self.logo.width)), Image.LANCZOS)
                lg = lg.crop(lg.getbbox())
                face = Image.new("RGBA", lg.size, IVORY + (0,))
                face.putalpha(lg.split()[3])
                self.cache[key] = face
            face = self.cache[key]
            p = ease((t - 53.0) / 0.9)
            self.paste(layer, soft_shadow(face, 12 * k, 0.6), (self.W - face.width) // 2, self.H * 0.44 - face.height / 2, p)
            self.paste(layer, face, (self.W - face.width) // 2, self.H * 0.44 - face.height / 2, p, (1 - p) * 8 * k)


def over(f, rgba):
    """Composite float RGBA (RGB order) onto BGR uint8 frame."""
    a = rgba[..., 3:4] / 255
    rgb = rgba[..., 2::-1]
    return (f * (1 - a) + rgb * a).astype(np.uint8)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--cache", default=os.path.join(HERE, "..", ".cache_frames"))
    a = ap.parse_args()

    W = a.width
    H = int(round(W * 9 / 16 / 2)) * 2
    rep = precompute(a.src, W, H, a.models, a.cache)
    seg = Segmenter(os.path.join(a.models, "u2net_human_seg.onnx"))
    look, typo = Look(W, H), Typo(W, H)

    dur = a.dur if a.dur else DURATION - a.start
    i0 = int(round(a.start * FPS))
    nframes = int(round(dur * FPS))
    trim = ["-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}"]
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", *trim, "-i", a.src, "-map", "0:v", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf), "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)

    for k, f in enumerate(read_frames(a.src, W, H, i0, nframes)):
        i = i0 + k
        t = i / FPS
        f = rep[i] if i in rep else remove_text(f, t)
        clean = f
        f = look.grade(f)
        layer = typo.blank()
        typo.lines(layer, t)
        typo.intro_outro(layer, t)
        f = over(f, np.array(layer).astype(np.float32))
        for rgba in (typo.words(t), typo.title(t)):
            if rgba is None:
                continue
            graded = f.copy()
            f = over(f, rgba)
            if any(s <= i <= e for s, e in BEHIND_PERSON):
                m = seg.mask(clean)
                f = (f * (1 - m) + graded * m).astype(np.uint8)
        f = look.finish(f, i)
        if t > DURATION - 0.5:
            f = cv2.convertScaleAbs(f, alpha=max(0.0, (DURATION - t) / 0.5))
        writer.stdin.write(f.tobytes())
        if k % 100 == 0:
            print(f"frame {k}/{nframes} t={t:.2f}", flush=True)
    writer.stdin.close()
    writer.wait()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
