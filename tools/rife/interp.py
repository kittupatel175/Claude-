"""Minimal CPU/GPU wrapper around RIFE v4.22 (AI frame interpolation).

Weights: https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.22.pkl
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .IFNet_HDv3_v4_22 import IFNet


class Rife:
    def __init__(self, weights, scale=1.0, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sd = torch.load(weights, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
        self.net = IFNet(scale, False)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval().to(self.device)
        self.scale = scale
        self._grid = {}

    def _setup(self, ph, pw):
        if (ph, pw) not in self._grid:
            div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], device=self.device)
            hor = torch.linspace(-1.0, 1.0, pw, device=self.device).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
            ver = torch.linspace(-1.0, 1.0, ph, device=self.device).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
            self._grid[(ph, pw)] = (div, torch.cat([hor, ver], 1))
        return self._grid[(ph, pw)]

    def _tensor(self, img, ph, pw):
        t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        h, w = img.shape[:2]
        return F.pad(t, (0, pw - w, 0, ph - h)).to(self.device)

    @torch.inference_mode()
    def between(self, a, b, ts):
        """Frames between BGR uint8 images a and b at fractions ts (0..1)."""
        h, w = a.shape[:2]
        m = max(64, int(64 / self.scale))
        ph, pw = math.ceil(h / m) * m, math.ceil(w / m) * m
        div, grid = self._setup(ph, pw)
        i0, i1 = self._tensor(a, ph, pw), self._tensor(b, ph, pw)
        f0, f1 = self.net.encode(i0), self.net.encode(i1)
        out = []
        for t in ts:
            step = torch.full([1, 1, ph, pw], float(t), device=self.device)
            r = self.net(i0, i1, step, div, grid, f0, f1)[0, :, :h, :w]
            img = (r.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            out.append(np.ascontiguousarray(img[..., ::-1]))
        return out
