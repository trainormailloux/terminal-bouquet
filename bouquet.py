"""
Spinning bouquet of flowers for the terminal, drawn with characters.

Setup: pip install numpy
Run:   python bouquet.py
Stop:  Ctrl+C
Works best in Windows Terminal. Zoom out (Ctrl + minus) for more detail.
"""

import math
import os
import shutil
import sys
import time

import numpy as np

os.system("")  # turns on color codes in the Windows command prompt

LIGHT = np.array([0.4, 0.75, -0.55])
LIGHT /= np.linalg.norm(LIGHT)

# (petal color, center color, petal count, style)
FLOWERS = [
    ((255, 210, 40), (130, 70, 20), 10, "thin"),     # sunflower (middle)
    ((235, 35, 55), (255, 220, 70), 5, "round"),     # red
    ((250, 250, 250), (255, 200, 0), 10, "thin"),    # white daisy
    ((255, 120, 190), (255, 240, 140), 6, "round"),  # pink
    ((165, 80, 235), (255, 230, 90), 5, "round"),    # purple
    ((255, 140, 30), (120, 45, 10), 8, "pointy"),    # orange
    ((80, 150, 255), (255, 255, 200), 5, "round"),   # blue
    ((255, 70, 140), (255, 220, 80), 8, "pointy"),   # hot pink
]
STEM = (40, 160, 50)
LEAF = (60, 190, 70)
WRAP = (240, 215, 170)
RIBBON = (215, 25, 65)

# kinds of things, and the characters used to fill them (dim -> bright)
PETAL, CENTER, STEM_K, LEAF_K, WRAP_K, RIBBON_K = range(6)
FILLS = {
    PETAL: "**##",
    CENTER: "oo@@",
    LEAF_K: "%%%%",
    WRAP_K: "::==",
    RIBBON_K: "====",
}
# object ids: every flower head and center gets its own id so each one gets its own outline
OBJ_STEM, OBJ_WRAP, OBJ_RIBBON, OBJ_LEAF = 0, 1, 2, 3
OBJ_KIND = {OBJ_STEM: STEM_K, OBJ_WRAP: WRAP_K, OBJ_RIBBON: RIBBON_K, OBJ_LEAF: LEAF_K}
# flower i: petals get ids 20 + 20*i + petal number, the center gets 20 + 20*i + 19
for _i in range(len(FLOWERS) + 1):
    for _p in range(19):
        OBJ_KIND[20 + 20 * _i + _p] = PETAL
    OBJ_KIND[20 + 20 * _i + 19] = CENTER
KIND_OF = np.zeros(20 + 20 * (len(FLOWERS) + 1), dtype=int)
for _o, _k in OBJ_KIND.items():
    KIND_OF[_o] = _k


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def basis(d):
    up = np.array([0, 0, 1.0]) if abs(d[1]) > 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, up))
    return u, np.cross(d, u)


class Solid:
    """3D points for stems, leaves, wrap and ribbon."""

    def __init__(self):
        self.parts = {k: [] for k in ("p", "n", "c", "two", "dark", "obj", "axis")}

    def add(self, p, n, color, two, obj, dark=1.0, axis=(0.0, 1.0, 0.0)):
        n = np.broadcast_to(n, p.shape).reshape(-1, 3)
        p = p.reshape(-1, 3)
        a = self.parts
        a["p"].append(p)
        a["n"].append(n)
        a["c"].append(np.broadcast_to(np.array(color, float), p.shape))
        a["two"].append(np.full(len(p), two))
        a["dark"].append(np.broadcast_to(np.asarray(dark, float), p.shape[:1]).ravel())
        a["obj"].append(np.full(len(p), obj))
        a["axis"].append(np.broadcast_to(np.array(axis, float), p.shape))

    def done(self):
        return {k: np.concatenate(v) for k, v in self.parts.items()}


def tube(solid, start, end, radius, steps=200, around=8):
    d = unit(end - start)
    u, v = basis(d)
    t = np.linspace(0, 1, steps)[:, None, None]
    a = np.linspace(0, math.tau, around, endpoint=False)[None, :, None]
    off = u * np.cos(a) + v * np.sin(a)
    p = start + (end - start) * t + off * radius
    solid.add(p, np.broadcast_to(off, p.shape), STEM, False, OBJ_STEM, axis=d)


def petal_reach(style, petals, th):
    c = np.abs(np.cos(petals / 2 * th))
    if style == "thin":
        return c ** 3
    if style == "pointy":
        return c ** 1.5
    return c ** 0.6


def flower_face(i, size, petals, style, color, center, spin):
    """A flat flower drawn facing the viewer: offsets in screen units."""
    th = np.linspace(0, math.tau, 500, endpoint=False)[:, None]
    b = np.linspace(0, 1, 40)[None, :]
    reach = petal_reach(style, petals, th + spin)
    r = size * (0.3 + 0.7 * reach) * b
    keep = np.broadcast_to((reach > (0.18 if style == "thin" else 0.4)) | (b < 0.36), r.shape)  # gaps
    which = np.round((th + spin) * petals / math.tau).astype(int) % petals  # petal number
    off = np.stack(np.broadcast_arrays(r * np.cos(th), r * np.sin(th)), -1)[keep]
    # petals: a little darker toward the tips and down the middle line
    # slightly darker toward the edges of each petal
    shade = (0.7 + 0.3 * reach) * (1.0 - 0.2 * b ** 2)
    pet = {"off": off, "k": np.broadcast_to(shade, r.shape)[keep],
           "c": np.broadcast_to(np.array(color, float), (len(off), 3)),
           "obj": np.full(len(off), 20 + 20 * i), "bias": np.zeros(len(off)),
           "petal": np.broadcast_to(which, r.shape)[keep]}

    csize = size * 0.32
    th = np.linspace(0, math.tau, 160, endpoint=False)[:, None]
    b = np.linspace(0, 1, 14)[None, :]
    r = csize * b
    off = np.stack(np.broadcast_arrays(r * np.cos(th), r * np.sin(th)), -1).reshape(-1, 2)
    cen = {"off": off, "k": np.broadcast_to(1.05 - 0.3 * b ** 2, (len(th), b.shape[1])).ravel(),
           "c": np.broadcast_to(np.array(center, float), (len(off), 3)),
           "obj": np.full(len(off), 20 + 20 * i + 19), "bias": np.full(len(off), 0.05),
           "petal": np.full(len(off), -1)}
    return pet, cen


def build_bouquet():
    solid = Solid()
    faces = []
    head_pos = []
    rnd = np.random.default_rng(5)
    tie = np.array([0.0, -0.3, 0.0])

    heads = [(0.0, 0.0)] + [(0.9, i * math.tau / 5 + 0.3) for i in range(5)]

    for i, (tilt, phi) in enumerate(heads):
        d = np.array([math.sin(tilt) * math.cos(phi), math.cos(tilt), math.sin(tilt) * math.sin(phi)])
        length = 1.25 - 0.17 * tilt + rnd.uniform(-0.04, 0.04)
        head = tie + d * length
        color, center, petals, style = FLOWERS[i % len(FLOWERS)]
        size = 0.5 if i == 0 else 0.46

        tube(solid, tie, head, 0.012)
        tube(solid, np.array([d[0] * 0.08, -1.0, d[2] * 0.08]), tie, 0.012, steps=100)

        # leaf
        if i % 2 == 1:
            u, v = basis(d)
            base = tie + d * length * rnd.uniform(0.3, 0.45)
            side = u if i % 4 == 1 else -u
            leaf_dir = unit(side * 0.85 + d * 0.55)
            leaf_n = unit(np.cross(leaf_dir, v))
            s = np.linspace(0, 1, 90)[:, None]
            f = np.linspace(-1, 1, 25)[None, :]
            w = np.sin(s * math.pi) * 0.09
            p = base + leaf_dir * (s * 0.45)[..., None] + v * (f * w)[..., None]
            solid.add(p, leaf_n, LEAF, True, OBJ_LEAF)

        pet, cen = flower_face(i, size, petals, style, color, center, rnd.uniform(0, math.tau))
        for part in (pet, cen):
            part["head"] = np.full(len(part["off"]), i)
            faces.append(part)
        head_pos.append(head)

    # paper wrap with folds
    th = np.linspace(0, math.tau, 300, endpoint=False)[:, None]
    t = np.linspace(0, 1, 90)[None, :]
    y = -0.92 + t * 0.66
    r = (0.06 + t * 0.26) * (1 + 0.08 * np.sin(th * 7))
    p = np.stack(np.broadcast_arrays(r * np.cos(th), y, r * np.sin(th)), axis=-1)
    n = unit(np.stack(np.broadcast_arrays(np.cos(th), 0.35 + 0 * t, np.sin(th)), axis=-1))
    dark = np.where(np.sin(th * 7) > 0.9, 0.75, 1.0) * np.ones_like(t)
    solid.add(p, n, WRAP, True, OBJ_WRAP, dark.ravel())

    # ribbon
    th = np.linspace(0, math.tau, 300, endpoint=False)[:, None]
    y = np.linspace(-0.66, -0.6, 10)[None, :]
    r = 0.06 + (y + 0.92) / 0.66 * 0.26 + 0.025
    p = np.stack(np.broadcast_arrays(r * np.cos(th), y, r * np.sin(th)), axis=-1)
    n = np.stack(np.broadcast_arrays(np.cos(th), 0 * y, np.sin(th)), axis=-1)
    solid.add(p, n, RIBBON, False, OBJ_RIBBON)

    face = {k: np.concatenate([f[k] for f in faces]) for k in faces[0]}
    return {"solid": solid.done(), "face": face, "heads": np.array(head_pos)}


def rasterize(model, angle, w, h):
    """Project everything into a w x h grid of square pixels (closest thing wins)."""
    ca, sa = math.cos(angle), math.sin(angle)
    tilt = 0.3
    ct, st = math.cos(tilt), math.sin(tilt)
    scale = min(h / 2.6, w / 2.9)
    cx, cy = w / 2, h * 0.5 + scale * 0.2
    dist = 5.0

    def turn(V):
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        x1 = x * ca + z * sa
        z1 = -x * sa + z * ca
        return x1, y * ct - z1 * st, y * st + z1 * ct

    # stems, leaves, wrap, ribbon: real 3D
    S = model["solid"]
    x, y, z = turn(S["p"])
    zz_s = z + dist
    f = scale * dist / zz_s
    sx_s, sy_s = cx + x * f, cy - y * f
    nx, ny, nz = turn(S["n"])
    lum = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    lum = np.where(S["two"], np.abs(lum), lum)
    k_s = (0.45 + 0.55 * np.clip(lum, 0, 1)) * S["dark"]
    axx, axy, _ = turn(S["axis"])

    # flower heads: always face the viewer, flowers at the back are a bit dimmer
    F = model["face"]
    hx, hy, hz = turn(model["heads"])
    hi = F["head"]
    zz_h = hz[hi] + dist
    f = scale * dist / zz_h
    sx_f = cx + hx[hi] * f + F["off"][:, 0] * f
    sy_f = cy - hy[hi] * f - F["off"][:, 1] * f
    front = np.clip(-hz[hi], -1, 1)
    k_f = F["k"] * (0.82 + 0.18 * front)
    zz_f = zz_h - F["bias"]

    sx = np.concatenate([sx_s, sx_f]).astype(np.int64)
    sy = np.concatenate([sy_s, sy_f]).astype(np.int64)
    zz = np.concatenate([zz_s, zz_f])
    k = np.concatenate([k_s, k_f])
    col = np.concatenate([S["c"], F["c"]])
    obj = np.concatenate([S["obj"], F["obj"]])
    # direction: along the stem for stems, outward from the flower center for petals
    dx = np.concatenate([axx, F["off"][:, 0]])
    dy = np.concatenate([axy, F["off"][:, 1]])
    petal = np.concatenate([np.full(len(axx), -1), F["petal"]])

    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    idx = (sy * w + sx)[ok]
    depth = ((zz[ok] - zz[ok].min()) * 1e5).astype(np.int64)
    order = np.argsort(idx * (1 << 24) + depth)
    sidx = idx[order]
    first = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
    pick = np.flatnonzero(ok)[order[first]]
    pix = sidx[first]

    out = {
        "col": np.zeros((w * h, 3)),
        "light": np.zeros(w * h),
        "obj": np.full(w * h, -1),
        "dir": np.zeros((w * h, 2)),
        "petal": np.full(w * h, -1),
    }
    out["petal"][pix] = petal[pick]
    out["col"][pix] = col[pick]
    out["light"][pix] = k[pick]
    out["obj"][pix] = obj[pick]
    out["dir"][pix, 0] = dx[pick]
    out["dir"][pix, 1] = dy[pick]
    return out


def to_text(r, cols, rows):
    """Turn each 2x4 block of pixels into one colored character."""
    def cells(a, extra=()):
        a = a.reshape((rows, 4, cols, 2) + extra)
        a = a.transpose((0, 2, 1, 3) + tuple(range(4, 4 + len(extra))))
        return a.reshape((rows * cols, 8) + extra)

    c = cells(r["col"], (3,))
    k = cells(r["light"])
    o = cells(r["obj"])
    dvec = cells(r["dir"], (2,))
    pet = cells(r["petal"])
    filled = o >= 0

    # the main object in each cell (stems count extra so thin stems stay visible)
    same = (o[:, :, None] == o[:, None, :]) & filled[:, None, :]
    votes = same.sum(2) * np.where(o == OBJ_STEM, 3, 1) * filled
    main = o[np.arange(len(o)), votes.argmax(1)]
    kind = KIND_OF[np.maximum(main, 0)]
    sel = (o == main[:, None]).astype(float)
    cov = sel.sum(1)
    n_sel = np.maximum(cov, 1)

    bright = (k * sel).sum(1) / n_sel
    base = (c * sel[..., None]).sum(1) / n_sel[:, None]
    rgb = np.clip(base * (0.55 + 0.5 * bright)[:, None], 0, 255)

    level = np.clip(((bright - 0.35) / 0.65) * 4, 0, 3).astype(int)
    chars = np.full(len(o), " ", dtype="<U1")
    for kind_id, ramp in FILLS.items():
        chars = np.where(kind == kind_id, np.array(list(ramp))[level], chars)

    # outlines: which side of the cell the shape sits on (pixel bit = row * 2 + col)
    gx = (sel[:, 1::2].sum(1) - sel[:, 0::2].sum(1)) / 4
    gy = (sel[:, 4:8].sum(1) - sel[:, 0:4].sum(1)) / 4
    gax, gay = np.abs(gx), np.abs(gy)
    shaped = np.isin(kind, [PETAL, LEAF_K, WRAP_K])
    outline = shaped & (cov >= 2) & (cov <= 6)
    horiz = outline & (gay > 1.5 * gax)
    vert = outline & (gax > 1.5 * gay)
    diag = outline & ~horiz & ~vert
    chars = np.where(horiz & (gy > 0), "_", chars)
    chars = np.where(horiz & (gy <= 0), "-", chars)
    chars = np.where(vert & (gx > 0), "(", chars)
    chars = np.where(vert & (gx <= 0), ")", chars)
    chars = np.where(diag & (gx * gy > 0), "/", chars)
    chars = np.where(diag & (gx * gy <= 0), "\\", chars)
    chars = np.where(shaped & (cov < 2), "'", chars)

    # where two petals of the same flower meet: a darker spoke pointing out from the center
    p_sel = np.where(sel > 0, pet, -1)
    p_max = p_sel.max(1)
    p_min = np.where(p_sel >= 0, p_sel, 99).min(1)
    spoke = (kind == PETAL) & (cov >= 7) & (p_min >= 0) & (p_max != p_min)
    rx = (dvec[..., 0] * sel).sum(1)
    ry = (dvec[..., 1] * sel).sum(1)
    rax, ray = np.abs(rx), np.abs(ry)
    spoke_ch = np.where(rax < 0.4 * ray, "|",
                        np.where(ray < 0.4 * rax, "-", np.where(rx * ry > 0, "/", "\\")))
    chars = np.where(spoke, spoke_ch, chars)
    rgb = np.where(spoke[:, None], rgb * 0.6, rgb)

    # stems: a line character pointing the way the stem goes on screen
    sx = (dvec[..., 0] * sel).sum(1)
    sy = (dvec[..., 1] * sel).sum(1)
    sax, say = np.abs(sx), np.abs(sy)
    stem_ch = np.where(sax < 0.45 * say, "|",
                       np.where(say < 0.45 * sax, "-", np.where(sx * sy > 0, "/", "\\")))
    chars = np.where(kind == STEM_K, stem_ch, chars)

    chars = np.where(main < 0, " ", chars)
    return chars, rgb.astype(int)


def render(model, angle, cols, rows):
    chars, rgb = to_text(rasterize(model, angle, cols * 2, rows * 4), cols, rows)
    out = []
    last = None
    i = 0
    for row in range(rows):
        for _ in range(cols):
            ch = chars[i]
            if ch != " ":
                c = (rgb[i, 0], rgb[i, 1], rgb[i, 2])
                if c != last:
                    out.append(f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m")
                    last = c
            out.append(ch)
            i += 1
        if row < rows - 1:
            out.append("\n")
    out.append("\x1b[0m")
    return "".join(out)


def main():
    model = build_bouquet()
    sys.stdout.write("\x1b[?25l\x1b[2J")
    angle = 0.0
    try:
        while True:
            size = shutil.get_terminal_size((100, 40))
            cols, rows = size.columns, size.lines - 1
            frame = render(model, angle, cols, rows)
            sys.stdout.write("\x1b[H" + frame)
            sys.stdout.flush()
            angle += 0.04
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
