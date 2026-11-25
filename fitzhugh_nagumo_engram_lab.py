#!/usr/bin/env python3
"""
FitzHugh-Nagumo engram lab  -  CREB-gated allocation, SWR-gated STDP, multi-memory recall
==========================================================================================

A grid of coupled FitzHugh-Nagumo (FHN) neurons stores several memories and recalls each one
from a partial cue.  The grid is drawn as a field of light bulbs: a neuron that spikes lights up.

Model in one paragraph
  * Every neuron has a CREB level (circadian + ultradian rhythm, random phase per neuron).  CREB sets
    its resting drive I = I0 + kc * CREB, i.e. how close it sits to the FHN Hopf (firing) threshold.
  * Memory m is a stimulus pattern presented at its own hour.  Neurons pushed past the Hopf threshold
    fire; that sparse set is the engram of memory m (CREB allocation: cells that happen to be excitable
    at that hour are recruited, so memories encoded close together in time share cells).
  * While a sharp-wave-ripple (SWR) gate is open, STDP updates the recurrent weights W_ij.  Plasticity
    is scaled by each cell's CREB level.  Learning uses soft bounds and a per-neuron incoming-weight
    budget (synaptic scaling), so weights cannot run away.
  * A partial cue drives a random fraction of one memory's stimulus.  Learned weights raise the drive of
    the un-cued partners across the Hopf point, so the rest of the engram fires (pattern completion).
    Global feedback inhibition keeps the other memories quiet.
  * Every memory is cued in one batched run, which yields a recall / cross-talk matrix.

Run
    python3 fhn_memory_interactive.py                  interactive window (needs a GUI backend)
    python3 fhn_memory_interactive.py --png out.png    render the dashboard headlessly and exit
    python3 fhn_memory_interactive.py --gif out.gif    export the bulb-field retrieval movie
    python3 fhn_memory_interactive.py --selftest       run the built-in sanity checks
    python3 fhn_memory_interactive.py --help           all options

Needs: numpy, matplotlib >= 3.7 (Pillow for GIF export, installed with matplotlib).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, replace, fields
from types import SimpleNamespace

import numpy as np

__version__ = "2.0"


# ----------------------------------------------------------------------------- command line
def _parse_cli(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive FitzHugh-Nagumo engram lab (multi-memory, light-bulb field).")
    p.add_argument("--seed", type=int, default=7, help="world seed (CREB phases, patterns, cue masks)")
    p.add_argument("--grid", type=int, default=12, help="grid side length G (network has G*G neurons)")
    p.add_argument("--n-mem", type=int, default=None, help="number of stored memories (1-8)")
    p.add_argument("--mode", default="result",
                   choices=["result", "retrieval", "encoding", "engrams", "creb"],
                   help="what the bulb field shows at start-up")
    p.add_argument("--headless", action="store_true", help="use the non-interactive Agg backend")
    p.add_argument("--png", metavar="FILE", help="render the full dashboard to FILE and exit")
    p.add_argument("--gif", metavar="FILE", help="export the retrieval movie of the bulb field and exit")
    p.add_argument("--selftest", action="store_true", help="run sanity checks and exit")
    return p.parse_args(argv)


CLI = _parse_cli(None if __name__ == "__main__" else [])

import matplotlib
if CLI.headless or CLI.png or CLI.gif or CLI.selftest:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- fixed constants
G = max(6, min(24, CLI.grid))
M = G * G
a, b, eps = 0.7, 0.8, 0.08             # FitzHugh-Nagumo
dt, tau_stdp = 0.1, 10.0               # Euler step, STDP trace decay
I0 = 0.10                              # baseline resting drive (CREB adds kc * c)
T_PRE, T_STIM, T_TAIL = 20.0, 120.0, 40.0    # stimulus onset / duration / tail (model time units)
T_TOTAL = T_PRE + T_STIM + T_TAIL
FIRE_MIN = 2                           # spikes needed to call a neuron "firing"
MAX_MEM = 8
LETTERS = "ABCDEFGH"

xy = np.array([(i // G, i % G) for i in range(M)], float)
D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
W0 = 0.01 * np.exp(-D ** 2 / (2 * 1.2 ** 2))         # weak local baseline coupling
np.fill_diagonal(W0, 0)
KERN = np.exp(-D ** 2 / (2 * 0.7 ** 2))


# ----------------------------------------------------------------------------- FHN helpers
def drive_at(v):
    """Constant drive I for which (v, (v+a)/b) is the fixed point."""
    return v ** 3 / 3 + (1 / b - 1) * v + a / b


_VH = np.sqrt(1 - eps * b)
I_HOPF = float(drive_at(-_VH))        # lower Hopf point: below this the neuron rests      (~0.331)
I_HOPF_HI = float(drive_at(_VH))      # upper Hopf point: above this it is stuck depolarised (~1.419)


def fixed_point(I):
    """Fixed point (v, n) of the FHN neuron at constant drive I (Newton on a monotone cubic)."""
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(80):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        v = v - f / (v ** 2 + (1 / b - 1))
    return v, (v + a) / b


def is_stable(v):
    return bool((1 - v ** 2 - eps * b < 0) and (eps * (1 - b * (1 - v ** 2)) > 0))


def regime(I):
    """'silent' below the lower Hopf point, 'fires' inside the band, 'blocked' above the upper one."""
    return "silent" if I <= I_HOPF else ("fires" if I < I_HOPF_HI else "blocked")


# ----------------------------------------------------------------------------- parameters
@dataclass
class Params:
    # --- world & encoding
    n_mem: int = 3            # memories stored
    t_enc: float = 10.0       # clock hour of the first encoding episode
    spacing: float = 3.0      # hours between consecutive episodes
    sim: float = 0.0          # similarity of the stimulus patterns (0 = independent, 1 = identical)
    kc: float = 0.22          # CREB excitability gain
    s0: float = 0.28          # stimulus strength
    enc_rec: float = 1.0      # recurrence while encoding (0 = feedforward only, 1 = full W)
    swr_off: float = 15.0     # SWR gate onset relative to stimulus onset
    swr_w: float = 60.0       # SWR gate width
    # --- plasticity & network
    eta: float = 0.12         # STDP LTP rate
    ratio: float = 0.5        # LTD / LTP ratio
    wmax: float = 0.30        # per-synapse cap
    budget: float = 1.2       # cap on the summed incoming weight of every neuron (synaptic scaling)
    tau_s: float = 5.0        # synaptic decay time
    g_inh: float = 1.0        # global feedback inhibition
    noise: float = 0.0        # membrane noise amplitude
    # --- retrieval
    cue_mem: int = 1          # which memory is cued (1-based; display + movie only)
    ret_dt: float = 0.0       # hours between the last encoding and retrieval
    cue_f: float = 0.40       # fraction of the stimulus delivered as cue
    cue_gain: float = 1.0     # cue strength relative to the stimulus
    cue_noise: float = 0.0    # fraction of cells that receive a spurious (distractor) cue
    # --- display only
    s_on: float = 0.5         # presynaptic activity assumed in the phase-plane nullclines
    speed: float = 1.0        # movie speed


DEFAULTS = Params()
# parameter groups decide how much has to be recomputed when a slider moves
ENC_KEYS = ("n_mem", "t_enc", "spacing", "sim", "kc", "s0", "enc_rec", "swr_off", "swr_w", "eta", "ratio", "wmax",
            "budget", "tau_s", "g_inh", "noise")
RET_KEYS = ("ret_dt", "cue_f", "cue_gain", "cue_noise")
VIEW_KEYS = ("cue_mem", "s_on", "speed")


def clean(P: Params) -> Params:
    """Clamp parameters to legal values (robust against odd slider / CLI input)."""
    P = replace(P)
    P.n_mem = int(np.clip(round(P.n_mem), 1, MAX_MEM))
    P.cue_mem = int(np.clip(round(P.cue_mem), 1, P.n_mem))
    P.sim = float(np.clip(P.sim, 0, 1))
    P.enc_rec = float(np.clip(P.enc_rec, 0, 1))
    P.cue_f = float(np.clip(P.cue_f, 0, 1))
    P.cue_noise = float(np.clip(P.cue_noise, 0, 1))
    P.wmax = max(P.wmax, 1e-3)
    P.budget = max(P.budget, 1e-3)
    P.tau_s = max(P.tau_s, 0.5)
    P.swr_w = max(P.swr_w, 1.0)
    return P


# ----------------------------------------------------------------------------- the random world
class World:
    """Everything random: CREB phases, stimulus fields, cue masks, noise stream."""

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.seed = seed
        self.phi = r.uniform(0, 2 * np.pi, M)
        self.psi = r.uniform(0, 2 * np.pi, M)

        def field():
            f = KERN @ r.standard_normal(M)
            return (f - f.mean()) / f.std()

        self.f_shared = field()
        self.f_unique = np.stack([field() for _ in range(MAX_MEM)])
        self.noise_seed = int(r.integers(1 << 30))
        self.new_cues(r)

    def new_cues(self, r=None):
        r = np.random.default_rng() if r is None else r
        self.u_cue = r.random((MAX_MEM, M))             # which cells carry the cue
        self.u_dis = r.random((MAX_MEM, M))             # which cells get a distractor
        self.a_dis = r.uniform(0.5, 1.0, (MAX_MEM, M))  # distractor amplitude

    def pattern(self, m, sim):
        """Sparse stimulus profile p in [0,1] of memory m (smooth field -> rank -> u^4)."""
        f = np.sqrt(sim) * self.f_shared + np.sqrt(1 - sim) * self.f_unique[m]
        u = (np.argsort(np.argsort(f)) + 0.5) / M
        return u ** 4

    def creb(self, t_h):
        return np.clip(0.5 + 0.3 * np.sin(2 * np.pi * t_h / 24 + self.phi)
                       + 0.2 * np.sin(2 * np.pi * t_h / 1.5 + self.psi), 0, 1)


# ----------------------------------------------------------------------------- simulation
def encode_episode(W, I_rest, I_ext, P, sigma, rng):
    """One encoding episode: M coupled FHN cells, STDP while the SWR gate is open.

    W is [post, pre].  Returns (spike counts, new W, voltage trace [steps, M], SWR start time).
    """
    W = W.copy()
    swr_start = T_PRE + P.swr_off
    swr_end = swr_start + P.swr_w
    t_on, t_off = T_PRE, T_PRE + T_STIM
    n = int(round(max(T_TOTAL, swr_end + 5) / dt))
    v, w = fixed_point(I_rest)
    s = np.zeros(M); x = np.zeros(M)
    armed = np.ones(M, bool)
    counts = np.zeros(M, int)
    V = np.empty((n, M), np.float32)
    eta_p, eta_d, wmax, bud = P.eta, P.eta * P.ratio, P.wmax, P.budget
    dec_s, dec_t = np.exp(-dt / P.tau_s), np.exp(-dt / tau_stdp)
    sq = P.noise * np.sqrt(dt)
    rec_g = getattr(P, "enc_rec", 1.0)
    for k in range(n):
        t = k * dt
        I = I_rest + rec_g * (W @ s) - P.g_inh * s.mean()
        if t_on <= t < t_off:
            I = I + I_ext
        vn = v + dt * (v - v ** 3 / 3 - w + I)
        if sq:
            vn = vn + sq * rng.standard_normal(M)
        w = w + dt * eps * (v + a - b * w)
        spk = armed & (vn > 0)
        armed = (armed & ~spk) | (vn < -0.5)
        v = vn
        V[k] = v
        if t >= t_on:
            counts += spk
        if swr_start <= t < swr_end and spk.any():
            idx = np.flatnonzero(spk)
            # LTP onto the spiking (post) cells from recently active (pre) cells, soft bound at wmax
            W[idx] += eta_p * sigma[idx, None] * (wmax - W[idx]) * x[None, :]
            # LTD from the spiking (pre) cells onto recently active (post) cells, multiplicative
            W[:, idx] -= eta_d * (sigma * x)[:, None] * W[:, idx]
            np.clip(W, 0, wmax, out=W)
            W[idx, idx] = 0
            # synaptic scaling: the summed incoming weight of a neuron never exceeds the budget
            rs = W[idx].sum(1)
            over = rs > bud
            if over.any():
                W[idx[over]] *= (bud / rs[over])[:, None]
        s = np.where(spk, 1.0, s * dec_s)        # saturating synaptic gate, 0..1
        x = x * dec_t + spk
    return counts, W, V, swr_start


def retrieve_batch(Wr, I_rest, I_ext, P, rng):
    """Cue every memory at once.  I_ext is [B, M]; returns counts, V, n, I_total (all [..., B, M])."""
    B = I_ext.shape[0]
    t_on, t_off = T_PRE, T_PRE + T_STIM
    n = int(round(T_TOTAL / dt))
    v, w = fixed_point(I_rest)
    v = np.tile(v, (B, 1)); w = np.tile(w, (B, 1))
    s = np.zeros((B, M)); armed = np.ones((B, M), bool)
    counts = np.zeros((B, M), int)
    V = np.empty((n, B, M), np.float32)
    Nn = np.empty((n, B, M), np.float32)
    Itot = np.empty((n, B, M), np.float32)
    dec_s = np.exp(-dt / P.tau_s)
    sq = P.noise * np.sqrt(dt)
    WT = np.ascontiguousarray(Wr.T)
    for k in range(n):
        t = k * dt
        I = I_rest + s @ WT - P.g_inh * s.mean(1, keepdims=True)
        if t_on <= t < t_off:
            I = I + I_ext
        vn = v + dt * (v - v ** 3 / 3 - w + I)
        if sq:
            vn = vn + sq * rng.standard_normal((B, M))
        w = w + dt * eps * (v + a - b * w)
        spk = armed & (vn > 0)
        armed = (armed & ~spk) | (vn < -0.5)
        v = vn
        V[k] = v; Nn[k] = w; Itot[k] = I
        if t >= t_on:
            counts += spk
        s = np.where(spk, 1.0, s * dec_s)
    return counts, V, Nn, Itot


# ----------------------------------------------------------------------------- experiment
_ENC_CACHE: dict = {}


def encoding(world, P, gate):
    """Sequentially encode P.n_mem memories (cached on everything that influences it)."""
    key = (world.seed, id(world), gate) + tuple(getattr(P, k) for k in ENC_KEYS)
    hit = _ENC_CACHE.get(key)
    if hit is not None:
        return hit
    n = P.n_mem
    W = W0.copy()
    rng = np.random.default_rng(world.noise_seed)
    E = SimpleNamespace(n=n, hours=[], c=[], I_rest=[], eng=[], cnt=[], V=[], swr_start=[], pat=[], W_after=[])
    for m in range(n):
        h = P.t_enc + m * P.spacing
        c = world.creb(h)
        I_rest = I0 + P.kc * c
        pat = world.pattern(m, P.sim)
        sigma = c if gate else np.ones(M)
        cnt, W, V, sw = encode_episode(W, I_rest, P.s0 * pat, P, sigma, rng)
        E.hours.append(h); E.c.append(c); E.I_rest.append(I_rest); E.pat.append(pat)
        E.cnt.append(cnt); E.eng.append(cnt >= FIRE_MIN); E.V.append(V); E.swr_start.append(sw)
        E.W_after.append(W.copy())
    E.W = W
    E.eng = np.array(E.eng)
    E.cnt = np.array(E.cnt)
    if len(_ENC_CACHE) > 6:
        _ENC_CACHE.pop(next(iter(_ENC_CACHE)))
    _ENC_CACHE[key] = E
    return E


def run_experiment(world, P, naive=False, gate=True):
    """Encode all memories, then cue each one.  Returns a SimpleNamespace with everything the UI needs."""
    P = clean(P)
    E = encoding(world, P, gate)
    n = E.n
    t_ret = P.t_enc + (n - 1) * P.spacing + P.ret_dt
    c_ret = world.creb(t_ret)
    I_ret = I0 + P.kc * c_ret

    mask = world.u_cue[:n] < P.cue_f                                    # [n, M] cue lines
    dis = (world.u_dis[:n] < P.cue_noise) & ~mask                       # [n, M] distractors
    ext = P.cue_gain * P.s0 * (np.array(E.pat) * mask + 0.7 * world.a_dis[:n] * dis)
    driven = ext > 0

    Wr = W0 if naive else E.W
    rng = np.random.default_rng(world.noise_seed + 7919)
    cnt, V, Nn, Itot = retrieve_batch(Wr, I_ret, ext, P, rng)
    fired = cnt >= FIRE_MIN

    eng = E.eng
    union = eng.any(0)
    rec = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            sel = eng[j] & ~driven[i] if i == j else eng[j] & ~eng[i] & ~driven[i]
            if sel.any():
                rec[i, j] = fired[i][sel].mean()
    fp = np.array([fired[i][~union & ~driven[i]].mean() if (~union & ~driven[i]).any() else np.nan
                   for i in range(n)])
    diag = np.diag(rec)
    off = rec[~np.eye(n, dtype=bool)]
    peak = Itot.max(0)                                                  # [n, M] peak total drive
    blocked = np.array([(peak[i][eng[i]] >= I_HOPF_HI).sum() for i in range(n)])
    inter = [(eng[i] & eng[j]).sum() / max(1, min(eng[i].sum(), eng[j].sum()))
             for i in range(n) for j in range(i + 1, n)]
    rowsum = Wr.sum(1)
    return SimpleNamespace(
        P=P, E=E, n=n, naive=naive, t_ret=t_ret, c_ret=c_ret, I_ret=I_ret, W=E.W, Wr=Wr, mask=mask, dis=dis,
        ext=ext, driven=driven, cnt=cnt, fired=fired, V=V, Nn=Nn, Itot=Itot, peak=peak, rec=rec, fp=fp,
        recall=float(np.nanmean(diag)) if np.isfinite(diag).any() else np.nan,
        cross=float(np.nanmean(off)) if n > 1 and np.isfinite(off).any() else np.nan,
        fp_mean=float(np.nanmean(fp)) if np.isfinite(fp).any() else np.nan,
        blocked=blocked, spont=int((I_ret > I_HOPF).sum()),
        overlap=float(np.mean(inter)) if inter else np.nan,
        rowsum_max=float(rowsum.max()), W_max=float(Wr.max()),
        peak_drive=float(peak[eng].max()) if eng.any() else float(peak.max()),
        dW=float(E.W[np.ix_(union, union)].mean() - W0[np.ix_(union, union)].mean()) if union.any() else np.nan)


# ----------------------------------------------------------------------------- colours, classification & scene rendering
BG = "#081014"
PANEL = "#101c22"
PANEL2 = "#16262e"
FG = "#d8e6ec"
MUTED = "#708a96"
ACCENT = "#58d5ff"
WARN = "#ff7a59"

MEM_COLORS = ["#3ddc84", "#5fb7ff", "#ff6ec7", "#ffd27a", "#b48cff", "#ff8a5b", "#5ce1e6", "#d4ff59"]


def rgb(hex_str):
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], float)


C_CUED = rgb("#5fb7ff")
C_RECALL = rgb("#3ddc84")
C_CROSS = rgb("#ff6ec7")
C_FP = rgb("#ff5a4e")
C_AMBER = rgb("#ffb347")

CLASS_RGB = np.array([
    [0.15, 0.22, 0.26],  # 0: silent
    C_CUED,              # 1: cued
    C_RECALL,            # 2: recalled via learned W
    C_CROSS,             # 3: cross-talk (another memory)
    C_FP                 # 4: false positive
])

CMAP_REC = LinearSegmentedColormap.from_list("rec", ["#0e1a20", "#1f6f8b", "#3ddc84", "#ffd27a"])
CMAP_W = LinearSegmentedColormap.from_list("w", ["#0c151a", "#1b4965", "#5fb7ff", "#ffd27a", "#ff5a4e"])


def classify_retrieval(R, cue):
    """Classify each neuron's response during retrieval of `cue` into classes 0..4, plus missed engram cells."""
    fired = R.fired[cue]
    driven = R.driven[cue]
    eng_c = R.E.eng[cue]
    union = R.E.eng.any(0)

    cls = np.zeros(M, int)
    cls[fired & ~union & ~driven] = 4          # false positive
    cls[fired & union & ~eng_c & ~driven] = 3  # cross-talk
    cls[fired & eng_c & ~driven] = 2           # recalled
    cls[fired & driven] = 1                    # cued
    missed = eng_c & ~fired
    return cls, missed


def filament_glow(V):
    """Convert a voltage array [T, M] into a smooth bulb-glow trace [T, M] in [0, 1]."""
    raw = np.clip((V + 0.4) / 2.2, 0.0, 1.0).astype(np.float32)
    out = np.empty_like(raw)
    g = np.zeros(raw.shape[1], np.float32)
    dec = np.float32(np.exp(-dt / 3.5))
    for k in range(len(raw)):
        g = np.maximum(raw[k], g * dec)
        out[k] = g
    return out


class BlitManager:
    """Fast animation helper using canvas background blitting."""

    def __init__(self, canvas, artists=()):
        self.canvas = canvas
        self.artists = list(artists)
        self.active = False
        self._bg = None

    def activate(self):
        if not self.active:
            self.active = True
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def deactivate(self):
        if self.active:
            self.active = False
            self._bg = None
            self.canvas.draw_idle()

    def update(self):
        if not self.active or self._bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._bg)
        for a in self.artists:
            if a is not None and a.get_visible():
                try:
                    a.axes.draw_artist(a)
                except Exception:
                    pass
        self.canvas.blit(self.canvas.figure.bbox)
        self.canvas.flush_events()


class BulbField:
    """Draws the GxG grid of neurons as hanging light bulbs in atmospheric fog."""

    def __init__(self, ax, silhouettes=False):
        self.ax = ax
        ax.set_facecolor("#071014")
        for sp in ax.spines.values():
            sp.set_color("#2a3a43")
        ax.set_xticks([]); ax.set_yticks([])

        # Perspective projection of the GxG grid
        r = xy[:, 0] / max(G - 1, 1)   # 0 (back) .. 1 (front)
        c = xy[:, 1] / max(G - 1, 1)   # 0 (left) .. 1 (right)
        depth = 0.52 + 0.48 * r
        self.bx = (c - 0.5) * depth * 1.75
        self.by = 0.18 - 0.58 * r + 0.04 * np.cos((c - 0.5) * np.pi)
        self.scale = depth

        # Atmospheric background gradient
        grad = np.linspace(0, 1, 64)[:, None]
        bg_rgb = np.zeros((64, 64, 3))
        bg_rgb[..., 0] = 0.03 + 0.04 * grad
        bg_rgb[..., 1] = 0.07 + 0.08 * grad
        bg_rgb[..., 2] = 0.09 + 0.10 * grad
        ax.imshow(bg_rgb, extent=[-1.05, 1.05, -0.58, 0.42], aspect="auto", zorder=0, origin="lower")

        # Hanging cords
        cords = [np.column_stack([[x, x], [y + 0.025 * s, 0.42]])
                 for x, y, s in zip(self.bx, self.by, self.scale)]
        ax.add_collection(LineCollection(cords, colors="#182830", linewidths=0.6, zorder=1))

        # Two figures in the fog
        self.sil = []
        if silhouettes:
            for fx, fy, sc in [(-0.14, -0.46, 0.95), (0.12, -0.44, 0.88)]:
                body = Polygon([
                    (fx - 0.045 * sc, fy),
                    (fx + 0.045 * sc, fy),
                    (fx + 0.055 * sc, fy + 0.18 * sc),
                    (fx + 0.025 * sc, fy + 0.25 * sc),
                    (fx - 0.025 * sc, fy + 0.25 * sc),
                    (fx - 0.055 * sc, fy + 0.18 * sc),
                ], closed=True, fc="#050b0e", ec="#16262e", lw=0.7, alpha=0.88, zorder=2)
                head = Circle((fx, fy + 0.29 * sc), 0.032 * sc, fc="#050b0e", ec="#16262e", lw=0.7, alpha=0.88, zorder=2)
                ax.add_patch(body); ax.add_patch(head)
                self.sil.extend([body, head])

        # Bulb scatter layers: outer halo, glass body, inner filament
        s_halo = (260 * self.scale) ** 2 / 110.0
        s_body = (115 * self.scale) ** 2 / 110.0
        s_core = (52 * self.scale) ** 2 / 110.0

        self.sc_halo = ax.scatter(self.bx, self.by, s=s_halo, c=np.zeros((M, 4)), edgecolors="none", zorder=3)
        self.sc_body = ax.scatter(self.bx, self.by, s=s_body, c=np.zeros((M, 4)), edgecolors="#2a3e48", linewidths=0.6, zorder=4)
        self.sc_core = ax.scatter(self.bx, self.by, s=s_core, c=np.zeros((M, 4)), edgecolors="none", zorder=5)

        # Selection ring
        self.sel_ring, = ax.plot([self.bx[0]], [self.by[0]], "o", ms=14, mfc="none", mec=ACCENT, mew=1.8, zorder=6)

        self.title = ax.text(0.015, 0.965, "", transform=ax.transAxes, va="top", ha="left",
                             fontsize=9.0, color=FG, fontweight="bold", zorder=8)
        self.tlabel = ax.text(0.985, 0.965, "", transform=ax.transAxes, va="top", ha="right",
                              fontsize=9.0, color=ACCENT, family="monospace", zorder=8)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.54, 0.36)
        self._leg = None

    def set_state(self, glow, col, faint=None):
        glow = np.clip(np.asarray(glow, float), 0.0, 1.0)
        col = np.asarray(col, float)
        if col.ndim == 1:
            col = np.tile(col, (M, 1))
        faint = np.zeros(M) if faint is None else np.asarray(faint, float)

        # Halo RGBA
        halo = np.zeros((M, 4))
        halo[:, :3] = col
        halo[:, 3] = 0.42 * (glow ** 1.4)
        self.sc_halo.set_facecolors(halo)

        # Bulb glass RGBA
        body = np.zeros((M, 4))
        base_glass = np.array([0.08, 0.14, 0.17])
        lit = glow[:, None]
        body[:, :3] = (1 - lit) * base_glass + lit * (0.75 * col + 0.25)
        miss = (glow < 0.05) & (faint > 0.5)
        body[miss, :3] = 0.45 * C_AMBER + 0.55 * base_glass
        body[:, 3] = 0.88 + 0.12 * glow
        self.sc_body.set_facecolors(body)

        # Inner filament core RGBA
        core = np.zeros((M, 4))
        core[:, :3] = np.clip(0.45 * col + 0.55, 0, 1)
        core[:, 3] = np.clip(glow * 1.15, 0, 1)
        core[miss, :3] = C_AMBER
        core[miss, 3] = 0.70
        self.sc_core.set_facecolors(core)

    def select(self, i):
        if i is None:
            self.sel_ring.set_visible(False)
        else:
            self.sel_ring.set_visible(True)
            self.sel_ring.set_data([self.bx[i]], [self.by[i]])

    def pick(self, x, y):
        d2 = (self.bx - x) ** 2 + (self.by - y) ** 2
        i = int(np.argmin(d2))
        return i if d2[i] < 0.015 else None

    def set_legend(self, items):
        if self._leg is not None:
            self._leg.remove()
            self._leg = None
        if not items:
            return
        handles = [Line2D([0], [0], marker="o", color="none", label=lab,
                          markerfacecolor=c, markeredgecolor="none", markersize=6.5)
                   for lab, c in items]
        self._leg = self.ax.legend(handles=handles, loc="lower right", fontsize=7.0,
                                   frameon=True, facecolor="#081014", edgecolor="#2a3a43",
                                   framealpha=0.85, ncol=min(len(items), 3))
        for t in self._leg.get_texts():
            t.set_color(FG)
        self._leg.set_zorder(9)

    def artists(self):
        return [self.sc_halo, self.sc_body, self.sc_core, self.sel_ring, self.tlabel]


# ============================================================================= interactive app
# column, key, label, lo, hi, step, format
SLIDERS = [
    (0, "n_mem", "Memories stored", 1, 8, 1, "%d"),
    (0, "t_enc", "First encoding hour", 0, 24, 0.5, "%.1f"),
    (0, "spacing", "Hours between memories", 0, 12, 0.25, "%.2f"),
    (0, "sim", "Pattern similarity", 0, 1, 0.05, "%.2f"),
    (0, "kc", "CREB gain kc", 0, 0.30, 0.005, "%.3f"),
    (0, "s0", "Stimulus strength s0", 0.10, 0.50, 0.01, "%.2f"),
    (0, "enc_rec", "Recurrence while encoding", 0, 1, 0.05, "%.2f"),
    (1, "swr_off", "SWR onset vs stimulus", -150, 250, 5, "%d"),
    (1, "swr_w", "SWR width", 10, 200, 5, "%d"),
    (1, "eta", "STDP LTP rate", 0.01, 0.40, 0.01, "%.2f"),
    (1, "ratio", "LTD / LTP ratio", 0, 1.5, 0.05, "%.2f"),
    (1, "wmax", "Synapse cap w_max", 0.05, 1.0, 0.01, "%.2f"),
    (1, "budget", "Incoming budget (sum W)", 0.2, 4.0, 0.05, "%.2f"),
    (1, "tau_s", "Synaptic decay tau", 2, 20, 0.5, "%.1f"),
    (1, "g_inh", "Global inhibition", 0, 5, 0.1, "%.1f"),
    (1, "noise", "Membrane noise", 0, 0.30, 0.005, "%.3f"),
    (2, "cue_mem", "Cue memory", 1, 8, 1, "%d"),
    (2, "ret_dt", "Retrieval delay (h)", 0, 24, 0.5, "%.1f"),
    (2, "cue_f", "Cue fraction", 0.05, 1.0, 0.05, "%.2f"),
    (2, "cue_gain", "Cue strength", 0.3, 1.6, 0.05, "%.2f"),
    (2, "cue_noise", "Cue distractors", 0, 0.5, 0.02, "%.2f"),
    (2, "s_on", "Presyn. activity (phase plane)", 0.1, 1.0, 0.05, "%.2f"),
    (2, "speed", "Movie speed", 0.25, 4.0, 0.25, "%.2f"),
]
COLUMN_TITLES = ["WORLD & ENCODING", "PLASTICITY & NETWORK", "RETRIEVAL & VIEW"]
INT_KEYS = ("n_mem", "cue_mem")
MODES = [("result", "Retrieval result"), ("retrieval", "Retrieval movie"), ("encoding", "Encoding movie"),
         ("engrams", "Engram map"), ("creb", "CREB state")]
MOVIE_MODES = ("retrieval", "encoding")


def style_axes(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#2a3a43")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


class App:
    def __init__(self, seed=7, P=None, mode="result"):
        self.seed = seed
        self.world = World(seed)
        self.P = clean(P) if P is not None else Params()
        self.flags = dict(auto=True, gate=True, naive=False, men=True)
        self.mode = mode
        self.sel, self.sel_user = None, False
        self.res = None
        self.muted = False
        self.stale = False
        self.playing = False
        self.t_view = T_TOTAL
        self._time_guard = False
        self._mode_cache = {}
        self.cursors = []
        self.sliders = {}
        self.dur = 0.0
        self.build()
        self.recompute()

    # ------------------------------------------------------------------ helpers
    @property
    def cue(self):
        return int(np.clip(self.P.cue_mem, 1, self.res.n if self.res else self.P.n_mem)) - 1

    def status(self, msg):
        self.status_txt.set_text(msg)

    # ------------------------------------------------------------------ UI construction
    def build(self):
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": MUTED,
                             "axes.edgecolor": "#2a3a43", "xtick.color": MUTED, "ytick.color": MUTED,
                             "savefig.facecolor": BG})
        self.fig = plt.figure(figsize=(18, 11.4), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("FitzHugh-Nagumo engram lab")
        except Exception:
            pass
        gs = self.fig.add_gridspec(2, 6, left=0.04, right=0.985, top=0.925, bottom=0.425, hspace=0.36,
                                   wspace=0.50, height_ratios=[1.05, 1])
        self.ax_scene = self.fig.add_subplot(gs[0, 0:3])
        self.ax_phase = self.fig.add_subplot(gs[1, 0:3])
        self.ax_rec = self.fig.add_subplot(gs[0, 3])
        self.ax_W = self.fig.add_subplot(gs[0, 4])
        self.ax_txt = self.fig.add_subplot(gs[0, 5])
        sub = gs[1, 3:6].subgridspec(3, 1, hspace=0.75)
        self.ax_t = [self.fig.add_subplot(sub[i]) for i in range(3)]
        self.fig.suptitle("FitzHugh-Nagumo engram lab:  CREB-gated allocation · SWR-gated STDP · "
                          "multi-memory recall      (click a bulb, or a row of the recall matrix)",
                          fontsize=11, color=FG, y=0.975)
        self.field = BulbField(self.ax_scene, silhouettes=True)
        self.bm = BlitManager(self.fig.canvas, [])

        # --- scrub bar and play button laid over the scene
        pos = self.ax_scene.get_position()
        ax_play = self.fig.add_axes([pos.x0 + 0.006, pos.y0 + 0.008, 0.028, 0.026], zorder=10)
        self.b_play = Button(ax_play, "▶", color="#0c151a", hovercolor="#1f3a46")
        self.b_play.label.set_color(ACCENT); self.b_play.label.set_fontsize(10)
        self.b_play.on_clicked(lambda e: self.toggle_play())
        ax_ts = self.fig.add_axes([pos.x0 + 0.045, pos.y0 + 0.016, pos.width - 0.058, 0.010], zorder=10)
        ax_ts.set_facecolor("#0c151a")
        self.s_time = Slider(ax_ts, "", 0, T_TOTAL, valinit=T_TOTAL, color="#2b7f99", initcolor="none")
        self.s_time.valtext.set_visible(False)
        self.s_time.drawon = False
        self.s_time.on_changed(self.on_time)

        # --- parameter sliders
        for col, title in enumerate(COLUMN_TITLES):
            self.fig.text((0.118, 0.383, 0.648)[col] - 0.105, 0.385, title, fontsize=8.5, color=ACCENT,
                          fontweight="bold")
        rows = [0, 0, 0]
        for col, key, label, lo, hi, step, fmt in SLIDERS:
            self.add_slider(col, rows[col], key, label, lo, hi, step, fmt)
            rows[col] += 1
        self.sync_cue_slider()

        # --- buttons, radio, checks
        def btn(rect, text, cb):
            axb = self.fig.add_axes(rect)
            bt = Button(axb, text, color=PANEL2, hovercolor="#26414d")
            bt.label.set_fontsize(8.5); bt.label.set_color(FG)
            bt.on_clicked(cb)
            return bt
        x0, x1, w, h = 0.815, 0.905, 0.085, 0.032
        self.b_run = btn([x0, 0.352, w, h], "Run", lambda e: self.recompute())
        self.b_cue = btn([x1, 0.352, w, h], "New cue mask", self.new_mask)
        self.b_world = btn([x0, 0.312, w, h], "New world", self.new_world)
        self.b_reset = btn([x1, 0.312, w, h], "Reset sliders", self.reset)
        self.b_png = btn([x0, 0.272, w, h], "Save PNG", self.on_save_png)
        self.b_gif = btn([x1, 0.272, w, h], "Export GIF", self.on_save_gif)

        axr = self.fig.add_axes([0.815, 0.140, 0.17, 0.118], facecolor=PANEL)
        axr.set_title("Bulb field shows", fontsize=8, color=MUTED, loc="left")
        for sp in axr.spines.values():
            sp.set_color("#2a3a43")
        labels = [m[1] for m in MODES]
        try:
            self.radio = RadioButtons(axr, labels, active=[m[0] for m in MODES].index(self.mode), activecolor=ACCENT,
                                      label_props={"color": [FG] * len(labels), "fontsize": [8.5] * len(labels)})
        except TypeError:
            self.radio = RadioButtons(axr, labels, active=[m[0] for m in MODES].index(self.mode), activecolor=ACCENT)
            for t in self.radio.labels:
                t.set_color(FG); t.set_fontsize(8.5)
        self.radio.on_clicked(self.on_mode)

        axc = self.fig.add_axes([0.815, 0.040, 0.17, 0.092], facecolor=PANEL)
        for sp in axc.spines.values():
            sp.set_color("#2a3a43")
        cl = ["auto-run on change", "CREB gates plasticity", "retrieve from naive net (control)", "two figures in the fog"]
        try:
            self.chk = CheckButtons(axc, cl, [True, True, False, True],
                                    label_props={"color": [FG] * 4, "fontsize": [8.5] * 4},
                                    frame_props={"edgecolor": [MUTED] * 4, "facecolor": ["none"] * 4},
                                    check_props={"color": [ACCENT] * 4})
        except TypeError:
            self.chk = CheckButtons(axc, cl, [True, True, False, True])
            for t in self.chk.labels:
                t.set_color(FG); t.set_fontsize(8.5)
        self.chk.on_clicked(self.on_flag)
        self.status_txt = self.fig.text(0.04, 0.012, "", fontsize=8.5, color=MUTED, family="monospace")
        self.timer = self.fig.canvas.new_timer(interval=250)
        self.timer.single_shot = True
        self.timer.add_callback(self.recompute)
        self.play_timer = self.fig.canvas.new_timer(interval=40)
        self.play_timer.add_callback(self.tick)
        self.idle_timer = self.fig.canvas.new_timer(interval=700)
        self.idle_timer.single_shot = True
        self.idle_timer.add_callback(self.bm.deactivate)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.text(0.04, 0.034, "keys:  space = play/pause    1-8 = choose cue memory", fontsize=8, color=MUTED)

    def add_slider(self, col, row, key, label, lo, hi, step, fmt):
        x0 = (0.118, 0.383, 0.648)[col]
        y = 0.350 - row * 0.0375
        ax = self.fig.add_axes([x0, y, 0.115, 0.015], facecolor=PANEL2)
        kw = dict(valinit=getattr(self.P, key), valstep=step, valfmt=fmt, color="#2b7f99")
        try:
            s = Slider(ax, label, lo, hi, initcolor="none", track_color=PANEL2, **kw)
        except TypeError:
            s = Slider(ax, label, lo, hi, **kw)
        s.label.set_fontsize(8); s.label.set_color(FG)
        s.valtext.set_fontsize(8); s.valtext.set_color(ACCENT)
        s.on_changed(lambda v, k=key: self.on_param(k, v))
        self.sliders[key] = s

    def sync_cue_slider(self):
        """The cue-memory slider can only reach as far as the number of stored memories."""
        s = self.sliders["cue_mem"]
        n = int(self.P.n_mem)
        s.valmax = n
        s.ax.set_xlim(s.valmin, max(n, s.valmin + 1))
        if self.P.cue_mem > n:
            self.P.cue_mem = n
            self.muted = True
            s.set_val(n)
            self.muted = False

    # ------------------------------------------------------------------ callbacks
    def on_param(self, key, v):
        val = int(round(v)) if key in INT_KEYS else float(v)
        setattr(self.P, key, val)
        if self.muted:
            return
        if key == "n_mem":
            self.sync_cue_slider()
        if key == "cue_mem":
            self.refresh_cue()
            return
        if key == "s_on":
            self.draw_phase(); self.fig.canvas.draw_idle()
            return
        if key == "speed":
            return
        if self.flags["auto"]:
            self.timer.stop(); self.timer.start()
        else:
            self.stale = True
            self.draw_text(); self.fig.canvas.draw_idle()

    def on_flag(self, label):
        st = self.chk.get_status()
        old = dict(self.flags)
        self.flags.update(auto=st[0], gate=st[1], naive=st[2], men=st[3])
        for p in getattr(self.field, "sil", []):
            p.set_visible(self.flags["men"])
        if (old["gate"], old["naive"]) != (self.flags["gate"], self.flags["naive"]):
            self.recompute()
        elif self.flags["auto"] and not old["auto"] and self.stale:
            self.recompute()
        else:
            self.fig.canvas.draw_idle()

    def on_mode(self, label):
        self.mode = [k for k, v in MODES if v == label][0]
        if self.mode in MOVIE_MODES and self.t_view >= T_TOTAL - 1:
            self.set_time(0.0)
        self.pause()
        self.draw_scene(); self.draw_traces()
        self.fig.canvas.draw_idle()

    def new_mask(self, _=None):
        self.world.new_cues()
        self.recompute()

    def new_world(self, _=None):
        self.seed += 1
        self.world = World(self.seed)
        self.sel, self.sel_user = None, False
        self.recompute()

    def reset(self, _=None):
        self.muted = True
        for k, s in self.sliders.items():
            s.set_val(getattr(DEFAULTS, k))
        self.muted = False
        self.P = replace(DEFAULTS)
        self.sync_cue_slider()
        self.recompute()

    def on_click(self, ev):
        if ev.inaxes is self.ax_scene and ev.xdata is not None:
            i = self.field.pick(ev.xdata, ev.ydata)
            if i is not None:
                self.select(i)
        elif ev.inaxes is self.ax_rec and ev.ydata is not None and self.res is not None:
            r = int(np.clip(round(ev.ydata), 0, self.res.n - 1))
            self.sliders["cue_mem"].set_val(r + 1)

    def on_key(self, ev):
        if ev.key == " ":
            self.toggle_play()
        elif ev.key in [str(i) for i in range(1, 9)] and self.res is not None:
            self.sliders["cue_mem"].set_val(min(int(ev.key), self.res.n))

    def select(self, i):
        self.pause()
        self.sel, self.sel_user = int(i), True
        self.field.select(self.sel)
        self.draw_phase(); self.draw_traces(); self.draw_text()
        self.fig.canvas.draw_idle()

    def on_save_png(self, _=None):
        path = time.strftime("fhn_memory_%Y%m%d_%H%M%S.png")
        self.save_png(path)
        self.status(f"saved {os.path.abspath(path)}")
        self.fig.canvas.draw_idle()

    def on_save_gif(self, _=None):
        path = time.strftime("fhn_memory_%Y%m%d_%H%M%S.gif")
        self.status("rendering GIF ... (a few seconds)")
        self.fig.canvas.draw()
        try:
            export_gif(self.res, self.cue, path)
            self.status(f"saved {os.path.abspath(path)}")
        except Exception as e:  # pragma: no cover
            self.status(f"GIF export failed: {e}")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ playback
    def pause(self):
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            self.b_play.label.set_text("▶")
        self.idle_timer.stop()
        self.bm.deactivate()

    def toggle_play(self):
        if self.playing:
            self.pause()
            self.fig.canvas.draw_idle()
            return
        if self.mode not in MOVIE_MODES:
            self.radio.set_active(1)                     # -> retrieval movie
        if self.t_view >= T_TOTAL - 1:
            self.set_time(0.0)
        self.playing = True
        self.b_play.label.set_text("❚❚")
        self.bm.activate()
        self.play_timer.start()

    def tick(self):
        t = self.t_view + 2.0 * self.P.speed
        if t >= T_TOTAL:
            t = T_TOTAL
            self.set_time(t)
            self.pause()
            self.fig.canvas.draw_idle()
            return
        self.set_time(t)

    def set_time(self, t):
        """Move the time cursor (and the scrub bar) without re-entering on_time."""
        self._time_guard = True
        try:
            self.s_time.set_val(float(t))
        finally:
            self._time_guard = False
        self.apply_time(float(t))
        if self.bm.active:
            self.bm.update()

    def on_time(self, v):
        if self._time_guard:
            return
        if self.mode not in MOVIE_MODES:
            self.radio.set_active(1)
        self.bm.activate()
        self.idle_timer.stop(); self.idle_timer.start()
        self.apply_time(float(v))
        self.bm.update()

    # ------------------------------------------------------------------ compute
    def recompute(self, *_):
        self.pause()
        self.timer.stop()
        t0 = time.time()
        self.stale = False
        try:
            self.res = run_experiment(self.world, self.P, naive=self.flags["naive"], gate=self.flags["gate"])
        except Exception as e:  # keep the window alive whatever the sliders do
            traceback.print_exc()
            self.status(f"error: {e}")
            self.fig.canvas.draw_idle()
            return
        self._mode_cache.clear()
        self.dur = time.time() - t0
        self.sync_cue_slider()
        if self.sel is None or not self.sel_user:
            self.sel = self.auto_pick()
        self.draw_all()

    def auto_pick(self):
        R, c = self.res, self.cue
        cls, _ = classify_retrieval(R, c)
        pool = np.flatnonzero(cls == 2)
        if not len(pool):
            pool = np.flatnonzero(cls == 1)
        if not len(pool):
            pool = np.flatnonzero(R.E.eng[c])
        if not len(pool):
            pool = np.arange(M)
        order = pool[np.argsort(R.peak[c][pool])]
        return int(order[len(order) // 2])

    # ------------------------------------------------------------------ scene data
    def mode_data(self):
        key = (self.mode, self.cue)
        d = self._mode_cache.get(key)
        if d is not None:
            return d
        R, c, E = self.res, self.cue, self.res.E
        L = LETTERS[c]
        if self.mode in ("result", "retrieval"):
            cls, missed = classify_retrieval(R, c)
            leg = [("cue input", C_CUED), ("recalled via learned W", C_RECALL), ("cross-talk (another memory)", C_CROSS),
                   ("false positive", C_FP), ("engram cell that stayed dark", C_AMBER)]
            src = "naive net, no learning" if R.naive else "trained net"
            ncue = int(R.driven[c].sum())
            if self.mode == "result":
                glow = np.where(cls > 0, 0.62 + 0.38 * np.minimum(R.cnt[c] / 5, 1), 0.0)
                d = dict(static=True, glow=glow, col=CLASS_RGB[cls], faint=missed.astype(float), legend=leg,
                         title=f"Retrieval result · cue = memory {L} ({ncue} cue lines) · {src}")
            else:
                d = dict(static=False, frames=filament_glow(R.V[:, c, :]), col=CLASS_RGB[cls], faint=missed.astype(float),
                         legend=leg, title=f"Retrieval movie · cue = memory {L} · {src}")
        elif self.mode == "encoding":
            d = dict(static=False, frames=filament_glow(E.V[c]), col=np.tile(rgb(MEM_COLORS[c]), (M, 1)), faint=None,
                     legend=[(f"memory {L}: cells that fire during its encoding episode", rgb(MEM_COLORS[c]))],
                     title=f"Encoding movie · memory {L} at {E.hours[c]:.1f} h")
        elif self.mode == "engrams":
            cols = np.array([rgb(MEM_COLORS[m]) for m in range(E.n)])
            cnt = E.eng.sum(0)
            col = np.where(cnt[:, None] > 0, (E.eng.T.astype(float) @ cols) / np.maximum(cnt, 1)[:, None], 0.3)
            d = dict(static=True, glow=np.where(cnt > 0, np.minimum(0.62 + 0.14 * (cnt - 1), 1.0), 0.0), col=col,
                     faint=None, legend=[(f"memory {LETTERS[m]} ({int(E.eng[m].sum())} cells)", cols[m]) for m in range(E.n)],
                     title="Engram map · one colour per memory, blends = cells shared between memories")
        else:  # creb
            d = dict(static=True, glow=E.c[c], col=np.tile(C_CUED, (M, 1)), faint=E.eng[c].astype(float),
                     legend=[("CREB level (bright = excitable and plastic)", C_CUED), ("amber filament = engram", C_AMBER)],
                     title=f"CREB level when memory {L} was encoded ({E.hours[c]:.1f} h)")
        self._mode_cache[key] = d
        return d

    def apply_time(self, t):
        self.t_view = t
        d = self.mode_data()
        if d["static"]:
            glow = d["glow"]
            self.field.tlabel.set_text("")
        else:
            fr = d["frames"]
            glow = fr[min(int(t / dt), len(fr) - 1)]
            self.field.tlabel.set_text(f"t = {t:6.1f}")
        self.field.set_state(glow, d["col"], d["faint"])
        for ln in self.cursors:
            ln.set_xdata([t, t])

    def draw_scene(self):
        d = self.mode_data()
        self.field.set_legend(d["legend"])
        self.field.title.set_text(d["title"])
        self.field.select(self.sel)
        self.apply_time(self.t_view)

    # ------------------------------------------------------------------ drawing
    def refresh_cue(self):
        self.pause()
        if not self.sel_user:
            self.sel = self.auto_pick()
        self.draw_scene(); self.draw_recall(); self.draw_phase(); self.draw_traces(); self.draw_text()
        self.fig.canvas.draw_idle()

    def draw_all(self):
        self.draw_scene(); self.draw_recall(); self.draw_W(); self.draw_text(); self.draw_phase(); self.draw_traces()
        self.status(f"last run {self.dur:.2f}s · seed {self.seed} · {self.res.n} memories · "
                    f"grid {G}x{G} · v{__version__}")
        self.fig.canvas.draw_idle()

    def draw_recall(self):
        R, ax = self.res, self.ax_rec
        ax.clear(); style_axes(ax)
        n = R.n
        cm = CMAP_REC.copy(); cm.set_bad("#0c1317")
        ax.imshow(np.ma.masked_invalid(R.rec), cmap=cm, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                v = R.rec[i, j]
                txt = "–" if not np.isfinite(v) else f"{100 * v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9 if n <= 5 else 7,
                        color="#04161c" if np.isfinite(v) and v > 0.55 else FG, fontweight="bold" if i == j else "normal")
        ax.add_patch(Rectangle((-0.5, self.cue - 0.5), n, 1, fill=False, ec=ACCENT, lw=2.0))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([LETTERS[k] for k in range(n)]); ax.set_yticklabels([LETTERS[k] for k in range(n)])
        for k, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
            tx.set_color(MEM_COLORS[k]); ty.set_color(MEM_COLORS[k])
        ax.tick_params(labelsize=9, length=0)
        ax.set_xlabel("memory read out (cells exclusive to it)", fontsize=7.5)
        ax.set_ylabel("memory cued", fontsize=8)
        r = "n/a" if not np.isfinite(R.recall) else f"{100 * R.recall:.0f}%"
        x = "n/a" if not np.isfinite(R.cross) else f"{100 * R.cross:.0f}%"
        ax.set_title(f"Recall (diagonal) vs cross-talk, % cells firing\nmean recall {r} · mean cross-talk {x}",
                     fontsize=8.5, color=FG)

    def draw_W(self):
        R, P, ax = self.res, self.P, self.ax_W
        ax.clear(); style_axes(ax)
        n, eng = R.n, R.E.eng
        grp = np.full(M, n)
        for m in reversed(range(n)):
            grp[eng[m]] = m
        order = np.lexsort((np.arange(M), grp))
        ax.imshow(R.Wr[np.ix_(order, order)], cmap=CMAP_W, vmin=0, vmax=max(P.wmax, 1e-3), interpolation="nearest")
        start = 0
        for m in range(n + 1):
            cnt = int((grp == m).sum())
            if cnt:
                col = MEM_COLORS[m] if m < n else "#46545b"
                ax.plot([start - 0.5, start + cnt - 0.5], [-2.2, -2.2], color=col, lw=3, clip_on=False, solid_capstyle="butt")
                ax.plot([-2.2, -2.2], [start - 0.5, start + cnt - 0.5], color=col, lw=3, clip_on=False, solid_capstyle="butt")
                start += cnt
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("presynaptic j", fontsize=8); ax.set_ylabel("postsynaptic i", fontsize=8)
        over = R.rowsum_max > P.budget + 1e-6 and not R.naive
        ax.set_title(f"Weights W, grouped by memory (colour scale 0–{P.wmax:.2f})\n"
                     f"max {R.W_max:.2f} · row-sum max {R.rowsum_max:.2f} / budget {P.budget:.2f}",
                     fontsize=8.5, color=WARN if over else FG)

    def draw_text(self):
        R, P, ax = self.res, self.P, self.ax_txt
        ax.clear(); ax.axis("off"); ax.set_facecolor(BG)
        pc = lambda x: "  n/a" if not np.isfinite(x) else f"{100 * x:4.0f}%"
        E, n, c = R.E, R.n, self.cue
        L = LETTERS[c]
        sizes = "  ".join(f"{LETTERS[m]} {int(E.eng[m].sum()):<3d}" for m in range(n))
        if n > 4:
            sizes = "  ".join(f"{LETTERS[m]} {int(E.eng[m].sum()):<3d}" for m in range(4)) + "\n           " + \
                    "  ".join(f"{LETTERS[m]} {int(E.eng[m].sum()):<3d}" for m in range(4, n))
        lines = ["ENGRAMS   " + sizes,
                 f"overlap   {pc(R.overlap)} (mean pairwise)", "",
                 f"CUE {L}  ({int(R.driven[c].sum())} cue lines)",
                 f" recall      {pc(R.rec[c, c])}", f" cross-talk  {pc(np.nanmean(np.delete(R.rec[c], c)) if n > 1 and np.isfinite(np.delete(R.rec[c], c)).any() else np.nan)}",
                 f" false pos.  {pc(R.fp[c])}", "",
                 f"ALL {n} CUES",
                 f" recall      {pc(R.recall)}", f" cross-talk  {pc(R.cross)}",
                 f" selectivity {R.recall - R.cross:5.2f}" if np.isfinite(R.cross) else " selectivity   n/a", "",
                 "NETWORK",
                 f" max W       {R.W_max:.2f} / {P.wmax:.2f}",
                 f" row-sum max {R.rowsum_max:.2f} / {P.budget:.2f}",
                 f" peak drive  {R.peak_drive:.2f}",
                 f" firing band {I_HOPF:.2f} – {I_HOPF_HI:.2f}"]
        ax.text(0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.4, color=FG)
        warn = []
        if R.naive:
            warn.append("NAIVE CONTROL: no learning")
        if R.blocked.sum():
            warn.append(f"! {int(R.blocked.sum())} engram cells in depolarisation block\n  (drive > {I_HOPF_HI:.2f}): lower budget\n  or raise inhibition")
        if R.spont:
            warn.append(f"! {R.spont} cells above Hopf at rest\n  (spontaneous firing)")
        if not E.eng[c].any():
            warn.append("! empty engram: raise s0 / kc or\n  change the encoding hour")
        if self.stale:
            warn.append("! sliders changed: press Run")
        if warn:
            ax.text(0, 0.30, "\n".join(warn), va="top", color=WARN, fontsize=7.8)
        i = self.sel
        cls, _ = classify_retrieval(R, c)
        name = ["silent", "cued", "recalled", "cross-talk", "false positive"][cls[i]]
        ax.text(0, 0.03, f"neuron {i} (row {i // G}, col {i % G}): {name}\nCREB@retrieval {R.c_ret[i]:.2f}   rest drive {R.I_ret[i]:.3f}",
                va="bottom", fontsize=7.8, color=MUTED)

    def draw_phase(self):
        R, P, ax = self.res, self.P, self.ax_phase
        ax.clear(); style_axes(ax)
        i, c, s_on = self.sel, self.cue, P.s_on
        Ir, Wr = R.I_ret[i], R.Wr
        union = R.E.eng.any(0)
        row = Wr[i].copy(); row[i] = 0
        j = int(np.argmax(row))
        cand = np.flatnonzero(~union)
        cand = cand[(cand != i) & (cand != j)]
        k = int(cand[np.argmin(D[i, cand])]) if len(cand) else int(np.argsort(row)[0])
        I_before, I_after, I_k = Ir + W0[i, j] * s_on, Ir + Wr[i, j] * s_on, Ir + Wr[i, k] * s_on
        win = slice(int(T_PRE / dt), int((T_PRE + T_STIM) / dt))
        It = R.Itot[:, c, i]
        I_mean, I_peak = float(It[win].mean()), float(It.max())
        vv = np.linspace(-2.6, 2.6, 400)
        ax.plot(vv, vv - vv ** 3 / 3 + Ir, "--", color="#6f8791", lw=1.2, label=f"rest drive ({Ir:.3f})")
        items = [(I_before, "#8ec9e6", f"1 partner j={j}, before learning ({I_before:.3f})", 1.5),
                 (I_after, "#2f9bff", f"1 partner j={j}, after learning ({I_after:.3f})", 2.4),
                 (I_k, "#ffa24a", f"non-memory cell k={k} ({I_k:.3f})", 1.5),
                 (I_mean, "#b48cff", f"actual mean drive in this run ({I_mean:.3f})", 2.0),
                 (I_peak, "#ff6ec7", f"actual peak drive ({I_peak:.3f})", 1.5)]
        for Itot, col, lab, lw in items:
            ax.plot(vv, vv - vv ** 3 / 3 + Itot, color=col, lw=lw, label=lab)
            fv, fw = fixed_point(Itot)
            ax.plot(fv, fw, "o", ms=8, mfc=col if is_stable(fv[0]) else PANEL, mec=col, mew=1.8, zorder=6)
        ax.plot(vv, (vv + a) / b, color="#ff5a4e", lw=1.8, label="n-nullcline")
        pts = np.c_[R.V[:, c, i], R.Nn[:, c, i]]
        segs = np.stack([pts[:-1], pts[1:]], 1)
        lc = LineCollection(segs, cmap="YlGnBu", lw=1.1, alpha=0.95, zorder=5)
        lc.set_array(np.linspace(0.25, 1, len(segs)))
        ax.add_collection(lc)
        ax.plot([], [], color="#d9f2ff", lw=1.2, label="true trajectory of this run (light = late)")
        ax.plot(pts[0, 0], pts[0, 1], "s", ms=5, color="#d9f2ff", zorder=7)
        ax.set_xlabel("V", fontsize=9); ax.set_ylabel("n (recovery variable)", fontsize=9)
        ax.set_ylim(-1.2, 2.4); ax.set_xlim(-2.6, 2.6)
        leg = ax.legend(fontsize=6.8, loc="upper left", ncol=2, frameon=True, facecolor=BG, edgecolor="#2a3a43", framealpha=0.85)
        for t in leg.get_texts():
            t.set_color(FG)
        cls, _ = classify_retrieval(R, c)
        name = ["silent", "cued", "recalled via W", "cross-talk", "false positive"][cls[i]]
        ax.set_title(f"Conditional phase space of neuron {i}  ({name};  W_ij={Wr[i, j]:.3f}, W_ik={Wr[i, k]:.3f}, presyn s={s_on:.2f})\n"
                     f"one partner:  before {regime(I_before)} · after {regime(I_after)} · non-memory {regime(I_k)}"
                     f"     this run:  mean drive {regime(I_mean)} · peak drive {regime(I_peak)}\n"
                     f"firing band {I_HOPF:.3f} – {I_HOPF_HI:.3f}   (filled dot = stable fixed point, open = unstable)",
                     fontsize=8.2, color=FG)

    def draw_traces(self):
        R, P = self.res, self.P
        i, c = self.sel, self.cue
        E = R.E
        axs = self.ax_t
        for ax in axs:
            ax.clear(); style_axes(ax)
        tE = np.arange(len(E.V[c])) * dt
        tR = np.arange(R.V.shape[0]) * dt
        axs[0].plot(tE, E.V[c][:, i], color=MEM_COLORS[c], lw=0.9)
        axs[0].axvspan(T_PRE, T_PRE + T_STIM, color="#ffd27a", alpha=0.10, lw=0)
        axs[0].axvspan(E.swr_start[c], E.swr_start[c] + P.swr_w, ymin=0.70, ymax=1, color="#5fb7ff", alpha=0.45, lw=0)
        axs[0].set_title(f"encoding of memory {LETTERS[c]} at {E.hours[c]:.1f} h  (yellow = stimulus, blue bar = SWR gate)",
                         fontsize=7.8, color=FG, loc="left", pad=3)
        axs[1].plot(tR, R.V[:, c, i], color="#7dffb8", lw=0.9)
        axs[1].axvspan(T_PRE, T_PRE + T_STIM, color="#ffd27a", alpha=0.10, lw=0)
        axs[1].set_title(f"retrieval cued by memory {LETTERS[c]}  (yellow = cue window)", fontsize=7.8, color=FG, loc="left", pad=3)
        axs[2].axhspan(I_HOPF, I_HOPF_HI, color="#3ddc84", alpha=0.10, lw=0)
        axs[2].axhline(I_HOPF, color="#3ddc84", lw=0.6, alpha=0.7)
        axs[2].axhline(I_HOPF_HI, color="#3ddc84", lw=0.6, alpha=0.7)
        axs[2].plot(tR, R.Itot[:, c, i], color="#ff6ec7", lw=0.9)
        top = max(I_HOPF_HI * 1.25, float(R.Itot[:, c, i].max()) * 1.05)
        axs[2].set_ylim(min(0.0, float(R.Itot[:, c, i].min())), top)
        axs[2].set_title("total drive onto this neuron vs firing band (green)", fontsize=7.8, color=FG, loc="left", pad=3)
        axs[2].set_xlabel("model time", fontsize=8)
        for k, ax in enumerate(axs):
            ax.set_xlim(0, T_TOTAL)
            ax.set_yticks([]) if k < 2 else None
        self.cursors = [ax.axvline(self.t_view, color="white", lw=0.9, alpha=0.8) for ax in axs]
        self.bm.artists = self.field.artists() + self.cursors
        for nm in ("poly", "_handle"):
            art = getattr(self.s_time, nm, None)
            if art is not None:
                self.bm.artists.append(art)

    # ------------------------------------------------------------------ output
    def save_png(self, path, dpi=110):
        self.pause()
        self.fig.canvas.draw()
        self.fig.savefig(path, dpi=dpi, facecolor=BG)


# ----------------------------------------------------------------------------- GIF export
def export_gif(res, cue, path, stride=14, fps=20, size=(8.6, 4.4), dpi=90, t_start=10.0):
    """Render the retrieval movie of one cue to an animated GIF (bulb field only)."""
    from PIL import Image
    fig = plt.figure(figsize=size, dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    field = BulbField(ax)
    cls, missed = classify_retrieval(res, cue)
    frames_glow = filament_glow(res.V[:, cue, :])
    col = CLASS_RGB[cls]
    field.title.set_text(f"cue = memory {LETTERS[cue]}  ·  {int(res.driven[cue].sum())} cue lines  ·  "
                         f"{'naive net' if res.naive else 'trained net'}")
    field.set_legend([("cue input", C_CUED), ("recalled via learned W", C_RECALL), ("cross-talk", C_CROSS),
                      ("false positive", C_FP), ("engram, stayed dark", C_AMBER)])
    imgs = []
    for k in range(int(t_start / dt), frames_glow.shape[0], stride):
        field.set_state(frames_glow[k], col, missed.astype(float))
        field.tlabel.set_text(f"t = {k * dt:5.1f}")
        fig.canvas.draw()
        im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        imgs.append(im.quantize(colors=96, dither=Image.Dither.FLOYDSTEINBERG))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(imgs)


# ----------------------------------------------------------------------------- self test
def selftest():
    """Sanity checks of the model and the UI (headless).  Returns a process exit code."""
    import tempfile
    results = []

    def check(name, ok, info=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {info}")

    print(f"FHN engram lab v{__version__} self-test (grid {G}x{G})")
    w = World(7)
    P = Params()
    R = run_experiment(w, P)
    check("firing band is ordered", 0.3 < I_HOPF < 0.4 and 1.3 < I_HOPF_HI < 1.5, f"[{I_HOPF:.3f}, {I_HOPF_HI:.3f}]")
    check("no engram cell is depolarisation-blocked at defaults", R.blocked.sum() == 0, f"peak {R.peak_drive:.2f}")
    check("no spontaneous firing at rest", R.spont == 0)
    rn = run_experiment(w, P, naive=True)
    check("trained net recalls far more than naive control", R.recall > rn.recall + 0.3,
          f"{R.recall:.2f} vs {rn.recall:.2f}")
    rec = []
    for sd in range(7, 13):
        r = run_experiment(World(sd), P)
        rec.append((r.recall, r.cross))
    rec = np.array(rec)
    check("recall beats cross-talk on average over 6 worlds", rec[:, 0].mean() > rec[:, 1].mean() + 0.15,
          f"recall {rec[:, 0].mean():.2f}, cross-talk {rec[:, 1].mean():.2f}")
    # weights stay bounded however aggressive plasticity is
    worst = Params(eta=0.4, ratio=0.0, wmax=1.0, swr_w=200, budget=1.5, s0=0.5, kc=0.3, n_mem=6)
    rw = run_experiment(w, worst)
    check("weights stay within [0, w_max]", rw.W.min() >= 0 and rw.W.max() <= worst.wmax + 1e-9, f"max {rw.W.max():.2f}")
    check("incoming weight never exceeds the budget", rw.rowsum_max <= worst.budget + 1e-6, f"{rw.rowsum_max:.3f}")
    check("results are finite under stress", np.isfinite(rw.V).all() and np.isfinite(rw.Itot).all())
    r1 = run_experiment(w, Params(noise=0.05)); r2 = run_experiment(w, Params(noise=0.05))
    check("runs are deterministic (also with noise)", np.array_equal(r1.cnt, r2.cnt))
    before = encoding(w, clean(P), True)
    run_experiment(w, replace(P, cue_f=0.3))
    check("retrieval-only changes reuse the encoding cache", encoding(w, clean(P), True) is before)
    ok = True
    for n in range(1, MAX_MEM + 1):
        try:
            run_experiment(w, replace(P, n_mem=n))
        except Exception as e:
            ok = False; print("   n_mem", n, e)
    check("1..8 memories all run", ok)
    r0 = run_experiment(w, replace(P, cue_f=0.0))
    check("empty cue recalls nothing", not np.isfinite(r0.recall) or r0.recall < 0.05)
    # UI smoke test
    app = App(seed=7, P=P)
    try:
        app.sliders["n_mem"].set_val(4); app.recompute()
        app.sliders["cue_mem"].set_val(2)
        app.select(40)
        for _, label in MODES:
            app.radio.set_active([m[1] for m in MODES].index(label))
        app.set_time(55.0)
        app.chk.set_active(2); app.chk.set_active(2)
        app.new_mask(); app.new_world(); app.reset()
        out = os.path.join(tempfile.gettempdir(), "fhn_selftest.png")
        app.save_png(out, dpi=60)
        check("UI: sliders, modes, clicks, reset and PNG export", os.path.getsize(out) > 20000)
        out_gif = os.path.join(tempfile.gettempdir(), "fhn_selftest.gif")
        nfr = export_gif(app.res, app.cue, out_gif, stride=120, size=(4, 2.2), dpi=60)
        check("GIF export", nfr > 3 and os.path.getsize(out_gif) > 1000, f"{nfr} frames")
    except Exception:
        traceback.print_exc()
        check("UI smoke test", False)
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} checks passed")
    return 0 if n_ok == len(results) else 1


# ----------------------------------------------------------------------------- main
def main():
    if CLI.selftest:
        sys.exit(selftest())
    P = Params()
    if CLI.n_mem:
        P.n_mem = CLI.n_mem
    app = App(seed=CLI.seed, P=P, mode=CLI.mode)
    R = app.res
    print(f"engrams {R.E.eng.sum(1).tolist()} | recall {R.recall:.2f} | cross-talk {R.cross:.2f} | "
          f"false-pos {R.fp_mean:.3f} | peak drive {R.peak_drive:.2f} (band {I_HOPF:.2f}-{I_HOPF_HI:.2f})")
    if CLI.png:
        app.save_png(CLI.png)
        print("wrote", CLI.png)
    if CLI.gif:
        print("wrote", CLI.gif, f"({export_gif(R, app.cue, CLI.gif)} frames)")
    if CLI.png or CLI.gif or CLI.headless:
        return
    plt.show()


if __name__ == "__main__":
    main()