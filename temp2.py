"""
Memory Allocation & Recall  --  presentation edition
====================================================
CREB-allocated, PV+-inhibited, SWR-consolidated FitzHugh-Nagumo engram model.

Flow (buttons or keyboard):
    ENCODE         animated encoding: sensory drive -> SWR replay -> engram forms
    CUE & RECALL   partial cue -> the silent part of the memory lights up
    FULL DEMO      runs both back to back
    RESET / NEW BRAIN

Keys:  SPACE next step | D full demo | R reset | N new brain | F fullscreen
Click any neuron on the cortical sheet to inspect it.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================================
# Theme
# ==========================================================================
BG_TOP, BG_BOT = "#0b1226", "#03050b"
PANEL = "#0d1424"
BORDER = "#22304f"
TEXT = "#e8eefb"
MUTED = "#7d8bab"

C_CUED = "#22d3ee"      # cyan
C_RECALL = "#34d399"    # green
C_ENGRAM = "#fbbf24"    # amber
C_FALSE = "#fb7185"     # rose
C_SILENT = "#18213a"
C_INH = "#a78bfa"       # violet
C_SWR = "#60a5fa"       # blue
C_CA = "#fb923c"        # orange

CMAP_CREB = LinearSegmentedColormap.from_list(
    "creb", ["#1b1740", "#5b21b6", "#db2777", "#fbbf24"])
CMAP_SYN = LinearSegmentedColormap.from_list(
    "syn", ["#070b16", "#1e3a8a", "#06b6d4", "#ecfeff"])


def rgb(c):
    return np.array(to_rgb(c))


def mix(a, b, t):
    """Linear blend; t may be a scalar or a per-neuron vector."""
    t = np.asarray(t, float)
    if t.ndim == 1:
        t = t[:, None]
    return a * (1 - t) + b * t


SIL, ENG, CUE, REC, FAL = (rgb(c) for c in
                           (C_SILENT, C_ENGRAM, C_CUED, C_RECALL, C_FALSE))
WHITE, BORD = np.ones(3), rgb(BORDER)

plt.rcParams.update({
    "figure.facecolor": BG_BOT, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": BORDER, "font.family": "DejaVu Sans", "font.size": 9,
    "keymap.home": [],   # free the 'r' key for reset
})

# ==========================================================================
# Model
# ==========================================================================
GRID = 12
N = GRID * GRID

A, B, EPS = 0.7, 0.8, 0.08           # FitzHugh-Nagumo
DT = 0.1
V_HOPF = -np.sqrt(1 - EPS * B)
I_HOPF = (V_HOPF + A) / B - V_HOPF + V_HOPF ** 3 / 3             # lower Hopf: rest -> spiking
I_HOPF_HI = (-V_HOPF + A) / B + V_HOPF + (-V_HOPF) ** 3 / 3       # upper Hopf: spiking -> depolarization block

FIRE_MIN = 2
T_ENCODE, T_RETRIEVE = 240.0, 150.0
STIM_ON, STIM_OFF, SWR_START = 20.0, 110.0, 110.0
TAU_SYN, TAU_CA, TAU_SWR, TAU_INH = 5.0, 25.0, 15.0, 12.0
SNAP_EVERY = 40                      # weight snapshot interval (steps)

_rng_w = np.random.RandomState(42)
_conn = _rng_w.rand(N, N) < 0.15
np.fill_diagonal(_conn, 0)
W0 = np.where(_conn, 0.008 * _rng_w.rand(N, N), 0.0)

# Tuned by random search + refinement (recall minus false-positive rate, averaged
# over noise seeds and worlds). On unseen worlds: ~86% recall, ~7% false positives
# (the previous defaults gave ~92% recall but ~73% false positives).
DEFAULTS = dict(
    kc_enc=0.10, kc_ret=0.00, s0=0.24, p_swr=0.002, g_swr=0.06, g_inh=5.3,
    eta=0.50, ratio=0.06, wmax=0.38, theta_p=0.6, cue_f=0.40, sigma_v=0.04,
)


def fixed_point(current):
    current = np.atleast_1d(np.asarray(current, float))
    v = np.full_like(current, -1.0)
    for _ in range(40):
        f = v ** 3 / 3 + (1 / B - 1) * v + (A / B - current)
        v = v - f / (v ** 2 + (1 / B - 1))
    return v, (v + A) / B


def is_stable(v):
    """Linear stability of the FHN fixed point at voltage v (scalar or array)."""
    return ((1 - v ** 2) - EPS * B < 0) & (EPS * (1 - B * (1 - v ** 2)) > 0)


def regime(current):
    """Dynamical regime of a neuron driven by a constant net current."""
    if current < I_HOPF:
        return "below Hopf"
    return "spiking range" if current <= I_HOPF_HI else "above upper Hopf"


class World:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.c_i = rng.uniform(0.1, 1.0, N)
        self.p = rng.uniform(0.0, 1.0, N)
        self.u_cue = rng.random(N)


def simulate_epoch(W_init, I_base, I_ext, c_i, learn, P, T_total, rng,
                   snap_every=0):
    W = W_init.copy()
    n_steps = int(T_total / DT)
    v, w = fixed_point(I_base)
    s, x, y = np.zeros(N), np.zeros(N), np.zeros(N)
    I_inh, R = 0.0, 0.0
    counts = np.zeros(N, int)
    dec_s, dec_ca, dec_R = (np.exp(-DT / TAU_SYN), np.exp(-DT / TAU_CA),
                            np.exp(-DT / TAU_SWR))
    noise = rng.normal(0, P["sigma_v"] * np.sqrt(DT), (n_steps, N))

    V_hist = np.empty((n_steps, N), np.float32)
    w_hist = np.empty((n_steps, N), np.float32)     # recovery variable (for phase plots)
    I_hist = np.empty((n_steps, N), np.float32)     # net input current actually felt
    x_hist = np.empty((n_steps, N), np.float32)
    R_hist = np.empty(n_steps, np.float32)
    inh_hist = np.empty(n_steps, np.float32)
    snaps = []

    for k in range(n_steps):
        t = k * DT
        in_stim = STIM_ON <= t < STIM_OFF

        if learn and t > SWR_START and rng.random() < P["p_swr"]:
            R += 1.0
        else:
            R *= dec_R
        I_inh += DT * (-I_inh + P["g_inh"] * s.mean()) / TAU_INH

        I = I_base + W @ s - I_inh
        if in_stim:
            I += I_ext
        if R > 0.01:
            I += P["g_swr"] * R

        vn = np.clip(v + DT * (v - v ** 3 / 3 - w + I) + noise[k], -3.0, 3.0)
        wn = w + DT * EPS * (vn + A - B * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        V_hist[k], x_hist[k], R_hist[k], inh_hist[k] = v, x, R, I_inh
        w_hist[k], I_hist[k] = w, I

        if in_stim:
            counts += spk
        s = s * dec_s + spk
        x = x * dec_ca + spk
        y = y * dec_ca + spk

        if learn and spk.any():
            sp = spk.astype(float)
            xb = np.maximum(x - P["theta_p"], 0.0)
            yb = np.maximum(y - P["theta_p"], 0.0)
            ltp = np.outer(c_i * sp, xb) * (P["wmax"] - W)
            ltd = np.outer(c_i * yb, sp) * W
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            np.fill_diagonal(W, 0.0)
        if snap_every and k % snap_every == 0:
            snaps.append(W.astype(np.float32))

    return dict(counts=counts, W=W, V=V_hist, w=w_hist, I=I_hist, X=x_hist,
                R=R_hist, inh=inh_hist, snaps=snaps)


def analyse(V):
    """Spike times, cumulative in-window counts and a decaying glow trace."""
    prev = np.vstack([np.full((1, V.shape[1]), -9.0, np.float32), V[:-1]])
    spk = (V > 0) & (prev <= 0)
    ts = np.arange(len(V)) * DT
    in_win = (ts >= STIM_ON) & (ts < STIM_OFF)
    cum = np.cumsum(spk & in_win[:, None], axis=0)
    dec = np.exp(-DT / 6.0)
    act = np.empty(spk.shape, np.float32)
    a_ = np.zeros(spk.shape[1])
    for k in range(len(spk)):
        a_ = a_ * dec + spk[k]
        act[k] = a_
    return dict(ts=ts, cum=cum, act=np.clip(act, 0, 1))


_BIF = {}


def bifurcation_diagram():
    """Spike-envelope (v_min/v_max of the limit cycle) versus constant input current."""
    if not _BIF:
        I = np.linspace(-0.5, 3.0, 176)
        v, w = fixed_point(I)
        vv, ww = v + 0.25, w.copy()
        vmin = np.full_like(I, np.inf)
        vmax = np.full_like(I, -np.inf)
        for k in range(9000):
            vn = vv + DT * (vv - vv ** 3 / 3 - ww + I)
            ww = ww + DT * EPS * (vn + A - B * ww)
            vv = vn
            if k >= 6000:
                vmin, vmax = np.minimum(vmin, vv), np.maximum(vmax, vv)
        _BIF.update(I=I, vmin=vmin, vmax=vmax)
    return _BIF


# ==========================================================================
# Application
# ==========================================================================
class App:
    SPEEDS = [0.5, 1.0, 2.0]
    ENC_SECONDS, REC_SECONDS = 7.0, 6.0

    def __init__(self, seed=101):
        self.seed = seed
        self.world = World(seed)
        self.P = dict(DEFAULTS)
        self.rng = np.random.default_rng()
        self.enc = self.ret = None
        self.sel = 0
        self.speed_i = 1
        self.busy = False
        self.cursor = None
        self.ph_view = "phase"                 # "phase" or "bif"
        self.ph_post = self.ph_dot = None      # live trajectory artists
        self._rec_k = 0
        self._timer = self._after_timer = None
        self._ui_stage = None
        self._last_j = -1
        self._order = np.arange(N)
        self._build()
        self.reset()

    # ------------------------------------------------------------------ build
    def _build(self):
        fig = self.fig = plt.figure(figsize=(18, 10), facecolor=BG_BOT)
        try:
            fig.canvas.manager.set_window_title("Memory Allocation & Recall")
        except Exception:
            pass

        bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
        bg.axis("off")
        bg.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto",
                  cmap=LinearSegmentedColormap.from_list("bg", [BG_BOT, BG_TOP]),
                  extent=[0, 1, 0, 1])

        # header
        fig.text(0.03, 0.962, "Memory Allocation & Recall", fontsize=26,
                 fontweight="bold", color=TEXT, va="center")
        fig.text(0.03, 0.926,
                 "CREB-primed allocation   ·   PV$^{+}$ winner-take-all inhibition"
                 "   ·   SWR replay   ·   burst-threshold plasticity",
                 fontsize=10.5, color=MUTED, va="center")
        fig.add_artist(Line2D([0.03, 0.11], [0.907, 0.907], lw=3.5, color=C_CUED,
                              transform=fig.transFigure))

        self.pills = []
        for x, label in [(0.655, "1  ENCODE"), (0.78, "2  CUE"), (0.895, "3  RECALL")]:
            self.pills.append(fig.text(x, 0.943, label, ha="center", va="center",
                                       fontsize=12, fontweight="bold"))
        for x in (0.7175, 0.8375):
            fig.text(x, 0.943, "›", ha="center", va="center", fontsize=18, color=MUTED)
        self._pill_accents = [C_ENGRAM, C_CUED, C_RECALL]

        # hero: cortical sheet
        ax = self.ax_hero = fig.add_axes([0.03, 0.245, 0.33, 0.62])
        self._style(ax)
        ax.set_xlim(-0.7, GRID - 0.3)
        ax.set_ylim(GRID - 0.3, -0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.6)
        idx = np.arange(N)
        xs, ys = idx % GRID, idx // GRID
        self.sc_glow = ax.scatter(xs, ys, s=2700, c=np.zeros((N, 4)), linewidths=0, zorder=2)
        self.sc_node = ax.scatter(xs, ys, s=760, c=np.tile(SIL, (N, 1)),
                                  edgecolors=np.tile(BORD, (N, 1)), linewidths=1.2, zorder=3)
        self.sc_sel = ax.scatter([0], [0], s=1350, facecolors="none",
                                 edgecolors="white", linewidths=2.2, zorder=5)
        handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec=c, ms=9, label=l)
                   for c, l in [(C_SILENT, "Silent"), (C_ENGRAM, "Engram (dormant)"),
                                (C_CUED, "Cued"), (C_RECALL, "Recalled"),
                                (C_FALSE, "False +")]]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.012),
                  ncol=5, frameon=False, fontsize=8.5, labelcolor=TEXT,
                  handletextpad=0.1, columnspacing=1.0)
        self.t_clock = fig.text(0.03, 0.188, "", fontsize=12, family="monospace",
                                color=C_CUED, fontweight="bold")
        self.t_status = fig.text(0.03, 0.162, "", fontsize=9.5, color=MUTED)

        # KPI cards
        self.kpi = []
        for i, (label, col) in enumerate([("ENGRAM SIZE", C_ENGRAM), ("CUED NEURONS", C_CUED),
                                          ("RECALL RATE", C_RECALL), ("FALSE POSITIVES", C_FALSE)]):
            a = fig.add_axes([0.40 + i * 0.1465, 0.785, 0.1305, 0.088])
            self._style(a)
            a.set_xticks([])
            a.set_yticks([])
            a.add_patch(Rectangle((0, 0.9), 1, 0.1, transform=a.transAxes, color=col, lw=0))
            a.text(0.5, 0.64, label, transform=a.transAxes, ha="center", va="center",
                   fontsize=8.5, fontweight="bold", color=MUTED)
            self.kpi.append(a.text(0.5, 0.28, "—", transform=a.transAxes, ha="center",
                                   va="center", fontsize=23, fontweight="bold", color=col))

        # analysis panels
        self.ax_W = fig.add_axes([0.40, 0.52, 0.18, 0.215])
        self.ax_ph = fig.add_axes([0.63, 0.52, 0.16, 0.215])
        self.ax_curve = fig.add_axes([0.83, 0.52, 0.14, 0.215])
        self.ax_tr = fig.add_axes([0.40, 0.245, 0.57, 0.205])
        self.b_view = self._btn([0.745, 0.743, 0.045, 0.021], "⇄ VIEW",
                                "#1a2440", "#26335a", TEXT, 7.5)
        self.b_view.on_clicked(lambda _e: self._toggle_view())

        # controls
        fig.add_artist(Rectangle((0.03, 0.022), 0.94, 0.118, transform=fig.transFigure,
                                 fc=PANEL, ec=BORDER, lw=1.2, alpha=0.75, zorder=-1))
        fig.text(0.045, 0.125, "PARAMETERS", fontsize=8.5, color=MUTED, fontweight="bold")
        fig.text(0.665, 0.125, "CONTROLS", fontsize=8.5, color=MUTED, fontweight="bold")
        specs = [("CREB gain", "kc_enc", 0, 0.4, 0.01, C_ENGRAM),
                 ("PV$^{+}$ inhibition", "g_inh", 0, 6, 0.1, C_INH),
                 ("SWR rate", "p_swr", 0, 0.02, 0.001, C_SWR),
                 ("SWR drive", "g_swr", 0, 0.5, 0.01, C_SWR),
                 ("Plasticity η", "eta", 0, 1, 0.05, C_RECALL),
                 ("Ca$^{2+}$ threshold", "theta_p", 0.5, 2, 0.1, C_CA)]
        self.sliders = {}
        for n, (label, key, lo, hi, st, col) in enumerate(specs):
            r, c = divmod(n, 3)
            self.sliders[key] = self._slider(
                [0.115 + c * 0.19, 0.078 - r * 0.044, 0.095, 0.024], label, key, lo, hi, st, col)

        dark = "#0a0f1c"
        self.b_enc = self._btn([0.665, 0.07, 0.095, 0.05], "▶  ENCODE", C_ENGRAM, "#fcd34d", dark)
        self.b_rec = self._btn([0.77, 0.07, 0.095, 0.05], "✦  CUE & RECALL", C_RECALL, "#6ee7b7", dark)
        self.b_demo = self._btn([0.875, 0.07, 0.09, 0.05], "▶▶  FULL DEMO", C_CUED, "#67e8f9", dark)
        self.b_reset = self._btn([0.665, 0.03, 0.095, 0.034], "↺  RESET", "#1a2440", "#26335a", TEXT, 10)
        self.b_new = self._btn([0.77, 0.03, 0.095, 0.034], "NEW BRAIN", "#1a2440", "#26335a", TEXT, 10)
        self.b_speed = self._btn([0.875, 0.03, 0.09, 0.034], "SPEED  1×", "#1a2440", "#26335a", TEXT, 10)
        self.b_enc.on_clicked(lambda _e: self.start_encode())
        self.b_rec.on_clicked(lambda _e: self.start_recall())
        self.b_demo.on_clicked(lambda _e: self.full_demo())
        self.b_reset.on_clicked(lambda _e: self.reset())
        self.b_new.on_clicked(lambda _e: self.new_brain())
        self.b_speed.on_clicked(lambda _e: self._cycle_speed())

        fig.text(0.5, 0.006, "SPACE next step   ·   D full demo   ·   R reset   ·   N new brain"
                 "   ·   F fullscreen   ·   click a neuron to inspect it",
                 ha="center", fontsize=8.5, color=MUTED)

        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ---------------------------------------------------------- small widgets
    @staticmethod
    def _style(ax):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)

    @staticmethod
    def _title(ax, text, color=TEXT, fs=10):
        ax.set_title(text, loc="left", fontsize=fs, fontweight="bold", color=color, pad=8)

    @staticmethod
    def _placeholder(ax, msg):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=10, style="italic")

    def _btn(self, rect, label, color, hover, fg, fs=11):
        ax = self.fig.add_axes(rect)
        b = Button(ax, label, color=color, hovercolor=hover)
        b.label.set_color(fg)
        b.label.set_fontsize(fs)
        b.label.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        return b

    def _slider(self, rect, label, key, lo, hi, step, color):
        ax = self.fig.add_axes(rect, facecolor="#18223a")
        kw = dict(valinit=self.P[key], valstep=step, color=color)
        try:
            s = Slider(ax, label, lo, hi, initcolor="none",
                       handle_style=dict(facecolor=TEXT, edgecolor=color, size=11), **kw)
        except TypeError:
            s = Slider(ax, label, lo, hi, **kw)
        s.label.set_color(TEXT)
        s.label.set_fontsize(9.5)
        s.valtext.set_color(color)
        s.valtext.set_fontsize(9.5)
        s.valtext.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_visible(False)
        s.on_changed(lambda v, k=key: self._on_param(k, v))
        return s

    def _set_pills(self, states):
        for t, st, col in zip(self.pills, states, self._pill_accents):
            if st == "active":
                t.set_color("#0a0f1c")
                fc, ec = col, col
            elif st == "done":
                t.set_color(col)
                fc, ec = PANEL, col
            else:
                t.set_color(MUTED)
                fc, ec = PANEL, BORDER
            t.set_bbox(dict(boxstyle="round,pad=0.55,rounding_size=0.8", fc=fc, ec=ec, lw=1.6))

    def _set_hero_title(self, text, color=TEXT):
        self.ax_hero.set_title(text, loc="left", fontsize=12, fontweight="bold",
                               color=color, pad=10)

    def _status(self, text, color=MUTED):
        self.t_status.set_text(text)
        self.t_status.set_color(color)

    def _set_kpis(self, eng=None, cued=None, recall=None, fp=None):
        vals = [eng, cued, None if recall is None else f"{recall:.0f}%", fp]
        for t, v in zip(self.kpi, vals):
            t.set_text("—" if v is None else str(v))

    # --------------------------------------------------------------- callbacks
    def _on_param(self, key, value):
        self.P[key] = float(value)
        if self.enc is not None and not self.busy:
            self._status("Parameters changed — press ENCODE to apply", C_ENGRAM)
            self.fig.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes is self.ax_hero and ev.xdata is not None and not self.busy:
            col = int(np.clip(round(ev.xdata), 0, GRID - 1))
            row = int(np.clip(round(ev.ydata), 0, GRID - 1))
            self.sel = row * GRID + col
            self.sc_sel.set_offsets([[col, row]])
            self._draw_inspector()
            self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        k = ev.key
        if k == " ":
            if self.busy:
                return
            if self.enc is None:
                self.start_encode()
            elif self.ret is None:
                self.start_recall()
            else:
                self.reset()
        elif k == "d":
            self.full_demo()
        elif k == "r":
            self.reset()
        elif k == "n":
            self.new_brain()

    def _toggle_view(self):
        self.ph_view = "bif" if self.ph_view == "phase" else "phase"
        animating = self.busy and self._ui_stage in ("cue", "rec")
        self._draw_phase(upto=self._rec_k if animating else None)
        self.fig.canvas.draw_idle()

    def _cycle_speed(self):
        self.speed_i = (self.speed_i + 1) % len(self.SPEEDS)
        sp = self.SPEEDS[self.speed_i]
        self.b_speed.label.set_text(f"SPEED  {sp:g}×")
        self.fig.canvas.draw_idle()

    def new_brain(self):
        self.seed += 1
        self.world = World(self.seed)
        self.reset()

    # ---------------------------------------------------------- animation core
    def _stop_anim(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._after_timer is not None:
            self._after_timer.stop()
            self._after_timer = None
        self.busy = False

    def _after(self, ms, fn):
        t = self.fig.canvas.new_timer(interval=ms)
        t.single_shot = True
        t.add_callback(fn)
        t.start()
        self._after_timer = t

    def _run_anim(self, n_steps, seconds, frame_fn, done_fn):
        self._stop_anim()
        self.busy = True
        t0 = time.perf_counter()
        duration = seconds / self.SPEEDS[self.speed_i]

        def tick():
            frac = (time.perf_counter() - t0) / duration
            if frac >= 1.0:
                self._stop_anim()
                frame_fn(n_steps - 1)
                done_fn()
            else:
                frame_fn(int(frac * n_steps))
            self.fig.canvas.draw_idle()

        self._timer = self.fig.canvas.new_timer(interval=40)
        self._timer.add_callback(tick)
        self._timer.start()

    # -------------------------------------------------------------- simulation
    def reset(self):
        self._stop_anim()
        self.enc = self.ret = None
        self.sel = 0
        self._refresh_static()

    def full_demo(self):
        if self.busy:
            return
        self.reset()
        self.start_encode(then=lambda: self._after(1000, self.start_recall))

    def start_encode(self, then=None):
        if self.busy:
            return
        self._stop_anim()
        self.ret = None
        self._set_pills(["active", "pending", "pending"])
        self._status("Simulating encoding dynamics…", C_ENGRAM)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        w, P = self.world, self.P
        I_ext = P["s0"] * w.p
        res = simulate_epoch(W0, 0.10 + P["kc_enc"] * w.c_i, I_ext, w.c_i, True, P,
                             T_ENCODE, self.rng, snap_every=SNAP_EVERY)
        eng = res["counts"] >= FIRE_MIN
        self.enc = dict(res=res, eng=eng, W=res["W"], I_ext=I_ext, tl=analyse(res["V"]))
        self._order = np.r_[np.flatnonzero(eng), np.flatnonzero(~eng)]
        e_idx = np.flatnonzero(eng)
        self.sel = int(e_idx[0]) if len(e_idx) else 0
        self.sc_sel.set_offsets([[self.sel % GRID, self.sel // GRID]])

        self._setup_W(frame=0)
        self._setup_curve()
        self._draw_inspector(cursor=True)
        self._ui_stage = None
        self._set_kpis(eng=0)
        self._status("")

        def done():
            self._done_encode()
            if then:
                then()

        self._run_anim(len(res["V"]), self.ENC_SECONDS, self._frame_encode, done)

    def _done_encode(self):
        self._refresh_static()

    def start_recall(self):
        if self.busy:
            return
        if self.enc is None:
            self._status("Encode a memory first  (press ENCODE or SPACE)", C_ENGRAM)
            self.fig.canvas.draw_idle()
            return
        self._stop_anim()
        w, P, e = self.world, self.P, self.enc
        mask = w.u_cue < P["cue_f"]
        I_base = 0.10 + P["kc_ret"] * w.c_i
        res = simulate_epoch(e["W"], I_base, e["I_ext"] * mask, w.c_i, False, P,
                             T_RETRIEVE, self.rng)
        fired = res["counts"] >= FIRE_MIN
        # Control: the very same cue delivered to the untrained (pre-memory) network
        pre = simulate_epoch(W0, I_base, e["I_ext"] * mask, w.c_i, False, P,
                             T_RETRIEVE, self.rng)
        self.ret = dict(res=res, mask=mask, fired=fired, tl=analyse(res["V"]),
                        pre=pre, pre_fired=pre["counts"] >= FIRE_MIN, I_base=I_base)

        showcase = np.flatnonzero(e["eng"] & ~mask & fired)
        if len(showcase):
            self.sel = int(showcase[0])
            self.sc_sel.set_offsets([[self.sel % GRID, self.sel // GRID]])

        self._setup_W()
        self._setup_curve(upto=0)
        self._draw_inspector(upto=0)
        self._ui_stage = None
        self._status("")
        self._run_anim(len(res["V"]), self.REC_SECONDS, self._frame_recall, self._done_recall)

    def _done_recall(self):
        self._refresh_static()

    # ------------------------------------------------------------- frame logic
    def _frame_encode(self, k):
        e = self.enc
        t = k * DT
        R = float(e["res"]["R"][k])
        self._apply_hero(*self._hero_encode(k, flash=True))

        tint = 0.20 * min(R, 2.0) / 2.0 + (0.05 if STIM_ON <= t < STIM_OFF else 0.0)
        self.ax_hero.set_facecolor(tuple(mix(rgb(PANEL), rgb(C_SWR), tint)))
        if t < STIM_ON:
            phase = "BASELINE"
        elif t < STIM_OFF:
            phase = "SENSORY DRIVE"
        else:
            phase = "SWR REPLAY" + ("  ⚡" if R > 0.3 else "")
        self.t_clock.set_text(f"t = {t:6.1f} ms   {phase}")

        if self._ui_stage != "enc":
            self._ui_stage = "enc"
            self._set_pills(["active", "pending", "pending"])
            self._set_hero_title("ENCODING · CREB-rich cells win the allocation race", C_ENGRAM)
        j = min(k // SNAP_EVERY, len(e["res"]["snaps"]) - 1)
        if j != self._last_j:
            self._last_j = j
            self.im_W.set_data(e["res"]["snaps"][j][np.ix_(self._order, self._order)])
        self._set_kpis(eng=int((e["tl"]["cum"][k] >= FIRE_MIN).sum()))
        if self.cursor is not None:
            self.cursor.set_xdata([t, t])
            self.cursor.set_visible(True)

    def _frame_recall(self, k):
        e, r = self.enc, self.ret
        t = k * DT
        self._apply_hero(*self._hero_recall(k, flash=True))
        self.ax_hero.set_facecolor(PANEL)
        stage = "cue" if t < STIM_ON else "rec"
        self.t_clock.set_text(f"t = {t:6.1f} ms   " +
                              ("CUE ARRIVING" if t < STIM_ON else
                               "PATTERN COMPLETION" if t < STIM_OFF else "RELAXATION"))
        if self._ui_stage != stage:
            self._ui_stage = stage
            if stage == "cue":
                self._set_pills(["done", "active", "pending"])
                self._set_hero_title(f"PARTIAL CUE · {self.P['cue_f']:.0%} of the pattern", C_CUED)
            else:
                self._set_pills(["done", "done", "active"])
                self._set_hero_title("PATTERN COMPLETION · the silent memory ignites", C_RECALL)

        fired = r["tl"]["cum"][k] >= FIRE_MIN
        tgt = e["eng"] & ~r["mask"]
        self._set_kpis(eng=int(e["eng"].sum()), cued=int((e["eng"] & r["mask"]).sum()),
                       recall=100.0 * (fired & tgt).sum() / max(tgt.sum(), 1),
                       fp=int((fired & ~e["eng"]).sum()))
        for line, y in zip(self._curve_lines, self._curve_data):
            line.set_data(r["tl"]["ts"][:k + 1], y[:k + 1])
        self._rec_k = k
        if self.ph_post is not None:
            res, i = r["res"], self.sel
            self.ph_post.set_data(res["V"][:k + 1, i], res["w"][:k + 1, i])
            self.ph_dot.set_data([res["V"][k, i]], [res["w"][k, i]])

    # ------------------------------------------------------------ hero colours
    def _heat(self, amount):
        return mix(SIL, CMAP_CREB(self.world.c_i)[:, :3], amount)

    def _hero_encode(self, k, flash):
        tl = self.enc["tl"]
        cum = tl["cum"][-1 if k is None else k]
        eng_now = cum >= FIRE_MIN
        faces = self._heat(0.55 if k is not None else 0.30)
        edges, lw = np.tile(BORD, (N, 1)), np.full(N, 1.0)
        faces[eng_now] = ENG
        edges[eng_now] = mix(ENG, WHITE, 0.35)
        lw[eng_now] = 2.2
        act = tl["act"][k] if (flash and k is not None) else np.zeros(N)
        faces = mix(faces, WHITE, 0.6 * act)
        return faces, edges, lw, 0.7 * act, mix(faces, WHITE, 0.2)

    def _hero_recall(self, k, flash):
        e, r = self.enc, self.ret
        eng, cued = e["eng"], r["mask"]
        tl = r["tl"]
        fired = tl["cum"][-1 if k is None else k] >= FIRE_MIN
        faces = self._heat(0.22)
        edges, lw = np.tile(BORD, (N, 1)), np.full(N, 1.0)

        edges[cued], lw[cued] = CUE * 0.8, 1.7
        m = eng & ~cued
        faces[m], edges[m], lw[m] = mix(SIL, ENG, 0.45), mix(ENG, SIL, 0.25), 2.2
        m = eng & cued
        faces[m], edges[m], lw[m] = mix(SIL, CUE, 0.55), CUE, 2.2
        m = fired & eng & cued
        faces[m] = CUE
        m = fired & eng & ~cued
        faces[m], edges[m], lw[m] = REC, mix(REC, WHITE, 0.4), 2.4
        m = fired & ~eng
        faces[m], edges[m], lw[m] = FAL, mix(FAL, WHITE, 0.3), 2.2

        act = tl["act"][k] if (flash and k is not None) else np.zeros(N)
        faces = mix(faces, WHITE, 0.6 * act)
        return faces, edges, lw, 0.75 * act, mix(faces, WHITE, 0.2)

    def _hero_naive(self):
        return (self._heat(0.65), np.tile(BORD, (N, 1)), np.full(N, 1.0),
                np.zeros(N), np.tile(SIL, (N, 1)))

    def _apply_hero(self, faces, edges, lw, glow, glow_rgb):
        self.sc_node.set_facecolors(faces)
        self.sc_node.set_edgecolors(edges)
        self.sc_node.set_linewidths(lw)
        self.sc_glow.set_facecolors(np.c_[glow_rgb, glow])

    # ----------------------------------------------------------- static panels
    def _refresh_static(self):
        self.ax_hero.set_facecolor(PANEL)
        self.t_clock.set_text("")
        self.sc_sel.set_offsets([[self.sel % GRID, self.sel // GRID]])

        if self.enc is None:
            self._apply_hero(*self._hero_naive())
            self._set_hero_title("NAIVE NETWORK · brighter = more CREB-excitable", MUTED)
            self._set_pills(["pending"] * 3)
            self._set_kpis()
            self._status("Press ENCODE (or SPACE) to store a memory", MUTED)
        elif self.ret is None:
            self._apply_hero(*self._hero_encode(None, flash=False))
            n = int(self.enc["eng"].sum())
            self._set_hero_title(f"MEMORY STORED · {n} of {N} neurons form the engram", C_ENGRAM)
            self._set_pills(["done", "pending", "pending"])
            self._set_kpis(eng=n)
            self._status("Press CUE & RECALL to test pattern completion", MUTED)
        else:
            self._apply_hero(*self._hero_recall(None, flash=False))
            e, r = self.enc, self.ret
            tgt = e["eng"] & ~r["mask"]
            rec = int((r["fired"] & tgt).sum())
            pct = 100.0 * rec / max(tgt.sum(), 1)
            fp = int((r["fired"] & ~e["eng"]).sum())
            cu = int((e["eng"] & r["mask"]).sum())
            self._set_hero_title(f"MEMORY RECALLED · {pct:.0f}% of the silent engram restored", C_RECALL)
            self._set_pills(["done", "done", "done"])
            self._set_kpis(eng=int(e["eng"].sum()), cued=cu, recall=pct, fp=fp)
            pre_rec = 100.0 * (r["pre_fired"] & tgt).sum() / max(tgt.sum(), 1)
            self._status(f"{rec}/{int(tgt.sum())} uncued engram neurons recalled from a "
                         f"{self.P['cue_f']:.0%} cue · {fp} false positives   |   "
                         f"same cue, untrained network: {pre_rec:.0f}% recall", C_RECALL)

        self._setup_W()
        self._setup_curve()
        self._draw_inspector()
        self.fig.canvas.draw_idle()

    def _setup_W(self, frame=None):
        ax = self.ax_W
        ax.clear()
        self._style(ax)
        self._title(ax, "SYNAPTIC MATRIX")
        if self.enc is None:
            data = W0
        elif frame is None:
            data = self.enc["W"][np.ix_(self._order, self._order)]
        else:
            data = self.enc["res"]["snaps"][frame][np.ix_(self._order, self._order)]
        self.im_W = ax.imshow(data, cmap=CMAP_SYN, vmin=0, vmax=self.P["wmax"],
                              interpolation="nearest")
        self._last_j = -1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("presynaptic  (engram cells first)", fontsize=8)
        ax.set_ylabel("postsynaptic", fontsize=8)
        if self.enc is not None and frame is None:
            n = int(self.enc["eng"].sum())
            if n > 0:
                ax.axhline(n - 0.5, color=C_ENGRAM, lw=1, ls="--")
                ax.axvline(n - 0.5, color=C_ENGRAM, lw=1, ls="--")

    def _setup_curve(self, upto=None):
        ax = self.ax_curve
        ax.clear()
        self._style(ax)
        self._title(ax, "RECALL BUILD-UP")
        self._curve_lines, self._curve_data = [], []
        if self.ret is None:
            self._placeholder(ax, "awaiting cue")
            return
        e, r = self.enc, self.ret
        tl = r["tl"]
        fired = tl["cum"] >= FIRE_MIN
        groups = [(e["eng"] & ~r["mask"], C_RECALL, "Recalled"),
                  (e["eng"] & r["mask"], C_CUED, "Cued"),
                  (~e["eng"], C_FALSE, "False +")]
        ax.axvspan(STIM_ON, STIM_OFF, color=C_CUED, alpha=0.06)
        for m, col, lab in groups:
            y = fired[:, m].sum(axis=1)
            ax.plot(tl["ts"], y, color=col, alpha=0.15, lw=1)
            line, = ax.plot([], [], color=col, lw=2.4, label=lab)
            self._curve_lines.append(line)
            self._curve_data.append(y)
        if upto is None:
            for line, y in zip(self._curve_lines, self._curve_data):
                line.set_data(tl["ts"], y)
        ymax = max(max(y.max() for y in self._curve_data), 1) * 1.12
        ax.set_xlim(0, T_RETRIEVE)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("time (ms)", fontsize=8)
        ax.set_ylabel("neurons active", fontsize=8)
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT)

    def _draw_inspector(self, cursor=False, upto=None):
        self._draw_phase(upto)
        self._draw_traces(cursor)

    def _drives(self, i):
        """Resting drive, and peak net current (3 ms smoothed) felt during the cue run.

        Recall arrives as brief recurrent surges, so the peak (not the mean) is what
        decides whether a neuron is pushed through its Hopf bifurcation.
        """
        d = dict(rest=0.10 + self.P["kc_ret"] * self.world.c_i[i])
        if self.ret is not None:
            ker = np.ones(30) / 30
            for key, run in (("pre", self.ret["pre"]), ("post", self.ret["res"])):
                d[key] = float(np.convolve(run["I"][:, i], ker, mode="same").max())
        return d

    def _draw_phase(self, upto=None):
        """Phase plane (real trajectories) or bifurcation diagram for the selected neuron.

        pre-memory  = same cue delivered to the untrained network (initial weights W0)
        post-memory = same cue delivered to the trained network (learned weights)
        upto        = animate: only draw the post-memory trajectory up to this step
        """
        ax = self.ax_ph
        ax.clear()
        self._style(ax)
        self.ph_post = self.ph_dot = None
        if self.enc is None:
            self._title(ax, "PHASE SPACE")
            self._placeholder(ax, "encode a memory, then\nclick any neuron")
            return
        i = self.sel
        d = self._drives(i)
        if self.ph_view == "bif":
            self._draw_bif(ax, i, d)
        else:
            self._draw_phase_plane(ax, i, d, upto)

    def _draw_phase_plane(self, ax, i, d, upto):
        self._title(ax, f"PHASE SPACE · #{i}", fs=9)
        ret = self.ret
        vv = np.linspace(-2.8, 3.0, 400)
        ax.axvspan(V_HOPF, -V_HOPF, color=C_FALSE, alpha=0.07)
        ax.plot(vv, (vv + A) / B, color=C_CA, lw=1.8, label="w-nullcline")

        curves = [(d["rest"], MUTED, ":", "v-nullcline: rest")]
        if ret is not None:
            curves += [(d["pre"], C_SWR, "--", f"pre-memory  peak I {d['pre']:.2f}"),
                       (d["post"], C_RECALL, "-", f"post-memory peak I {d['post']:.2f}")]
        for cur, col, ls, lab in curves:
            ax.plot(vv, vv - vv ** 3 / 3 + cur, color=col, ls=ls, lw=1.8, label=lab)
            fv, fw = fixed_point(cur)
            ax.plot(fv, fw, "o", ms=7, mfc=col if is_stable(fv[0]) else PANEL,
                    mec=col, mew=1.8, zorder=8)

        xmax, ylo, yhi = 2.6, -1.0, 2.4
        if ret is not None:
            pre, post = ret["pre"], ret["res"]
            ax.plot(pre["V"][:, i], pre["w"][:, i], color=C_SWR, lw=0.9, alpha=0.6, zorder=6)
            n = len(post["V"]) if upto is None else upto + 1
            self.ph_post, = ax.plot(post["V"][:n, i], post["w"][:n, i], color=C_RECALL,
                                    lw=1.3, alpha=0.95, zorder=7)
            self.ph_dot, = ax.plot([post["V"][n - 1, i]], [post["w"][n - 1, i]], "o", ms=8,
                                   mfc="white", mec=C_RECALL, mew=2, zorder=9)
            vmax = max(post["V"][:, i].max(), pre["V"][:, i].max())
            wmin = min(post["w"][:, i].min(), pre["w"][:, i].min())
            wmax = max(post["w"][:, i].max(), pre["w"][:, i].max(), fixed_point(d["post"])[1][0])
            xmax, ylo, yhi = max(xmax, vmax + 0.3), min(ylo, wmin - 0.3), max(yhi, wmax + 0.4)
        ax.set_xlim(-2.6, xmax)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("membrane voltage v", fontsize=8)
        ax.set_ylabel("recovery w", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False, labelcolor=TEXT)
        if ret is not None:
            reg = regime(d["post"])
            col = {"spiking range": C_RECALL, "above upper Hopf": C_ENGRAM}.get(reg, MUTED)
            ax.text(0.97, 0.04, f"post-memory peak: {reg}", transform=ax.transAxes,
                    ha="right", fontsize=7.5, fontweight="bold", color=col)

    def _draw_bif(self, ax, i, d):
        self._title(ax, f"BIFURCATION · #{i}", fs=9)
        b = bifurcation_diagram()
        post = d.get("post", d["rest"])
        xmax = max(2.0, post + 0.4)
        Ig = np.linspace(-0.5, xmax, 500)
        v, _ = fixed_point(Ig)
        stable = is_stable(v)

        ax.axvspan(I_HOPF, I_HOPF_HI, color=C_SWR, alpha=0.07)
        m = b["I"] <= xmax
        ax.fill_between(b["I"][m], b["vmin"][m], b["vmax"][m], color=C_INH, alpha=0.30,
                        lw=0, label="spike envelope")
        ax.plot(Ig, np.where(stable, v, np.nan), color=C_CUED, lw=2.0, label="stable rest")
        ax.plot(Ig, np.where(~stable, v, np.nan), color=C_FALSE, lw=1.6, ls="--", label="unstable")
        for h in (I_HOPF, I_HOPF_HI):
            ax.axvline(h, color=C_FALSE, ls=":", lw=1.1)
        ax.text(I_HOPF + 0.04, -2.2, "Hopf", color=C_FALSE, fontsize=7)
        ax.text(I_HOPF_HI + 0.04, -2.2, "Hopf", color=C_FALSE, fontsize=7)

        pts = [(d["rest"], MUTED, "rest")]
        if self.ret is not None:
            pts += [(d["pre"], C_SWR, "pre-memory"), (d["post"], C_RECALL, "post-memory")]
        for cur, col, lab in pts:
            vstar = fixed_point(cur)[0][0]
            ax.axvline(cur, color=col, lw=1.2, alpha=0.8)
            ax.plot([cur], [vstar], "o", ms=8, mfc=col, mec="white", mew=1.2, zorder=9,
                    label=f"{lab}{' peak' if lab != 'rest' else ''}  {cur:.2f}")
        ax.set_xlim(-0.5, xmax)
        ax.set_ylim(-2.4, max(2.4, fixed_point(xmax)[0][0] + 0.3))
        ax.set_xlabel("net input current I", fontsize=8)
        ax.set_ylabel("voltage  (fixed point / spikes)", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False, labelcolor=TEXT)

    def _draw_traces(self, cursor):
        ax = self.ax_tr
        ax.clear()
        self._style(ax)
        self.cursor = None
        if self.enc is None:
            self._title(ax, "TEMPORAL DYNAMICS")
            self._placeholder(ax, "encode a memory to see calcium, inhibition and SWR dynamics")
            return
        i, P, res = self.sel, self.P, self.enc["res"]
        t = np.arange(len(res["V"])) * DT
        theta = P["theta_p"]
        ca = res["X"][:, i]
        role = "ENGRAM" if self.enc["eng"][i] else "SILENT"
        self._title(ax, f"TEMPORAL DYNAMICS · neuron #{i} ({role}, CREB {self.world.c_i[i]:.2f})")

        ax.axvspan(STIM_ON, STIM_OFF, color=C_ENGRAM, alpha=0.07)
        ax.text(STIM_ON + 3, 7.4, "sensory encoding", color=C_ENGRAM, fontsize=8.5)
        ax.text(SWR_START + 3, 7.4, "SWR replay", color=C_SWR, fontsize=8.5)
        ax.plot(t, res["V"][:, i] + 5.0, color=TEXT, lw=1.0, label="Voltage")
        ax.axhline(5.0, color=BORDER, ls="--", lw=0.8)
        ax.plot(t, ca, color=C_CA, lw=1.6, label="Ca$^{2+}$ trace")
        ax.axhline(theta, color=C_FALSE, ls="--", lw=1.2, label="θ$_p$ threshold")
        ax.fill_between(t, theta, ca, where=ca > theta, color=C_RECALL, alpha=0.45)
        ax.plot(t, res["inh"] - 1.5, color=C_INH, lw=1.6, label="PV$^{+}$ inhibition")
        ax.axhline(-1.5, color=BORDER, ls="--", lw=0.8)
        swr = -3.5 + 1.5 * res["R"]
        ax.fill_between(t, -3.5, swr, color=C_SWR, alpha=0.25)
        ax.plot(t, swr, color=C_SWR, lw=1.0, label="SWR burst")

        ax.set_yticks([-3.5, -1.5, 0, theta, 5.0])
        ax.set_yticklabels(["SWR", "Inhib.", "0", "θ$_p$", "V"], fontsize=8.5)
        ax.set_xlim(0, T_ENCODE)
        ax.set_ylim(-4.3, 8.2)
        ax.set_xlabel("time (ms)", fontsize=8)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=5, fontsize=8,
                  frameon=False, labelcolor=TEXT)
        self.cursor = ax.axvline(0, color="white", lw=1.3, alpha=0.8, visible=cursor, zorder=9)


if __name__ == "__main__":
    app = App()
    plt.show()