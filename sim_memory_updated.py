"""
Emergent SWR, CREB-allocated, PV-inhibited FitzHugh-Nagumo engram model.

Interactive visualization: a 12x12 network of FitzHugh-Nagumo neurons is
trained with burst-threshold plasticity (CREB-weighted LTP/LTD, SWR replay,
global PV+ inhibition), then probed with a partial cue to test pattern
completion. Adjust parameters with the sliders and press "Re-Simulate".
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
BORDER_COLOR = "#30363d"
TEXT_COLOR = "#c9d1d9"
MUTED_COLOR = "#8b949e"
TITLE_COLOR = "#f0f6fc"

C_ENGRAM_CUED = "#388bfd"
C_ENGRAM_RECALL = "#3fb950"
C_ENGRAM_MISSED = "#d29922"
C_FALSE_POS = "#f85149"
C_SILENT = "#21262d"
C_INHIBITION = "#bc8cff"
C_SWR = "#58a6ff"
C_CALCIUM = "#ff7b72"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": BORDER_COLOR, "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR, "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR, "grid.color": BORDER_COLOR,
    "font.family": "sans-serif", "font.size": 9,
})

MAIN_TITLE = "Emergent Memory Allocation: PV+ Inhibition & Burst-Threshold Plasticity"

# --------------------------------------------------------------------------
# Model constants
# --------------------------------------------------------------------------
GRID = 12
N = GRID * GRID

# FitzHugh-Nagumo parameters
A, B, EPS = 0.7, 0.8, 0.08
DT = 0.1
V_HOPF = -np.sqrt(1 - EPS * B)
I_HOPF = (V_HOPF + A) / B - V_HOPF + V_HOPF ** 3 / 3

FIRE_MIN = 2  # spikes required (during stimulus window) to count as "fired"

# Protocol timing (ms)
T_ENCODE = 240.0
T_RETRIEVE = 150.0
STIM_ON, STIM_OFF = 20.0, 110.0
SWR_START = 110.0

# Time constants (ms)
TAU_SYN = 5.0
TAU_CA = 25.0
TAU_SWR = 15.0
TAU_INH = 12.0

# Fixed random connectivity (15% density, weak initial weights)
_rng_w = np.random.RandomState(42)
_conn = _rng_w.rand(N, N) < 0.15
np.fill_diagonal(_conn, 0)
W0 = np.where(_conn, 0.008 * _rng_w.rand(N, N), 0.0)

DEFAULTS = dict(
    kc_enc=0.15,    # CREB gain, encoding
    kc_ret=0.00,    # CREB gain, retrieval (0 => recall must be synaptic)
    s0=0.20,        # stimulus drive
    p_swr=0.004,    # SWR Poisson rate per step
    g_swr=0.25,     # SWR replay drive
    g_inh=2.5,      # PV+ winner-take-all inhibition
    eta=0.45,       # plasticity rate
    ratio=0.45,     # LTD/LTP ratio
    wmax=0.35,      # soft weight bound
    theta_p=0.85,   # Ca2+ plasticity threshold
    cue_f=0.40,     # fraction of stimulus given as cue at retrieval
    sigma_v=0.04,   # voltage noise
)


# --------------------------------------------------------------------------
# Dynamics helpers
# --------------------------------------------------------------------------
def fixed_point(current):
    """Newton solve for the FHN fixed point (v*, w*) at input current."""
    current = np.atleast_1d(np.asarray(current, float))
    v = np.full_like(current, -1.0)
    for _ in range(40):
        f = v ** 3 / 3 + (1 / B - 1) * v + (A / B - current)
        v = v - f / (v ** 2 + (1 / B - 1))
    return v, (v + A) / B


def is_stable(v):
    """Linear stability of the FHN fixed point at voltage v."""
    return ((1 - v ** 2) - EPS * B < 0) and (EPS * (1 - B * (1 - v ** 2)) > 0)


class World:
    """Per-seed random neuron properties."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.c_i = rng.uniform(0.1, 1.0, N)    # CREB competence
        self.p = rng.uniform(0.0, 1.0, N)      # stimulus pattern
        self.u_cue = rng.random(N)             # cue-membership draw


def simulate_epoch(W_init, I_base, I_ext, c_i, learn, P, T_total, rng):
    """Run one epoch. Returns spike counts, final W, and trace histories."""
    W = W_init.copy()
    n_steps = int(T_total / DT)

    v, w = fixed_point(I_base)
    s, x, y = np.zeros(N), np.zeros(N), np.zeros(N)
    I_inh, R = 0.0, 0.0
    counts = np.zeros(N, int)

    dec_s = np.exp(-DT / TAU_SYN)
    dec_ca = np.exp(-DT / TAU_CA)
    dec_R = np.exp(-DT / TAU_SWR)

    noise = rng.normal(0, P["sigma_v"] * np.sqrt(DT), (n_steps, N))

    V_hist = np.empty((n_steps, N), np.float32)
    x_hist = np.empty((n_steps, N), np.float32)
    R_hist = np.empty(n_steps, np.float32)
    inh_hist = np.empty(n_steps, np.float32)

    for k in range(n_steps):
        t = k * DT
        in_stim = STIM_ON <= t < STIM_OFF

        # SWR envelope: Poisson events stack; otherwise decay
        if learn and t > SWR_START and rng.random() < P["p_swr"]:
            R += 1.0
        else:
            R *= dec_R

        # Global PV+ inhibition tracks mean synaptic activity
        I_inh += DT * (-I_inh + P["g_inh"] * s.mean()) / TAU_INH

        I = I_base + W @ s - I_inh
        if in_stim:
            I += I_ext
        if R > 0.01:
            I += P["g_swr"] * R

        # FitzHugh-Nagumo Euler step
        vn = np.clip(v + DT * (v - v ** 3 / 3 - w + I) + noise[k], -3.0, 3.0)
        wn = w + DT * EPS * (vn + A - B * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn

        V_hist[k], x_hist[k], R_hist[k], inh_hist[k] = v, x, R, I_inh

        if in_stim:
            counts += spk

        s = s * dec_s + spk
        x = x * dec_ca + spk
        y = y * dec_ca + spk

        # Burst-threshold plasticity with soft bounds
        if learn and spk.any():
            sp = spk.astype(float)
            x_burst = np.maximum(x - P["theta_p"], 0.0)
            y_burst = np.maximum(y - P["theta_p"], 0.0)
            ltp = np.outer(c_i * sp, x_burst) * (P["wmax"] - W)
            ltd = np.outer(c_i * y_burst, sp) * W
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            np.fill_diagonal(W, 0.0)

    return counts, W, V_hist, x_hist, R_hist, inh_hist


def compute_run(world, P):
    """Encode with CREB-biased drive + learning, then retrieve with a cue."""
    rng = np.random.default_rng()
    I_ext = P["s0"] * world.p

    I_base_enc = 0.10 + P["kc_enc"] * world.c_i
    cnt_enc, W, Venc, Xenc, Renc, Inh_enc = simulate_epoch(
        W0, I_base_enc, I_ext, world.c_i, True, P, T_ENCODE, rng)
    engram = cnt_enc >= FIRE_MIN

    mask = (world.u_cue < P["cue_f"]).astype(float)
    I_base_ret = 0.10 + P["kc_ret"] * world.c_i
    cnt_ret, _, *_ = simulate_epoch(
        W, I_base_ret, I_ext * mask, world.c_i, False, P, T_RETRIEVE, rng)

    return {
        "W": W, "eng": engram, "fired": cnt_ret >= FIRE_MIN, "mask": mask,
        "I_base_ret": I_base_ret, "I_ext": I_ext,
        "Venc": Venc, "Xenc": Xenc, "Renc": Renc, "Inh_enc": Inh_enc,
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
CMAP = ListedColormap([C_SILENT, C_ENGRAM_MISSED, C_ENGRAM_CUED,
                       C_ENGRAM_RECALL, C_FALSE_POS])
LEGEND = [Patch(color=c, label=l) for c, l in [
    (C_SILENT, "Silent"), (C_ENGRAM_MISSED, "Missed"), (C_ENGRAM_CUED, "Cued"),
    (C_ENGRAM_RECALL, "Recalled"), (C_FALSE_POS, "False Pos.")]]

SLIDER_SPECS = [  # label, key, lo, hi, step, row, col
    ("CREB Gain (kc)", "kc_enc", 0.0, 0.40, 0.01, 0, 0),
    ("Inhibition (g_inh)", "g_inh", 0.0, 6.0, 0.1, 0, 1),
    ("SWR Rate (p_swr)", "p_swr", 0.0, 0.02, 0.001, 0, 2),
    ("SWR Drive (g_swr)", "g_swr", 0.0, 0.50, 0.01, 1, 0),
    ("Plasticity (eta)", "eta", 0.0, 1.0, 0.05, 1, 1),
    ("Ca2+ Thresh (theta_p)", "theta_p", 0.5, 2.0, 0.1, 1, 2),
]


def as_grid(z):
    return np.asarray(z).reshape(GRID, GRID)


class App:
    def __init__(self, seed=101):
        self.seed = seed
        self.world = World(seed)
        self.P = dict(DEFAULTS)
        self.sel = None
        self.res = None
        self.sliders = {}
        self._build_ui()
        self.recompute()

    # ---- construction ----------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9.5))
        gs = self.fig.add_gridspec(
            3, 4, height_ratios=[1.2, 1.2, 0.5], hspace=0.4, wspace=0.25,
            top=0.92, bottom=0.04, left=0.04, right=0.96)

        self.ax_creb = self.fig.add_subplot(gs[0, 0])
        self.ax_stim = self.fig.add_subplot(gs[0, 1])
        self.ax_out = self.fig.add_subplot(gs[0, 2])
        self.ax_W = self.fig.add_subplot(gs[0, 3])
        self.ax_txt = self.fig.add_subplot(gs[1, 0])
        self.ax_ph = self.fig.add_subplot(gs[1, 1])
        self.ax_tr = self.fig.add_subplot(gs[1, 2:])

        self._set_title(MAIN_TITLE, TITLE_COLOR)

        gs_ctrl = gs[2, :].subgridspec(2, 5, hspace=0.7, wspace=0.3)
        for label, key, lo, hi, step, r, c in SLIDER_SPECS:
            ax = self.fig.add_subplot(gs_ctrl[r, c])
            sl = Slider(ax, label, lo, hi, valinit=self.P[key],
                        valstep=step, color=C_SWR)
            sl.label.set_fontsize(9)
            sl.valtext.set_fontsize(9)
            sl.on_changed(lambda v, k=key: self._on_param(k, v))
            self.sliders[key] = sl

        self.b_run = self._button(gs_ctrl[0, 4], "Re-Simulate", bold=True)
        self.b_run.on_clicked(lambda _e: self.recompute())
        self.b_seed = self._button(gs_ctrl[1, 4], "New World Seed")
        self.b_seed.on_clicked(self._new_world)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _button(self, spec, label, bold=False):
        btn = Button(self.fig.add_subplot(spec), label,
                     color=PANEL_COLOR, hovercolor=BORDER_COLOR)
        btn.label.set_color(TEXT_COLOR)
        if bold:
            btn.label.set_fontweight("bold")
        return btn

    def _set_title(self, text, color):
        self.fig.suptitle(text, fontsize=15, fontweight="bold",
                          color=color, y=0.98)

    # ---- callbacks -------------------------------------------------------
    def _on_param(self, key, value):
        self.P[key] = float(value)
        self._set_title("Parameters Changed - Press 'Re-Simulate' to Update",
                        C_ENGRAM_MISSED)
        self.fig.canvas.draw_idle()

    def _new_world(self, _event=None):
        self.seed += 1
        self.world = World(self.seed)
        self.sel = None
        self.recompute()

    def _on_click(self, ev):
        if ev.inaxes in (self.ax_creb, self.ax_stim, self.ax_out) \
                and ev.xdata is not None:
            col = int(np.clip(round(ev.xdata), 0, GRID - 1))
            row = int(np.clip(round(ev.ydata), 0, GRID - 1))
            self.sel = row * GRID + col
            self.draw_all()

    def recompute(self):
        self._set_title("Simulating... Please wait.", TEXT_COLOR)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.res = compute_run(self.world, self.P)
        if self.sel is None:
            eng = np.flatnonzero(self.res["eng"])
            self.sel = int(eng[0]) if len(eng) else 0

        self._set_title(MAIN_TITLE, TITLE_COLOR)
        self.draw_all()

    # ---- shared helpers --------------------------------------------------
    def _syn_input(self):
        """Recurrent synaptic input to the selected neuron from recalled cells."""
        R, i = self.res, self.sel
        sources = R["fired"] & (np.arange(N) != i)
        return R["W"][i, sources].sum()

    def _ring(self, ax):
        ax.scatter([self.sel % GRID], [self.sel // GRID], s=220,
                   facecolors="none", edgecolors="#ffffff",
                   linewidths=2, zorder=10)

    # ---- drawing ---------------------------------------------------------
    def draw_all(self):
        self.draw_maps()
        self.draw_W()
        self.draw_text()
        self.draw_phase()
        self.draw_traces()
        self.fig.canvas.draw_idle()

    def draw_maps(self):
        R, P = self.res, self.P

        ax = self.ax_creb
        ax.clear()
        ax.imshow(as_grid(self.world.c_i), cmap="magma", vmin=0, vmax=1)
        ax.set_title("1. CREB Competence (c_i)", fontsize=10, fontweight="bold")

        ax = self.ax_stim
        ax.clear()
        ax.imshow(as_grid(P["s0"] * self.world.p), cmap="inferno",
                  vmin=0, vmax=0.35)
        ey, ex = np.where(as_grid(R["eng"]))
        ax.scatter(ex, ey, s=90, marker="s", facecolors="none",
                   edgecolors=C_SWR, linewidths=1.2)
        ax.set_title(f"2. Encoding (Engram: {R['eng'].sum()}/{N})",
                     fontsize=10, fontweight="bold")

        eng, fired, cued = R["eng"], R["fired"], R["mask"] == 1
        outcome = np.zeros(N, int)
        outcome[eng & ~fired] = 1            # missed
        outcome[eng & fired & cued] = 2      # cued
        outcome[eng & fired & ~cued] = 3     # recalled
        outcome[~eng & fired] = 4            # false positive

        ax = self.ax_out
        ax.clear()
        ax.imshow(as_grid(outcome), cmap=CMAP, vmin=0, vmax=4)
        ax.set_title("3. Pattern Completion", fontsize=10, fontweight="bold")
        ax.legend(handles=LEGEND, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)

        for ax in (self.ax_creb, self.ax_stim, self.ax_out):
            self._ring(ax)
            ax.set_xticks([])
            ax.set_yticks([])

    def draw_W(self):
        R, ax = self.res, self.ax_W
        ax.clear()
        order = np.r_[np.flatnonzero(R["eng"]), np.flatnonzero(~R["eng"])]
        ax.imshow(R["W"][np.ix_(order, order)], cmap="viridis",
                  vmin=0, vmax=self.P["wmax"])
        n_eng = int(R["eng"].sum())
        if n_eng > 0:
            ax.axhline(n_eng - 0.5, color=C_SWR, lw=1, ls="--")
            ax.axvline(n_eng - 0.5, color=C_SWR, lw=1, ls="--")
        ax.set_title("4. Synaptic Matrix (W)", fontsize=10, fontweight="bold")

    def draw_text(self):
        R, i, ax = self.res, self.sel, self.ax_txt
        ax.clear()
        ax.axis("off")

        targets = R["eng"] & (R["mask"] == 0)  # engram cells not directly cued
        recall = R["fired"][targets].mean() if targets.any() else 0.0
        syn = self._syn_input()
        drive = R["I_base_ret"][i]
        role = "ENGRAM" if R["eng"][i] else "SILENT"

        txt = (f"NETWORK METRICS\n"
               f"───────────────\n"
               f"Engram Size   : {R['eng'].sum()}\n"
               f"Uncued Targets: {targets.sum()}\n"
               f"Recall Rate   : {recall * 100:.1f}%\n\n"
               f"NEURON #{i} ({role})\n"
               f"───────────────\n"
               f"CREB Score    : {self.world.c_i[i]:.2f}\n"
               f"Ret. Drive    : {drive:.3f}\n"
               f"Rec. Syn Input: +{syn:.3f}\n"
               f"Total Margin  : {drive + syn - I_HOPF:+.3f}")
        ax.text(0.1, 0.95, txt, va="top", family="monospace",
                fontsize=10, color=TEXT_COLOR)

    def draw_phase(self):
        R, i, ax = self.res, self.sel, self.ax_ph
        ax.clear()
        I_rest = R["I_base_ret"][i]
        I_total = I_rest + self._syn_input()

        ax.axvspan(V_HOPF, -V_HOPF, color=C_FALSE_POS, alpha=0.1,
                   label="Unstable Zone")
        ax.axvline(V_HOPF, color=C_FALSE_POS, ls=":", lw=1.5)

        vv = np.linspace(-2.5, 2.5, 400)
        ax.plot(vv, (vv + A) / B, color=C_CALCIUM, lw=2.0, label="w-nullcline")

        for current, col, ls, lab in [(I_rest, MUTED_COLOR, "--", "Resting"),
                                      (I_total, C_SWR, "-", "w/ Synaptic Input")]:
            ax.plot(vv, vv - vv ** 3 / 3 + current, color=col, ls=ls,
                    lw=2.5, label=lab)
            fv, fw = fixed_point(current)
            ax.plot(fv, fw, "o", ms=9,
                    mfc=col if is_stable(fv[0]) else PANEL_COLOR,
                    mec=col, mew=2, zorder=8)

        ax.set_title(f"5. Phase Space (Neuron {i})", fontsize=10,
                     fontweight="bold")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.1, 2.3)
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    def draw_traces(self):
        R, i, P, ax = self.res, self.sel, self.P, self.ax_tr
        ax.clear()
        t = np.arange(len(R["Venc"])) * DT
        theta = P["theta_p"]
        ca = R["Xenc"][:, i]

        ax.axvspan(STIM_ON, STIM_OFF, color=C_ENGRAM_MISSED, alpha=0.1)
        ax.text(STIM_ON + 5, 6.5, "Sensory Encoding",
                color=C_ENGRAM_MISSED, fontsize=9)

        # Voltage (offset +5)
        ax.plot(t, R["Venc"][:, i] + 5.0, color=TEXT_COLOR, lw=1.0,
                label="Voltage V_i")
        ax.axhline(5.0, color=BORDER_COLOR, ls="--", lw=0.8)

        # Ca2+ trace vs plasticity threshold
        ax.plot(t, ca, color=C_CALCIUM, lw=1.5, label="Ca2+ Trace x_i(t)")
        ax.axhline(theta, color=C_FALSE_POS, ls="--", lw=1.2,
                   label="Plasticity Thresh θ_p")
        ax.fill_between(t, theta, ca, where=(ca > theta),
                        color=C_ENGRAM_RECALL, alpha=0.4)

        # Global inhibition (offset -1.5)
        ax.plot(t, R["Inh_enc"] - 1.5, color=C_INHIBITION, lw=1.5,
                label="PV+ Inhibition I_inh")
        ax.axhline(-1.5, color=BORDER_COLOR, ls="--", lw=0.8)

        # SWR envelope (offset -3.5)
        swr = -3.5 + 1.5 * R["Renc"]
        ax.fill_between(t, -3.5, swr, color=C_SWR, alpha=0.2)
        ax.plot(t, swr, color=C_SWR, lw=1.0, label="SWR Burst R(t)")

        ax.set_yticks([-3.5, -1.5, 0, theta, 5.0])
        ax.set_yticklabels(["SWR", "Inhibition", "0", "θ_p", "Voltage"],
                           fontsize=9)
        ax.set_xlim(0, T_ENCODE)
        ax.set_title("6. Temporal Dynamics (SWR Stacking & Inhibition)",
                     fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=5, frameon=False)


if __name__ == "__main__":
    App()
    plt.show()