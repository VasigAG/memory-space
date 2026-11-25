"""
Morris-Lecar bifurcation explorer
=================================

    C dV/dt = I - gL (V-EL) - gCa m_inf(V) (V-ECa) - gK w (V-EK)
      dw/dt = phi (w_inf(V) - w) cosh((V-V3)/(2 V4))

    m_inf(V) = 0.5 (1 + tanh((V-V1)/V2))      w_inf(V) = 0.5 (1 + tanh((V-V3)/V4))

Every parameter is a live slider:
  * V-nullcline shape   : gCa, gK, gL, EL, ECa, EK, V1, V2   (+ input current I)
  * w-nullcline shape   : V3 (position), V4 (steepness)
  * flow / time scales  : phi (K+ speed), C

Extra panels
  * Equilibria-vs-I diagram (stable = white, saddle = yellow, unstable = red).
      click it to set I, scroll wheel to zoom the I range.
  * "Scan cycles" overlays the min/max voltage of stable limit cycles found by
    simulation from several initial conditions (orange).  Where orange sits on
    top of a white branch you have bistability (rest + oscillation).
  * Regime buttons load parameter sets for monostability, bistability,
    saddle-node, SNIC, Hopf, homoclinic ... (they are starting points: drag
    I and the other sliders to cross the bifurcations yourself).

Keys: space = pulse, R = reset, left/right = I -/+ 0.1, up/down = I +/- 1.
"""
import math
import itertools
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------------
# Morris-Lecar model (parameters live in module globals and are edited in place)
# ----------------------------------------------------------------------------
DEFAULTS = dict(C=20.0, gL=2.0, gCa=4.0, gK=8.0,
                EL=-60.0, ECa=120.0, EK=-84.0,
                V1=-1.2, V2=18.0, V3=12.0, V4=17.4, PHI=0.23)
C = gL = gCa = gK = EL = ECa = EK = V1 = V2 = V3 = V4 = PHI = 0.0


def set_params(**kw):
    globals().update(kw)


set_params(**DEFAULTS)

# Plotting window
VMIN, VMAX = -100.0, 60.0
WMIN, WMAX = -0.03, 1.0
DV, DW = VMAX - VMIN, WMAX - WMIN


def rhs(V, w, I):
    m = 0.5 * (1.0 + math.tanh((V - V1) / V2))
    wi = 0.5 * (1.0 + math.tanh((V - V3) / V4))
    dV = (I - gL * (V - EL) - gCa * m * (V - ECa) - gK * w * (V - EK)) / C
    dw = PHI * (wi - w) * math.cosh(min(max((V - V3) / (2.0 * V4), -50.0), 50.0))
    return dV, dw


def rhs_vec(V, w, I):
    m = 0.5 * (1.0 + np.tanh((V - V1) / V2))
    wi = 0.5 * (1.0 + np.tanh((V - V3) / V4))
    dV = (I - gL * (V - EL) - gCa * m * (V - ECa) - gK * w * (V - EK)) / C
    dw = PHI * (wi - w) * np.cosh(np.clip((V - V3) / (2.0 * V4), -50.0, 50.0))
    return dV, dw


def rk4(V, w, I, dt):
    k1 = rhs(V, w, I)
    k2 = rhs(V + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], I)
    k3 = rhs(V + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], I)
    k4 = rhs(V + dt * k3[0], w + dt * k3[1], I)
    return (V + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            w + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def w_inf(V):
    return 0.5 * (1.0 + np.tanh((np.asarray(V) - V3) / V4))


def v_null(V, I):
    """w-value of the V-nullcline (dV/dt = 0)."""
    V = np.asarray(V, dtype=float)
    m = 0.5 * (1.0 + np.tanh((V - V1) / V2))
    return (I - gL * (V - EL) - gCa * m * (V - ECa)) / (gK * (V - EK))


def _jac_entries(V, w):
    """Jacobian entries (work for floats and arrays)."""
    a = np.clip((V - V3) / (2.0 * V4), -50, 50)
    m = 0.5 * (1.0 + np.tanh((V - V1) / V2))
    dm = 0.5 / V2 / np.cosh((V - V1) / V2) ** 2
    wi = 0.5 * (1.0 + np.tanh((V - V3) / V4))
    dwi = 0.5 / V4 / np.cosh((V - V3) / V4) ** 2
    J00 = (-gL - gCa * (dm * (V - ECa) + m) - gK * w) / C
    J01 = -gK * (V - EK) / C
    J10 = PHI * (dwi * np.cosh(a) + (wi - w) * np.sinh(a) / (2.0 * V4))
    J11 = -PHI * np.cosh(a)
    return J00, J01, J10, J11


def jacobian(V, w, I):
    J00, J01, J10, J11 = _jac_entries(V, w)
    return np.array([[J00, J01], [J10, J11]], dtype=float)


def equilibria(I):
    V = np.linspace(EK + 0.1, VMAX, 900)
    g = v_null(V, I) - w_inf(V)
    out = []
    for i in np.where(g[:-1] * g[1:] < 0)[0]:
        a, b = V[i], V[i + 1]
        ga = g[i]
        for _ in range(24):
            mid = 0.5 * (a + b)
            gm = float(v_null(mid, I) - w_inf(mid))
            if ga * gm <= 0:
                b = mid
            else:
                a, ga = mid, gm
        Ve = 0.5 * (a + b)
        we = float(w_inf(Ve))
        J = jacobian(Ve, we, I)
        det, tr = np.linalg.det(J), np.trace(J)
        kind = "saddle" if det < 0 else ("stable" if tr < 0 else "unstable")
        out.append(dict(V=Ve, w=we, kind=kind, J=J, I=I))
    return out


def rest_point(I, eqs=None):
    eqs = equilibria(I) if eqs is None else eqs
    st = [e for e in eqs if e["kind"] == "stable"]
    if st:
        return st[0]["V"], st[0]["w"]
    return -60.0, 0.0


def bif_diagram(Ilo, Ihi, n=400):
    """All equilibria for n values of I (vectorised). kind: 0 stable, 1 saddle, 2 unstable."""
    Vg = np.linspace(EK + 0.1, VMAX, 1600)
    m = 0.5 * (1.0 + np.tanh((Vg - V1) / V2))
    base = -gL * (Vg - EL) - gCa * m * (Vg - ECa)
    den = gK * (Vg - EK)
    wi = w_inf(Vg)
    Is = np.linspace(Ilo, Ihi, n)
    g = (Is[:, None] + base[None, :]) / den[None, :] - wi[None, :]
    ii, jj = np.nonzero(g[:, :-1] * g[:, 1:] < 0)
    g0, g1 = g[ii, jj], g[ii, jj + 1]
    Ve = Vg[jj] + (Vg[jj + 1] - Vg[jj]) * g0 / (g0 - g1)
    we = w_inf(Ve)
    J00, J01, J10, J11 = _jac_entries(Ve, we)
    det = J00 * J11 - J01 * J10
    tr = J00 + J11
    kind = np.where(det < 0, 1, np.where(tr < 0, 0, 2))
    return Is[ii], Ve, kind


def scan_cycles(Ilo, Ihi, n=80, T=2500.0, dt=0.25):
    """Simulate from several initial conditions for n values of I (vectorised);
    return (I, Vmin, Vmax) of every run that is still oscillating at the end."""
    Is = np.linspace(Ilo, Ihi, n)
    ics = [(-70.0, 0.0), (-20.0, 0.0), (30.0, 0.3), (0.0, 0.9), (-60.0, 0.6)]
    k = len(ics)
    I = np.tile(Is, (k, 1))
    V = np.array([[ic[0]] * n for ic in ics])
    w = np.array([[ic[1]] * n for ic in ics])
    N = int(T / dt)
    N0 = int(0.6 * N)
    vmax = np.full_like(V, -1e9)
    vmin = np.full_like(V, 1e9)
    with np.errstate(all="ignore"):
        for step in range(N):
            k1 = rhs_vec(V, w, I)
            k2 = rhs_vec(V + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], I)
            k3 = rhs_vec(V + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], I)
            k4 = rhs_vec(V + dt * k3[0], w + dt * k3[1], I)
            V = V + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w = w + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            V = np.clip(V, -300, 300)
            w = np.clip(w, -3, 4)
            if step >= N0:
                vmax = np.maximum(vmax, V)
                vmin = np.minimum(vmin, V)
    osc = (vmax - vmin) > 4.0
    return I[osc], vmin[osc], vmax[osc]


def separatrix(saddle, unstable_pt=None, h=0.010, nmax=450):
    """Stable manifold of the saddle integrated backward in time."""
    J = saddle["J"]
    ev, evec = np.linalg.eig(J)
    k = int(np.argmin(ev.real))
    v = np.real(evec[:, k])
    vn = np.array([v[0] / DV, v[1] / DW])
    vn /= np.linalg.norm(vn)
    p0 = np.array([(saddle["V"] - VMIN) / DV, (saddle["w"] - WMIN) / DW])
    up = None
    if unstable_pt is not None:
        up = np.array([(unstable_pt["V"] - VMIN) / DV, (unstable_pt["w"] - WMIN) / DW])

    def F(x, y, I):
        Vv, ww = VMIN + x * DV, WMIN + y * DW
        u, vv = rhs(Vv, ww, I)
        u, vv = u / DV, vv / DW
        s = math.hypot(u, vv)
        d = max(s, 4e-3)
        return -u / d, -vv / d

    branches = []
    for sgn in (+1.0, -1.0):
        x, y = p0 + sgn * 0.005 * vn
        pts = [(saddle["V"], saddle["w"]), (VMIN + x * DV, WMIN + y * DW)]
        for _ in range(nmax):
            k1 = F(x, y, saddle["I"])
            k2 = F(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], saddle["I"])
            k3 = F(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], saddle["I"])
            k4 = F(x + h * k3[0], y + h * k3[1], saddle["I"])
            x += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if not (-0.02 < x < 1.02 and -0.02 < y < 1.02):
                break
            pts.append((VMIN + x * DV, WMIN + y * DW))
            if up is not None and math.hypot(x - up[0], y - up[1]) < 0.015:
                break
        branches.append(np.array(pts))
    return branches


def streamlines(I, seeds, n=35, h=0.012):
    def vel(p):
        Vv = VMIN + p[:, 0] * DV
        ww = WMIN + p[:, 1] * DW
        u, v = rhs_vec(Vv, ww, I)
        u, v = u / DV, v / DW
        s = np.hypot(u, v)
        d = np.maximum(s, 4e-3)
        return u / d, v / d, s

    def march(sign):
        p = seeds.copy()
        pts = []
        for _ in range(n):
            u, v, _s = vel(p)
            pm = p + 0.5 * sign * h * np.stack([u, v], 1)
            u, v, _s = vel(pm)
            p = p + sign * h * np.stack([u, v], 1)
            pts.append(p.copy())
        return np.stack(pts, 1)

    fwd = march(+1.0)
    bwd = march(-1.0)[:, ::-1]
    P = np.concatenate([bwd, seeds[:, None, :], fwd], axis=1)
    out = (P[..., 0] < -0.01) | (P[..., 0] > 1.01) | (P[..., 1] < -0.01) | (P[..., 1] > 1.01)
    P[out] = np.nan
    Vv = VMIN + P[..., 0] * DV
    ww = WMIN + P[..., 1] * DW
    u, v = rhs_vec(np.nan_to_num(Vv), np.nan_to_num(ww), I)
    speed = np.hypot(u / DV, v / DW)
    return np.stack([Vv, ww], -1), speed


# ----------------------------------------------------------------------------
# Simulation state
# ----------------------------------------------------------------------------
class Neuron:
    DT = 0.05  # ms

    def __init__(self, I):
        self.reset(I)

    def reset(self, I):
        self.V, self.w = rest_point(I)
        self.t = 0.0
        self.pulse_amp = 0.0
        self.pulse_end = -1.0
        self.spikes = 0
        self.last_spike_t = -1e9

    def kick(self, amp, dur):
        self.pulse_amp, self.pulse_end = amp, self.t + dur
        self.spikes = 0
        self.last_spike_t = -1e9

    def place(self, V, w):
        self.V, self.w = V, w
        self.pulse_amp, self.pulse_end = 0.0, -1.0
        self.spikes = 0
        self.last_spike_t = -1e9

    @property
    def pulsing(self):
        return self.t < self.pulse_end

    def step(self, I):
        Ieff = I + (self.pulse_amp if self.t < self.pulse_end else 0.0)
        Vold = self.V
        self.V, self.w = rk4(self.V, self.w, Ieff, self.DT)
        self.t += self.DT
        if Vold < 0.0 <= self.V:
            self.spikes += 1
            self.last_spike_t = self.t


# ----------------------------------------------------------------------------
# Regime presets: name -> dict(p=parameter overrides on DEFAULTS, I, Irange, note)
# ----------------------------------------------------------------------------
REGIMES = {
    # rest + saddle + unstable focus; stable cycle appears at I~35.3 (coexists with rest),
    # Hopf at I~36.3, saddle-node at I~40.0
    "Homoclinic (default)": dict(p={}, I=36.0, Irange=(0.0, 45.0)),
    # same params, I=38: TWO stable rest states (after Hopf) + saddle, and a cycle around them
    "2 rests: Hopf+fold (I=38)": dict(p={}, I=38.0, Irange=(30.0, 45.0)),
    # saddle-node on invariant circle: cycles of arbitrarily low frequency, I_c ~ 39.97
    "SNIC (type I)": dict(p=dict(PHI=0.067), I=39.0, Irange=(30.0, 50.0)),
    # subcritical Hopf at I~93.9; cycles exist from I~88.5 -> rest/cycle bistability
    "Hopf subcritical": dict(p=dict(gCa=4.4, PHI=0.04, V3=2.0, V4=30.0), I=91.0, Irange=(80.0, 105.0)),
    # supercritical Hopf at I~72.4: oscillation grows from zero amplitude
    "Hopf supercritical": dict(p=dict(gCa=4.0, PHI=0.3, V3=2.0, V4=17.4), I=70.0, Irange=(60.0, 90.0)),
    # S-shaped V-nullcline: two stable rest states + saddle for I in ~[-47, 43] (hysteresis, no cycles)
    "Bistable S-shaped": dict(p=dict(gCa=5.0, V1=-20.0, V2=8.0, V3=-10.0, V4=15.0, gK=6.0, PHI=0.1),
                              I=0.0, Irange=(-80.0, 60.0)),
    # single globally attracting rest state, no cycles
    "Monostable": dict(p=dict(gCa=2.0), I=10.0, Irange=(0.0, 60.0)),
    # no stable equilibrium: only the unstable focus -> forced oscillation
    "Oscillator (no rest)": dict(p=dict(gCa=4.4, PHI=0.04, V3=2.0, V4=30.0), I=105.0, Irange=(80.0, 130.0)),
}

PULSE_DEMOS = {
    "No spike": (60.0, 3.0),
    "Fire once": (230.0, 3.0),
    "Continuous": (125.0, 3.0),
}

# ----------------------------------------------------------------------------
# GUI Application
# ----------------------------------------------------------------------------
BG, PANEL, FG = "#06070c", "#12141c", "#c8ccd8"
C_V, C_W, C_SEP = "#ff2d5a", "#35d0ba", "#ffc247"

# name, lo, hi, step, label, colour
PARAM_SPECS = [
    ("gCa", 0.0, 12.0, 0.05, "gCa", C_V),
    ("gK", 0.5, 20.0, 0.1, "gK", C_V),
    ("gL", 0.0, 6.0, 0.05, "gL", C_V),
    ("EL", -90.0, -30.0, 0.5, "EL", C_V),
    ("ECa", 40.0, 160.0, 1.0, "ECa", C_V),
    ("EK", -95.0, -60.0, 0.5, "EK", C_V),
    ("V1", -50.0, 30.0, 0.5, "V1  (m_inf mid)", C_V),
    ("V2", 3.0, 40.0, 0.2, "V2  (m_inf width)", C_V),
    ("V3", -30.0, 45.0, 0.5, "V3  (w_inf mid)", C_W),
    ("V4", 5.0, 45.0, 0.2, "V4  (w_inf width)", C_W),
    ("PHI", 0.01, 0.8, 0.005, "phi (K+ speed)", "#8f9bff"),
    ("C", 5.0, 50.0, 0.5, "C", "#8f9bff"),
]


class App:
    def __init__(self):
        self.I = 36.0
        self.neuron = Neuron(self.I)
        self.ms_per_frame = 1.5
        self._field_dirty = False
        self.eqs = []
        self._build()
        self.load_regime("Homoclinic (default)", pulse=False)

    # ------------------------------------------------------------------ build
    def _build(self):
        plt.rcParams.update({
            "text.color": FG, "axes.labelcolor": FG,
            "xtick.color": FG, "ytick.color": FG,
            "axes.edgecolor": "#3a3f50", "font.size": 9
        })
        self.fig = plt.figure(figsize=(15.5, 9.0), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("Morris-Lecar bifurcation explorer")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.05, 0.29, 0.48, 0.67], facecolor=BG)
        ax = self.ax
        ax.set_xlim(VMIN, VMAX)
        ax.set_ylim(WMIN, WMAX)
        ax.set_xlabel("Volts V (mV)")
        ax.set_ylabel("Frac. of potassium channels  (w)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        gx = np.linspace(VMIN + 3, VMAX - 3, 36)
        gy = np.linspace(0.02, 0.98, 22)
        self.GX, self.GY = np.meshgrid(gx, gy)
        Uq, Vq = self._field_arrows(self.I)
        self.quiv = ax.quiver(self.GX, self.GY, Uq, Vq,
                              color="#9aa1b4", alpha=0.35, angles="uv", pivot="mid",
                              scale=60, scale_units="width", width=0.0016,
                              headwidth=4.5, headlength=5.5, zorder=2)

        rng = np.random.default_rng(7)
        gxs, gys = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 14))
        seeds = np.stack([gxs.ravel(), gys.ravel()], 1)
        seeds += (rng.random(seeds.shape) - 0.5) * np.array([0.04, 0.06])
        self.seeds = np.clip(seeds, 0, 1)
        self.flow = LineCollection([], linewidths=0.9, zorder=1)
        ax.add_collection(self.flow)

        (self.wnull,) = ax.plot([], [], color=C_W, lw=1.8, zorder=4, label="w-nullcline")
        self.wlab = ax.text(-96, 0.05, "w-nullcline", color=C_W, fontsize=13, family="serif",
                            ha="left", va="bottom", zorder=6)

        self.vglow = [ax.plot([], [], color=C_V, lw=lw, alpha=a, zorder=3, solid_capstyle="round")[0]
                      for lw, a in ((11, 0.07), (6, 0.14))]
        (self.vnull,) = ax.plot([], [], color=C_V, lw=2.4, zorder=4, label="V-nullcline")
        (self.ghost,) = ax.plot([], [], color="#ff9ab3", lw=1.3, ls=(0, (4, 3)), zorder=4)
        self.vlab = ax.text(0, 0, "V-nullcline", color=C_V, fontsize=14, family="serif",
                            ha="center", va="bottom", zorder=6)

        self.sep = [ax.plot([], [], color=C_SEP, lw=1.7, ls=(0, (6, 3)), zorder=5)[0] for _ in range(2)]
        self.sep[0].set_label("separatrix")
        self.eq_stable, = ax.plot([], [], "o", ms=9, mfc="white", mec="white", zorder=7, label="stable")
        self.eq_saddle, = ax.plot([], [], "D", ms=8, mfc=C_SEP, mec=C_SEP, zorder=7, label="saddle")
        self.eq_unst, = ax.plot([], [], "o", ms=9, mfc=BG, mec=C_V, mew=2, zorder=7, label="unstable")

        (self.trace,) = ax.plot([], [], color="#7fd6ff", lw=1.2, alpha=0.75, zorder=8)
        self.comet = LineCollection([], linewidths=3.0, zorder=9, capstyle="round")
        ax.add_collection(self.comet)
        (self.glow_pt,) = ax.plot([], [], "o", ms=22, color="#ffffff", alpha=0.18, zorder=9)
        (self.pt,) = ax.plot([], [], "o", ms=9, color="#ffffff", mec="#7fd6ff", mew=1.5, zorder=10)
        self.status = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
                              fontsize=10, family="monospace", zorder=11,
                              bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec="#2a2f40", alpha=0.85))

        leg = ax.legend(loc="upper right", frameon=True, facecolor=BG, edgecolor="#2a2f40",
                        labelcolor=FG, fontsize=8)
        leg.set_zorder(12)

        # ---- V(t) panel
        self.axv = self.fig.add_axes([0.68, 0.835, 0.30, 0.12], facecolor=BG)
        self.axv.set_xlim(-180, 0)
        self.axv.set_ylim(VMIN, VMAX)
        self.axv.axhline(0, color="#2a2f40", lw=0.8, ls="--")
        self.axv.set_title("V(t)  - last 180 ms", fontsize=9)
        self.axv.set_xlabel("ms (0 = now)", fontsize=8)
        self.axv.set_ylabel("mV", fontsize=8)
        for s in ("top", "right"):
            self.axv.spines[s].set_visible(False)
        (self.vline,) = self.axv.plot([], [], color="#7fd6ff", lw=1.4)

        # ---- equilibria-vs-I panel
        self.axb = self.fig.add_axes([0.68, 0.605, 0.30, 0.135], facecolor=BG)
        axb = self.axb
        axb.set_ylim(VMIN, VMAX)
        axb.set_title("Equilibria vs I   (click: set I, scroll: zoom)", fontsize=9)
        axb.set_xlabel("I (uA/cm2)", fontsize=8)
        axb.set_ylabel("V*  (mV)", fontsize=8)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        self.bif_lines = [
            axb.plot([], [], ".", ms=2.5, color="white", label="stable")[0],
            axb.plot([], [], ".", ms=2.5, color=C_SEP, label="saddle")[0],
            axb.plot([], [], ".", ms=2.5, color=C_V, label="unstable")[0],
        ]
        (self.cyc_dots,) = axb.plot([], [], ".", ms=3.5, color="#ff8c1a", label="limit cycle (min/max)")
        axb.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor=BG,
                   edgecolor="#2a2f40", labelcolor=FG, markerscale=2)
        self.bif_I = axb.axvline(self.I, color="#7fd6ff", lw=1.2)

        self.hist_t = deque(maxlen=450)
        self.hist_V = deque(maxlen=450)
        self.trail = deque(maxlen=55)
        self.trace_V = deque(maxlen=2800)
        self.trace_w = deque(maxlen=2800)

        # ---- stimulus sliders (bottom left)
        def mk_slider(rect, label, lo, hi, init, step, color, fmt=None, fs=9):
            a = self.fig.add_axes(rect, facecolor=PANEL)
            s = Slider(a, label, lo, hi, valinit=init, valstep=step, color=color,
                       **({"valfmt": fmt} if fmt else {}))
            try:
                s.track.set_facecolor('#232736')
            except Exception:
                pass
            s.label.set_color(FG)
            s.label.set_fontsize(fs)
            s.valtext.set_color(FG)
            s.valtext.set_fontsize(fs)
            return s

        self.s_I = mk_slider([0.20, 0.215, 0.30, 0.025], "Input current I  ", 0, 45, 36.0, 0.05, C_V, "%.2f")
        self.s_amp = mk_slider([0.20, 0.170, 0.30, 0.025], "Pulse amplitude  ", -150, 300, 125.0, 1.0, "#6c8cff")
        self.s_dur = mk_slider([0.20, 0.125, 0.30, 0.025], "Pulse duration (ms)  ", 0.5, 30, 3.0, 0.5, "#6c8cff")
        self.s_spd = mk_slider([0.20, 0.080, 0.30, 0.025], "Speed (ms/frame)  ", 0.2, 8.0, self.ms_per_frame, 0.1, "#888ea3")
        self.s_I.on_changed(self._on_I)
        self.s_spd.on_changed(lambda v: setattr(self, "ms_per_frame", v))

        # ---- parameter sliders (right column)
        self.fig.text(0.68, 0.535, "Model parameters  (V-nullcline red, w-nullcline teal)",
                      color=FG, fontsize=9)
        self.s_par = {}
        y0, dy = 0.503, 0.0235
        for k, (name, lo, hi, step, label, col) in enumerate(PARAM_SPECS):
            s = mk_slider([0.77, y0 - k * dy, 0.19, 0.016], label + "  ", lo, hi,
                          DEFAULTS[name], step, col, "%.3g", fs=8)
            s.on_changed(lambda v, n=name: self._on_param(n, v))
            self.s_par[name] = s

        # ---- buttons
        def mk_btn(rect, label, cb, color="#1d2030", fs=9):
            a = self.fig.add_axes(rect)
            b = Button(a, label, color=color, hovercolor="#2c3250")
            b.label.set_color(FG)
            b.label.set_fontsize(fs)
            b.on_clicked(cb)
            return b

        self.b_pulse = mk_btn([0.05, 0.020, 0.11, 0.04], "PULSE (space)", lambda e: self.pulse(), "#3a1a28")
        self.b_reset = mk_btn([0.17, 0.020, 0.09, 0.04], "Reset (R)", lambda e: self.reset())
        self.b_clear = mk_btn([0.27, 0.020, 0.09, 0.04], "Clear trace", lambda e: self.clear_trace())
        self.b_scan = mk_btn([0.37, 0.020, 0.14, 0.04], "Scan limit cycles", lambda e: self.scan(), "#2a2210")
        self.b_defp = mk_btn([0.565, 0.020, 0.10, 0.04], "Default params", lambda e: self.load_regime("Homoclinic (default)", pulse=False))

        self.fig.text(0.565, 0.262, "Pulse demos\n(default params, I=36)", color=FG, fontsize=7.5)
        self.b_demo = []
        for i, name in enumerate(PULSE_DEMOS):
            self.b_demo.append(mk_btn([0.565, 0.200 - i * 0.045, 0.10, 0.036], name,
                                      lambda e, n=name: self.demo(n), fs=8))
        self.fig.text(0.565, 0.105,
                      "Click the phase plane\nto place the neuron.",
                      color="#8a8fa3", fontsize=7, va="top")

        self.regime_btns = []
        self.fig.text(0.68, 0.215, "Regimes  (load parameters + I)", color=FG, fontsize=9)
        self._regime_btn_specs = []

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._mk_btn = mk_btn
        self.build_regime_buttons()
        for art in self.animated():
            art.set_animated(True)

    def build_regime_buttons(self):
        for b in self.regime_btns:
            b.ax.remove()
        self.regime_btns = []
        names = [n for n in REGIMES]
        cols, rows = 2, 6
        w, h = 0.145, 0.026
        for i, name in enumerate(names[:cols * rows]):
            r, c = divmod(i, cols)
            rect = [0.68 + c * (w + 0.01), 0.18 - r * (h + 0.006), w, h]
            self.regime_btns.append(
                self._mk_btn(rect, name, lambda e, n=name: self.load_regime(n), fs=6.5))

    def animated(self):
        return [
            self.vnull, self.vglow[0], self.vglow[1], self.vlab, self.wnull,
            self.sep[0], self.sep[1],
            self.eq_stable, self.eq_saddle, self.eq_unst,
            self.ghost,
            self.trace, self.comet, self.glow_pt, self.pt,
            self.vline, self.status, self.bif_I,
        ]

    # --------------------------------------------------------------- geometry
    def _field_arrows(self, I):
        u, v = rhs_vec(self.GX, self.GY, I)
        u, v = u / DV, v / DW
        asp = 0.48 * 15.5 / (0.67 * 9.0)
        U, Vv = u * asp, v
        s = np.hypot(U, Vv) + 1e-12
        return U / s, Vv / s

    def _field_streamlines(self, I):
        P, sp = streamlines(I, self.seeds)
        seg = np.stack([P[:, :-1], P[:, 1:]], 2).reshape(-1, 2, 2)
        spd = sp[:, :-1].ravel()
        ok = np.isfinite(seg).all(axis=(1, 2))
        seg, spd = seg[ok], spd[ok]
        a = np.clip(0.03 + 0.30 * (np.log10(spd + 1e-4) + 3.0) / 3.0, 0.02, 0.32)
        cols = np.empty((len(a), 4))
        cols[:] = (0.62, 0.66, 0.78, 1.0)
        cols[:, 3] = a
        return seg, cols

    def update_vector_field(self):
        U, V = self._field_arrows(self.I)
        self.quiv.set_UVC(U, V)
        seg, cols = self._field_streamlines(self.I)
        self.flow.set_segments(seg)
        self.flow.set_color(cols)

    def update_nullclines_and_equilibria(self):
        I = self.I
        Vs = np.linspace(EK + 0.35, VMAX, 500)
        wv = v_null(Vs, I)
        for ln in self.vglow + [self.vnull]:
            ln.set_data(Vs, wv)
        Vw = np.linspace(VMIN, VMAX, 400)
        self.wnull.set_data(Vw, w_inf(Vw))

        yv = float(v_null(28.0, I))
        self.vlab.set_position((28.0, min(max(yv, 0.05), 0.88) + 0.04))

        eqs = equilibria(I)
        self.eqs = eqs
        for kind, art in (("stable", self.eq_stable), ("saddle", self.eq_saddle), ("unstable", self.eq_unst)):
            pts = [(e["V"], e["w"]) for e in eqs if e["kind"] == kind]
            art.set_data([p[0] for p in pts], [p[1] for p in pts])

        saddles = [e for e in eqs if e["kind"] == "saddle"]
        unst = [e for e in eqs if e["kind"] == "unstable"]
        if saddles:
            br = separatrix(saddles[0], unst[0] if unst else None)
            for ln, b in zip(self.sep, br):
                ln.set_data(b[:, 0], b[:, 1])
        else:
            for ln in self.sep:
                ln.set_data([], [])

    def update_bif(self):
        lo, hi = self.s_I.valmin, self.s_I.valmax
        Ie, Ve, kd = bif_diagram(lo, hi)
        for k, ln in enumerate(self.bif_lines):
            mk = kd == k
            ln.set_data(Ie[mk], Ve[mk])
        self.cyc_dots.set_data([], [])
        self.axb.set_xlim(lo, hi)

    # --------------------------------------------------------------- handlers
    def _on_I(self, val):
        self.I = float(val)
        self.update_nullclines_and_equilibria()
        self._field_dirty = True

    def _on_param(self, name, val):
        set_params(**{name: float(val)})
        self._params_changed()

    def _params_changed(self):
        self.update_nullclines_and_equilibria()
        self.update_bif()
        self._field_dirty = True
        self.ghost.set_data([], [])

    def set_I_range(self, lo, hi, I=None):
        s = self.s_I
        s.valmin, s.valmax = lo, hi
        s.valstep = (hi - lo) / 1500.0
        s.ax.set_xlim(lo, hi)
        try:
            s._fmt.set_locs((lo, hi))
        except Exception:
            pass
        I = self.I if I is None else I
        I = min(max(I, lo), hi)
        s.eventson = False
        s.set_val(I)
        s.eventson = True
        self.I = float(s.val)

    def load_regime(self, name, pulse=False):
        R = REGIMES[name]
        params = {**DEFAULTS, **R["p"]}
        set_params(**params)
        for pn, s in self.s_par.items():
            s.eventson = False
            s.set_val(params[pn])
            s.eventson = True
        self.set_I_range(*R["Irange"], I=R["I"])
        self.update_nullclines_and_equilibria()
        self.update_bif()
        self._field_dirty = True
        self.reset()

    def demo(self, name):
        self.load_regime("Homoclinic (default)")
        amp, dur = PULSE_DEMOS[name]
        self.s_amp.set_val(amp)
        self.s_dur.set_val(dur)
        self.pulse()

    def scan(self):
        lo, hi = self.s_I.valmin, self.s_I.valmax
        self.status.set_text("scanning limit cycles ...")
        Ic, vmin, vmax = scan_cycles(lo, hi)
        self.cyc_dots.set_data(np.concatenate([Ic, Ic]), np.concatenate([vmin, vmax]))
        self._field_dirty = True      # forces a full redraw (panel is not animated)

    def pulse(self):
        amp, dur = self.s_amp.val, self.s_dur.val
        self.neuron.kick(amp, dur)
        self.clear_trace()
        Vs = np.linspace(EK + 0.35, VMAX, 400)
        self.ghost.set_data(Vs, v_null(Vs, self.I + amp))

    def reset(self):
        self.neuron.reset(self.I)
        self.neuron.V, self.neuron.w = rest_point(self.I, self.eqs)
        self.ghost.set_data([], [])
        self.clear_trace()

    def clear_trace(self):
        self.trace_V.clear()
        self.trace_w.clear()
        self.trail.clear()
        self.hist_t.clear()
        self.hist_V.clear()

    def _toolbar_active(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "")

    def _on_click(self, ev):
        if ev.button != 1 or self._toolbar_active():
            return
        if ev.inaxes is self.ax:
            self.neuron.place(ev.xdata, ev.ydata)
            self.ghost.set_data([], [])
            self.clear_trace()
        elif ev.inaxes is self.axb and ev.xdata is not None:
            self.s_I.set_val(ev.xdata)

    def _on_scroll(self, ev):
        if ev.inaxes is not self.axb or ev.xdata is None:
            return
        f = 0.8 if ev.step > 0 else 1.25
        lo, hi = self.s_I.valmin, self.s_I.valmax
        c = ev.xdata
        self.set_I_range(c - (c - lo) * f, c + (hi - c) * f)
        self.update_bif()
        self._field_dirty = True

    def _on_key(self, ev):
        if ev.key == " ":
            self.pulse()
        elif ev.key in ("r", "R"):
            self.reset()
        elif ev.key in ("left", "right", "up", "down"):
            d = {"left": -0.1, "right": 0.1, "up": 1.0, "down": -1.0}[ev.key]
            self.s_I.set_val(min(max(self.I + d, self.s_I.valmin), self.s_I.valmax))

    # ------------------------------------------------------------- animation
    def advance(self, ms):
        n = max(1, int(round(ms / Neuron.DT)))
        nr = self.neuron
        for i in range(n):
            nr.step(self.I)
            if i % 4 == 0:
                self.trail.append((nr.V, nr.w))
                self.trace_V.append(nr.V)
                self.trace_w.append(nr.w)
                self.hist_t.append(nr.t)
                self.hist_V.append(nr.V)
            if abs(nr.V) > 300 or abs(nr.w) > 5:
                nr.reset(self.I)
                nr.V, nr.w = rest_point(self.I, self.eqs)
                break

    def regime_text(self):
        n = {k: sum(e["kind"] == k for e in self.eqs) for k in ("stable", "saddle", "unstable")}
        if n["stable"] >= 2:
            lab = "BISTABLE (2 stable equilibria)"
        elif n["stable"] == 1 and n["saddle"] == 0 and n["unstable"] == 0:
            lab = "one stable equilibrium (Scan: cycles?)"
        elif n["stable"] == 0 and n["unstable"] >= 1:
            lab = "no stable rest -> oscillates"
        elif n["stable"] == 1 and n["unstable"] >= 1:
            lab = "rest + unstable focus (cycle likely)"
        elif n["stable"] == 1:
            lab = "rest + saddle (excitable threshold)"
        else:
            lab = "no stable equilibrium"
        return n, lab

    def outcome_text(self):
        nr = self.neuron
        if nr.pulsing:
            return "pulse ON"
        if nr.spikes == 0:
            return "no spike"
        rest = rest_point(self.I, self.eqs)
        near = math.hypot((nr.V - rest[0]) / DV, (nr.w - rest[1]) / DW) < 0.04
        if nr.spikes >= 2 and nr.t - nr.last_spike_t < 150:
            return "CONTINUOUS FIRING"
        if near and nr.spikes == 1:
            return "fired once, back at rest"
        return "spiking..."

    def draw_frame(self, _=None):
        if self._field_dirty:
            self._field_dirty = False
            self.update_vector_field()
            self.fig.canvas.draw()
            anim = getattr(self, "anim", None)
            try:
                anim._blit_cache.clear()
            except Exception:
                pass
        self.advance(self.ms_per_frame)
        nr = self.neuron
        if not nr.pulsing and len(self.ghost.get_xdata()) > 0:
            self.ghost.set_data([], [])

        self.trace.set_data(self.trace_V, self.trace_w)

        tl = np.array(self.trail)
        if len(tl) > 2:
            seg = np.stack([tl[:-1], tl[1:]], 1)
            k = len(seg)
            cols = np.zeros((k, 4))
            cols[:] = (1.0, 0.93, 0.45, 1.0)
            cols[:, 3] = np.linspace(0.0, 1.0, k) ** 1.5
            self.comet.set_segments(seg)
            self.comet.set_color(cols)

        self.pt.set_data([nr.V], [nr.w])
        self.glow_pt.set_data([nr.V], [nr.w])
        self.bif_I.set_xdata([self.I, self.I])

        if self.hist_t:
            cur_t = nr.t
            self.vline.set_data([t - cur_t for t in self.hist_t], self.hist_V)

        n, lab = self.regime_text()
        self.status.set_text(
            f"I = {self.I:6.2f}   t = {nr.t:7.1f} ms\n"
            f"V = {nr.V:6.1f} mV  w = {nr.w:4.2f}\n"
            f"eq: {n['stable']} stable, {n['saddle']} saddle, {n['unstable']} unstable\n"
            f"=> {lab}\n"
            f"spikes: {nr.spikes}  ->  {self.outcome_text()}"
        )
        return self.animated()

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.draw_frame, init_func=self.animated,
            frames=itertools.count(), interval=20,
            blit=True, cache_frame_data=False
        )
        plt.show()


def main():
    App().run()


if __name__ == "__main__":
    main()