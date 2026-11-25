"""
Interactive FitzHugh-Nagumo grid memory model  (pure Python: numpy + matplotlib)

Run:    python3 fhn_memory_interactive.py
Needs:  pip install numpy matplotlib   (and a GUI backend, e.g. Tk or Qt)

What happens on every run
  1. CREB (circadian + ultradian, random phase per neuron) sets each neuron's resting drive.
  2. A stimulus drives the grid. Neurons whose drive crosses the FHN Hopf point fire -> sparse engram.
  3. If the SWR gate overlaps that firing, STDP (scaled by CREB competence) updates W_ij.
  4. A partial cue is then given. Learned W_ij shifts each partner's V-nullcline across the Hopf
     point, so the rest of the engram fires (pattern completion).

How to play
  * Move a slider  -> the whole experiment re-runs (auto-run), or press "Run".
  * CLICK any neuron on the top-row maps -> its conditional phase plane + voltage traces update.
  * "New cue mask" picks a different random part of the stimulus as the cue.
  * "New world" re-draws CREB phases and the stimulus pattern.
  * Check boxes: CREB gating of plasticity, and a naive-network control (no learning).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ------------------------------------------------------------------ fixed model constants
G = 12
M = G * G
a, b, eps = 0.7, 0.8, 0.08            # FitzHugh-Nagumo
dt, tau_s, tau_stdp = 0.1, 5.0, 10.0  # integration step, synaptic decay, STDP trace decay
I0 = 0.10                             # baseline resting drive (CREB adds kc * c)
T_PRE, T_STIM = 20.0, 120.0           # stimulus onset / duration (model time units)
FIRE_MIN = 2                          # spikes needed to call a neuron "firing"

xy = np.array([(i // G, i % G) for i in range(M)], float)
D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
W0 = 0.01 * np.exp(-D ** 2 / (2 * 1.2 ** 2))          # weak local baseline coupling
np.fill_diagonal(W0, 0)
KERN = np.exp(-D ** 2 / (2 * 0.7 ** 2))

DEFAULTS = dict(t_enc=10.0, ret_dt=0.0, kc=0.22, s0=0.28, swr_off=15.0, swr_w=60.0,
                eta=0.12, ratio=0.5, wmax=0.30, cue_f=0.40, s_on=0.5)


def fixed_point(I):
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(50):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        v = v - f / (v ** 2 + (1 / b - 1))
    return v, (v + a) / b


def stable(v):
    return (1 - v ** 2 - eps * b < 0) and (eps * (1 - b * (1 - v ** 2)) > 0)


vH = -np.sqrt(1 - eps * b)
I_HOPF = (vH + a) / b - vH + vH ** 3 / 3             # drive at the Hopf bifurcation (~0.331)


class World:
    """Everything random: per-neuron CREB phases, the stimulus pattern, the cue mask."""

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.phi = r.uniform(0, 2 * np.pi, M)
        self.psi = r.uniform(0, 2 * np.pi, M)
        f = KERN @ r.standard_normal(M)
        u = (np.argsort(np.argsort(f)) + 0.5) / M
        self.p = u ** 4
        self.u_cue = r.random(M)

    def creb(self, t_h):
        return np.clip(0.5 + 0.3 * np.sin(2 * np.pi * t_h / 24 + self.phi)
                       + 0.2 * np.sin(2 * np.pi * t_h / 1.5 + self.psi), 0, 1)


def simulate(W, I_rest, I_ext, P, swr_start=None, learn=False, sigma=None, t_end=None):
    """Euler integration of M coupled FHN neurons.  Returns spike counts, final W, voltages."""
    W = W.copy()
    t_on, t_off = T_PRE, T_PRE + T_STIM
    n = int(t_end / dt)
    v, w = fixed_point(I_rest)
    s = np.zeros(M); x = np.zeros(M); y = np.zeros(M)
    counts = np.zeros(M, int)
    V = np.empty((n, M), np.float32)
    swr_end = None if swr_start is None else swr_start + P["swr_w"]
    eta_p, eta_d, wmax = P["eta"], P["eta"] * P["ratio"], P["wmax"]
    dec_s, dec_t = np.exp(-dt / tau_s), np.exp(-dt / tau_stdp)
    for k in range(n):
        t = k * dt
        I = I_rest + W @ s
        if t_on <= t < t_off:
            I = I + I_ext
        vn = v + dt * (v - v ** 3 / 3 - w + I)
        wn = w + dt * eps * (v + a - b * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        V[k] = v
        if t_on <= t < t_off + 40:
            counts += spk
        if learn and swr_start is not None and swr_start <= t < swr_end and spk.any():
            sp = spk.astype(float)
            W += eta_p * np.outer(sigma * sp, x)       # pre-before-post  -> LTP onto spiking post
            W -= eta_d * np.outer(sigma * y, sp)       # post-before-pre  -> LTD from spiking pre
            np.clip(W, 0, wmax, out=W)
            np.fill_diagonal(W, 0)
        s = s * dec_s + spk
        x = x * dec_t + spk
        y = y * dec_t + spk
    return counts, W, V


def compute(world, P, flags):
    c_enc = world.creb(P["t_enc"])
    I_enc = I0 + P["kc"] * c_enc
    sigma = c_enc if flags["gate"] else np.ones(M)
    swr_start = T_PRE + P["swr_off"]
    t_end = max(T_PRE + T_STIM + 40, swr_start + P["swr_w"] + 5)
    cnt, W, Venc = simulate(W0, I_enc, P["s0"] * world.p, P, swr_start, True, sigma, t_end)
    eng = cnt >= FIRE_MIN

    mask = (world.u_cue < P["cue_f"]).astype(float)
    c_ret = world.creb(P["t_enc"] + P["ret_dt"])
    I_ret = I0 + P["kc"] * c_ret
    Wr = W0 if flags["naive"] else W
    cnt_r, _, Vret = simulate(Wr, I_ret, P["s0"] * world.p * mask, P, None, False, None, T_PRE + T_STIM + 40)
    fired = cnt_r >= FIRE_MIN

    unc = mask == 0
    e_unc = eng & unc
    return dict(c_enc=c_enc, c_ret=c_ret, I_enc=I_enc, I_ret=I_ret, W=W, Wr=Wr, cnt=cnt, eng=eng, mask=mask,
                fired=fired, Venc=Venc, Vret=Vret, swr_start=swr_start,
                recall=fired[e_unc].mean() if e_unc.any() else np.nan,
                fp=fired[~eng & unc].mean() if (~eng & unc).any() else np.nan,
                n_unc_eng=int(e_unc.sum()), spont=int((I_ret > I_HOPF).sum()),
                dW=W[np.ix_(eng, eng)].mean() - W0[np.ix_(eng, eng)].mean() if eng.any() else np.nan)


def single_neuron(I_rest_i, I_in, T=300.0):
    v, w = fixed_point(I_rest_i)
    v, w = v[0], w[0]
    tr = []
    for _ in range(int(T / dt)):
        tr.append((v, w))
        vn = v + dt * (v - v ** 3 / 3 - w + I_rest_i + I_in)
        w = w + dt * eps * (v + a - b * w)
        v = vn
    return np.array(tr)


CMAP = ListedColormap(["#eeeeee", "#f2cf5b", "#4c78a8", "#54a24b", "#e45756"])
LEGEND = [Patch(color="#eeeeee", label="silent"), Patch(color="#f2cf5b", label="engram, missed"),
          Patch(color="#4c78a8", label="engram, cued"), Patch(color="#54a24b", label="engram, recalled via W"),
          Patch(color="#e45756", label="false positive")]


class App:
    def __init__(self, seed=7):
        self.seed = seed
        self.world = World(seed)
        self.P = dict(DEFAULTS)
        self.flags = dict(auto=True, gate=True, naive=False)
        self.sel = None
        self.res = None
        self.muted = False
        self.stale = False
        self.sliders = {}
        self.build()
        self.recompute()

    # ------------------------------------------------------------------ UI construction
    def build(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title("FHN grid memory - interactive") if hasattr(
            self.fig.canvas, "manager") and self.fig.canvas.manager else None
        gs = self.fig.add_gridspec(2, 4, left=0.05, right=0.98, top=0.89, bottom=0.36, hspace=0.55, wspace=0.28)
        self.ax_creb = self.fig.add_subplot(gs[0, 0])
        self.ax_stim = self.fig.add_subplot(gs[0, 1])
        self.ax_out = self.fig.add_subplot(gs[0, 2])
        self.ax_txt = self.fig.add_subplot(gs[0, 3])
        self.ax_W = self.fig.add_subplot(gs[1, 0])
        self.ax_ph = self.fig.add_subplot(gs[1, 1:3])
        self.ax_tr = self.fig.add_subplot(gs[1, 3])
        self.fig.suptitle("FitzHugh-Nagumo grid memory: CREB-gated encoding, SWR-gated STDP, conditional phase space"
                          "     (click any neuron on the top-row maps)", fontsize=11)

        spec_l = [("Encoding time of day (h)", "t_enc", 0, 24, 0.5), ("Retrieval delay (h)", "ret_dt", 0, 24, 0.5),
                  ("CREB excitability gain kc", "kc", 0.0, 0.30, 0.005), ("Stimulus strength s0", "s0", 0.10, 0.50, 0.01),
                  ("SWR onset vs stimulus", "swr_off", -150, 250, 5), ("SWR width", "swr_w", 10, 200, 5)]
        spec_r = [("STDP LTP rate", "eta", 0.01, 0.40, 0.01), ("LTD / LTP ratio", "ratio", 0.0, 1.5, 0.05),
                  ("Weight cap w_max", "wmax", 0.05, 0.60, 0.01), ("Cue fraction of stimulus", "cue_f", 0.05, 0.95, 0.05),
                  ("Presyn. activity s (phase plane)", "s_on", 0.1, 1.0, 0.05)]
        for row, (lab, key, lo, hi, st) in enumerate(spec_l):
            self.add_slider(lab, key, lo, hi, st, 0.16, 0.285 - row * 0.043)
        for row, (lab, key, lo, hi, st) in enumerate(spec_r):
            self.add_slider(lab, key, lo, hi, st, 0.55, 0.285 - row * 0.043)

        def btn(rect, text, cb):
            axb = self.fig.add_axes(rect)
            bt = Button(axb, text)
            bt.label.set_fontsize(9)
            bt.on_clicked(cb)
            return bt
        self.b_run = btn([0.80, 0.265, 0.085, 0.035], "Run", lambda e: self.recompute())
        self.b_cue = btn([0.895, 0.265, 0.085, 0.035], "New cue mask", self.new_mask)
        self.b_world = btn([0.80, 0.22, 0.085, 0.035], "New world", self.new_world)
        self.b_reset = btn([0.895, 0.22, 0.085, 0.035], "Reset sliders", self.reset)
        axc = self.fig.add_axes([0.80, 0.09, 0.18, 0.11])
        self.chk = CheckButtons(axc, ["auto-run on change", "CREB gates plasticity", "retrieve from naive net\n(control: no learning)"],
                                [True, True, False])
        for t in self.chk.labels:
            t.set_fontsize(8.5)
        self.chk.on_clicked(self.on_flag)
        self.status_txt = self.fig.text(0.80, 0.045, "", fontsize=8.5, color="#444")
        self.timer = self.fig.canvas.new_timer(interval=300)
        self.timer.single_shot = True
        self.timer.add_callback(self.recompute)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def add_slider(self, label, key, lo, hi, step, x0, y):
        ax = self.fig.add_axes([x0, y, 0.19, 0.022])
        s = Slider(ax, label, lo, hi, valinit=self.P[key], valstep=step)
        s.label.set_fontsize(8.5)
        s.valtext.set_fontsize(8.5)
        s.on_changed(lambda v, k=key: self.on_param(k, v))
        self.sliders[key] = s

    # ------------------------------------------------------------------ callbacks
    def on_param(self, key, v):
        self.P[key] = float(v)
        if self.muted:
            return
        if key == "s_on":
            self.draw_phase()
            self.fig.canvas.draw_idle()
            return
        if self.flags["auto"]:
            self.timer.stop()
            self.timer.start()
        else:
            self.stale = True
            self.draw_text()
            self.fig.canvas.draw_idle()

    def on_flag(self, label):
        st = self.chk.get_status()
        old = dict(self.flags)
        self.flags.update(auto=st[0], gate=st[1], naive=st[2])
        if (old["gate"], old["naive"]) != (self.flags["gate"], self.flags["naive"]) or \
                (self.flags["auto"] and not old["auto"] and self.stale):
            if self.flags["auto"] or True:
                self.recompute()

    def new_mask(self, _=None):
        self.world.u_cue = np.random.default_rng().random(M)
        self.recompute()

    def new_world(self, _=None):
        self.seed += 1
        self.world = World(self.seed)
        self.sel = None
        self.recompute()

    def reset(self, _=None):
        self.muted = True
        for k, s in self.sliders.items():
            s.set_val(DEFAULTS[k])
        self.muted = False
        self.P = dict(DEFAULTS)
        self.recompute()

    def on_click(self, ev):
        if ev.inaxes in (self.ax_creb, self.ax_stim, self.ax_out) and ev.xdata is not None:
            c = int(np.clip(round(ev.xdata), 0, G - 1))
            r = int(np.clip(round(ev.ydata), 0, G - 1))
            self.select(r * G + c)

    def select(self, i):
        self.sel = int(i)
        self.draw_maps(); self.draw_phase(); self.draw_trace(); self.draw_text()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ compute + draw
    def recompute(self, *_):
        t0 = time.time()
        self.stale = False
        self.res = compute(self.world, self.P, self.flags)
        R = self.res
        if self.sel is None:
            pick = np.where(R["eng"] & (R["mask"] == 0) & R["fired"])[0]
            # prefer a neuron that one memory partner alone can push across the Hopf point
            s_on = self.P["s_on"]
            flip = [i for i in pick
                    if R["I_ret"][i] + W0[i].max() * s_on < I_HOPF < R["I_ret"][i] + R["Wr"][i].max() * s_on]
            if len(flip):
                pick = np.array(flip)
            elif not len(pick):
                pick = np.where(R["eng"])[0]
            self.sel = int(pick[len(pick) // 2]) if len(pick) else 0
        self.dur = time.time() - t0
        self.draw_all()

    def draw_all(self):
        self.draw_maps(); self.draw_W(); self.draw_text(); self.draw_phase(); self.draw_trace()
        self.status_txt.set_text(f"last run {self.dur:.2f}s   seed {self.seed}")
        self.fig.canvas.draw_idle()

    def _ring(self, ax, color):
        i = self.sel
        ax.scatter([i % G], [i // G], s=260, facecolors="none", edgecolors=color, linewidths=2.2, zorder=5)

    def draw_maps(self):
        R, P = self.res, self.P
        g = lambda z: np.asarray(z).reshape(G, G)
        ax = self.ax_creb; ax.clear()
        ax.imshow(g(R["c_enc"]), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"CREB at encoding ({P['t_enc']:.1f} h)\nbright = excitable & plastic", fontsize=9)
        self._ring(ax, "white")
        ax = self.ax_stim; ax.clear()
        ax.imshow(g(P["s0"] * self.world.p), cmap="magma", vmin=0, vmax=0.5)
        ey, ex = np.where(g(R["eng"]))
        ax.scatter(ex, ey, s=150, marker="s", facecolors="none", edgecolors="#00e5ff", linewidths=1.4)
        ax.set_title(f"Stimulus drive; engram outlined\n{R['eng'].sum()}/{M} fire ({100*R['eng'].mean():.0f}%)", fontsize=9)
        self._ring(ax, "white")
        ax = self.ax_out; ax.clear()
        o = np.zeros(M, int)
        eng, fired, mk = R["eng"], R["fired"], R["mask"]
        o[eng & ~fired] = 1; o[eng & fired & (mk == 1)] = 2; o[eng & fired & (mk == 0)] = 3; o[~eng & fired] = 4
        ax.imshow(g(o), cmap=CMAP, vmin=0, vmax=4)
        src = "naive net" if self.flags["naive"] else "trained net"
        ax.set_title(f"Retrieval ({src}), cue = {int(mk.sum())} cells", fontsize=9)
        ax.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=6.5, frameon=False)
        self._ring(ax, "black")
        for ax in (self.ax_creb, self.ax_stim, self.ax_out):
            ax.set_xticks([]); ax.set_yticks([])

    def draw_W(self):
        R = self.res
        ax = self.ax_W; ax.clear()
        order = np.r_[np.where(R["eng"])[0], np.where(~R["eng"])[0]]
        ax.imshow(R["W"][np.ix_(order, order)], cmap="Greens", vmin=0, vmax=max(self.P["wmax"], 1e-3))
        ne = int(R["eng"].sum())
        ax.axhline(ne - 0.5, color="#444", lw=0.6, ls="--"); ax.axvline(ne - 0.5, color="#444", lw=0.6, ls="--")
        ax.set_title(f"Learned W_ij (engram first)\ncolour max = {self.P['wmax']:.2f}", fontsize=9)
        ax.set_xlabel("presynaptic j", fontsize=8); ax.set_ylabel("postsynaptic i", fontsize=8)
        ax.tick_params(labelsize=7)

    def draw_text(self):
        R = self.res
        ax = self.ax_txt; ax.clear(); ax.axis("off")
        pc = lambda x: "n/a" if np.isnan(x) else f"{100*x:.0f}%"
        lines = [f"ENGRAM        {R['eng'].sum()} / {M}",
                 f"UNCUED ENGRAM {R['n_unc_eng']}",
                 f"RECALL        {pc(R['recall'])}",
                 f"FALSE POS.    {pc(R['fp'])}",
                 f"mean dW(engram) {R['dW']:.3f}" if not np.isnan(R["dW"]) else "mean dW(engram) n/a",
                 f"max rest drive {R['I_ret'].max():.3f}",
                 f"Hopf threshold {I_HOPF:.3f}"]
        warn = []
        if R["spont"]:
            warn.append(f"! {R['spont']} neurons above Hopf at rest\n  (spontaneous firing)")
        if not R["eng"].any():
            warn.append("! no engram: raise s0 / kc / pick\n  another encoding hour")
        if self.stale:
            warn.append("! sliders changed: press Run")
        ax.text(0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
        if warn:
            ax.text(0, 0.38, "\n".join(warn), va="top", color="#b00020", fontsize=8.5)
        i = self.sel
        ax.text(0, 0.12, f"selected neuron {i}  (row {i//G}, col {i%G})\n"
                         f"CREB@enc {R['c_enc'][i]:.2f}  rest drive {R['I_ret'][i]:.3f}",
                va="top", fontsize=8.5, color="#333")

    def draw_phase(self):
        R, s_on, i = self.res, self.P["s_on"], self.sel
        ax = self.ax_ph; ax.clear()
        Ir, W, eng = R["I_ret"][i], R["Wr"], R["eng"]
        j = int(np.argmax(W[i]))
        cand = np.where(~eng)[0]
        cand = cand[(cand != i) & (cand != j)]
        k = int(cand[np.argmin(D[i, cand])]) if len(cand) else int(np.argsort(W[i])[1])
        I_before, I_after, I_k = Ir + W0[i, j] * s_on, Ir + W[i, j] * s_on, Ir + W[i, k] * s_on
        vv = np.linspace(-2.6, 2.6, 400)
        ax.plot(vv, vv - vv ** 3 / 3 + Ir, "--", color="#999", label=f"no input (drive {Ir:.3f})")
        act = R["fired"].copy(); act[i] = False                      # partners that are firing at retrieval
        I_all = Ir + s_on * W[i, act].sum()
        items = [(I_before, "#9ecae1", f"input from j={j}, BEFORE learning ({I_before:.3f})", 1.8),
                 (I_after, "#1f77b4", f"input from j={j}, AFTER learning ({I_after:.3f})", 2.6),
                 (I_k, "#ff7f0e", f"input from non-memory k={k} ({I_k:.3f})", 1.8),
                 (I_all, "#9467bd", f"all {int(act.sum())} firing cells active ({I_all:.3f})", 1.8)]
        for Itot, col, lab, lw in items:
            ax.plot(vv, vv - vv ** 3 / 3 + Itot, color=col, lw=lw, label=lab)
            fv, fw = fixed_point(Itot)
            ax.plot(fv, fw, "o", ms=9, mfc=col if stable(fv[0]) else "w", mec=col, mew=2, zorder=6)
        ax.plot(vv, (vv + a) / b, color="#d62728", lw=1.8, label="n-nullcline")
        tr = single_neuron(Ir, I_after - Ir)
        ax.plot(tr[:, 0], tr[:, 1], color="k", lw=0.9, alpha=0.75, label="trajectory (after-learning input)")
        verdict = lambda I: "fires" if I > I_HOPF else "silent"
        ax.set_title(f"Conditional phase space of neuron {i}   (W_ij={W[i, j]:.3f}, W_ik={W[i, k]:.3f}, presyn s={s_on:.2f})\n"
                     f"before: {verdict(I_before)}   after: {verdict(I_after)}   non-memory: {verdict(I_k)}"
                     f"   all firing partners: {verdict(I_all)}\n"
                     f"   [Hopf drive {I_HOPF:.3f}; filled dot = stable, open = unstable]", fontsize=8.5)
        ax.set_xlabel("V", fontsize=9); ax.set_ylabel("n (recovery variable)", fontsize=9)
        ax.set_ylim(-1.2, 2.3); ax.set_xlim(-2.6, 2.6)
        ax.legend(fontsize=7, loc="upper left")

    def draw_trace(self):
        R, i, P = self.res, self.sel, self.P
        ax = self.ax_tr; ax.clear()
        tE = np.arange(len(R["Venc"])) * dt
        tR = np.arange(len(R["Vret"])) * dt
        ax.axvspan(T_PRE, T_PRE + T_STIM, color="#ffe9a8", alpha=0.6, lw=0)
        ax.axvspan(R["swr_start"], R["swr_start"] + P["swr_w"], ymin=0.5, ymax=1, color="#9ecae1", alpha=0.6, lw=0)
        ax.plot(tE, R["Venc"][:, i] + 0.0, color="#444", lw=0.9)
        ax.plot(tR, R["Vret"][:, i] - 4.6, color="#2a9d3f", lw=0.9)
        ax.text(0.02, 0.97, "encoding  (blue band = SWR gate)", transform=ax.transAxes, fontsize=7.5, va="top")
        ax.text(0.02, 0.45, "retrieval  (yellow band = cue)", transform=ax.transAxes, fontsize=7.5, va="top")
        ax.set_yticks([]); ax.set_xlim(0, max(tE[-1], tR[-1]))
        ax.set_xlabel("model time", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title(f"Voltage of neuron {i}", fontsize=9)


def main():
    app = App()
    plt.show()


if __name__ == "__main__":
    main()