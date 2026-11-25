"""
Emergent SWR, CREB-Allocated, and PV-Inhibited FitzHugh-Nagumo Engram Model
Minimal biophysical formulation for publication.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "font.family": "sans-serif", "font.size": 8.5
})

# Fixed biophysical constants
G = 12
M = G * G
a, b, eps = 0.7, 0.8, 0.08
dt, tau_s, tau_stdp = 0.1, 5.0, 12.0
tau_R = 12.0      # SWR envelope decay
tau_inh = 8.0     # PV+ Global Inhibition decay
I0 = 0.10
T_PRE, T_STIM, T_POST = 20.0, 90.0, 130.0
T_ENC_TOTAL = T_PRE + T_STIM + T_POST
T_RET_TOTAL = T_PRE + T_STIM + 40.0
FIRE_MIN = 2

xy = np.array([(i // G, i % G) for i in range(M)], float)
D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
W0 = 0.008 * np.exp(-D ** 2 / (2 * 1.2 ** 2))
np.fill_diagonal(W0, 0)
KERN = np.exp(-D ** 2 / (2 * 0.7 ** 2))

vH = -np.sqrt(1 - eps * b)
I_HOPF = (vH + a) / b - vH + vH ** 3 / 3

DEFAULTS = dict(
    t_enc=10.0, ret_dt=0.0, kc=0.22, s0=0.26,
    g_inh=4.5, p_swr=0.004, g_swr=0.08, # Inhibition, SWR Rate, SWR Drive
    eta=0.35, ratio=0.45, wmax=0.32, cue_f=0.35, s_on=0.55
)

def fixed_point(I):
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(40):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        v = v - f / (v ** 2 + (1 / b - 1))
    return v, (v + a) / b

def is_stable(v):
    return ((1 - v ** 2) - eps * b < 0) and (eps * (1 - b * (1 - v ** 2)) > 0)

class World:
    def __init__(self, seed):
        self.seed = seed
        r = np.random.default_rng(seed)
        self.phi = r.uniform(0, 2 * np.pi, M)
        self.psi = r.uniform(0, 2 * np.pi, M)
        f = KERN @ r.standard_normal(M)
        self.p = ((np.argsort(np.argsort(f)) + 0.5) / M) ** 4
        self.u_cue = r.random(M)
        self.swr_rng_seed = int(r.integers(1, 100000))

    def creb(self, t_h):
        return np.clip(0.5 + 0.3 * np.sin(2 * np.pi * t_h / 24 + self.phi)
                       + 0.2 * np.sin(2 * np.pi * t_h / 1.5 + self.psi), 0, 1)

def simulate(W_init, I_rest, I_ext, P, c_enc, learn=False, t_end=T_ENC_TOTAL, swr_seed=42):
    rng = np.random.default_rng(swr_seed)
    W = W_init.copy()
    n = int(t_end / dt)

    v, w = fixed_point(I_rest)
    s, x, y = np.zeros(M), np.zeros(M), np.zeros(M)
    R = 0.0
    I_inh = 0.0 # Global PV+ Inhibitory Feedback

    counts = np.zeros(M, int)
    V, R_trace = np.empty((n, M), np.float32), np.empty(n, np.float32)

    dec_s, dec_t, dec_R = np.exp(-dt / tau_s), np.exp(-dt / tau_stdp), np.exp(-dt / tau_R)
    
    for k in range(n):
        t = k * dt
        s_bar = s.mean()

        # Global Competitive Inhibition Update
        I_inh += dt * (-I_inh + P["g_inh"] * s_bar) / tau_inh

        # Poisson SWR Nucleation (Post-stimulus only)
        if learn and t > T_PRE + T_STIM:
            if rng.random() < P["p_swr"] * dt:
                R = 1.0 
        else:
            R = 0.0
        R_trace[k] = R

        # Total Current (Rest + Synaptic + Ext + Global SWR Drive - Global Inhibition)
        I = I_rest + W @ s - I_inh
        if T_PRE <= t < T_PRE + T_STIM:
            I += I_ext
        if learn and R > 0.01:
            I += P["g_swr"] * R 

        # FHN Integration
        vn = v + dt * (v - v ** 3 / 3 - w + I)
        wn = w + dt * eps * (v + a - b * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        V[k] = v

        if T_PRE <= t < T_PRE + T_STIM + 25.0:
            counts += spk

        # Emergent Burst-Threshold STDP (Strictly Local)
        if learn and spk.any():
            sp = spk.astype(float)
            theta_p = 1.2 # Calcium integration threshold for CaMKII activation
            
            x_burst = np.maximum(x - theta_p, 0.0)
            y_burst = np.maximum(y - theta_p, 0.0)
            
            ltp = np.outer(c_enc * sp, x_burst) * (P["wmax"] - W)
            ltd = np.outer(c_enc * y_burst, sp) * W
            
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            np.fill_diagonal(W, 0.0)

        s = s * dec_s + spk
        x = x * dec_t + spk
        y = y * dec_t + spk
        R = R * dec_R

    return counts, W, V, R_trace

def compute(world, P, flags):
    c_enc = world.creb(P["t_enc"])
    I_enc = I0 + P["kc"] * c_enc
    sigma = c_enc if flags["gate"] else np.ones(M)

    cnt, W, Venc, R_tr = simulate(W0, I_enc, P["s0"] * world.p, P, sigma, learn=True, t_end=T_ENC_TOTAL, swr_seed=world.swr_rng_seed)
    eng = cnt >= FIRE_MIN
    mask = (world.u_cue < P["cue_f"]).astype(float)
    c_ret = world.creb(P["t_enc"] + P["ret_dt"])
    I_ret = I0 + P["kc"] * c_ret
    Wr = W0 if flags["naive"] else W

    cnt_r, _, Vret, _ = simulate(Wr, I_ret, P["s0"] * world.p * mask, P, sigma, learn=False, t_end=T_RET_TOTAL, swr_seed=world.swr_rng_seed)
    fired = cnt_r >= FIRE_MIN
    e_unc = eng & (mask == 0)
    
    return dict(
        c_enc=c_enc, c_ret=c_ret, I_enc=I_enc, I_ret=I_ret, W=W, Wr=Wr,
        cnt=cnt, eng=eng, mask=mask, fired=fired, Venc=Venc, Vret=Vret, R_tr=R_tr,
        recall=fired[e_unc].mean() if e_unc.any() else np.nan,
        fp=fired[~eng & (mask == 0)].mean() if (~eng & (mask == 0)).any() else np.nan,
        n_unc_eng=int(e_unc.sum()), spont=int((I_ret > I_HOPF).sum()),
        dW=W[np.ix_(eng, eng)].mean() - W0[np.ix_(eng, eng)].mean() if eng.any() else np.nan
    )

def single_neuron_traj(I_total, v_init, w_init, T=260.0):
    v, w = float(v_init), float(w_init)
    n_steps = int(T / dt)
    tr = np.empty((n_steps, 2))
    for k in range(n_steps):
        tr[k] = (v, w)
        vn = v + dt * (v - v ** 3 / 3 - w + I_total)
        w = w + dt * eps * (v + a - b * w)
        v = vn
    return tr

CMAP = ListedColormap(["#21262d", "#d29922", "#388bfd", "#3fb950", "#f85149"])
LEGEND = [Patch(color="#21262d", label="Silent"), Patch(color="#d29922", label="Engram: Missed"),
          Patch(color="#388bfd", label="Engram: Cued"), Patch(color="#3fb950", label="Engram: Recalled"),
          Patch(color="#f85149", label="False Positive")]

class App:
    def __init__(self, seed=7):
        self.seed = seed
        self.world = World(seed)
        self.P = dict(DEFAULTS)
        self.flags = dict(auto=True, gate=True, naive=False)
        self.sel = None
        self.res = None
        self.muted = False
        self.sliders = {}
        self.build()
        self.recompute()

    def build(self):
        self.fig = plt.figure(figsize=(17, 10.5))
        gs = self.fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.91, bottom=0.33, hspace=0.45, wspace=0.26, width_ratios=[1, 1.15, 1.15, 1.2])
        self.ax_creb = self.fig.add_subplot(gs[0, 0])
        self.ax_stim = self.fig.add_subplot(gs[0, 1])
        self.ax_out = self.fig.add_subplot(gs[0, 2])
        self.ax_txt = self.fig.add_subplot(gs[0, 3])
        self.ax_W = self.fig.add_subplot(gs[1, 0])
        self.ax_ph = self.fig.add_subplot(gs[1, 1:3])
        self.ax_tr = self.fig.add_subplot(gs[1, 3])
        self.fig.suptitle("Minimal Biophysical Engram Allocation: CREB, WTA PV-Inhibition, SWRs & Local STDP", fontsize=12, fontweight="bold", color="#f0f6fc")

        spec_l = [("Encoding Time (h)", "t_enc", 0, 24, 0.5), ("Retrieval Delay (h)", "ret_dt", 0, 24, 0.5),
                  ("CREB Gain kc", "kc", 0.0, 0.30, 0.005), ("Sensory Drive s0", "s0", 0.10, 0.50, 0.01),
                  ("PV+ Inhibition g_inh", "g_inh", 0.0, 15.0, 0.5), ("SWR Poisson Rate", "p_swr", 0.0, 0.02, 0.001)]
        spec_r = [("SWR Replay Gain", "g_swr", 0.0, 0.25, 0.01), ("Plasticity Rate eta", "eta", 0.02, 0.80, 0.01), 
                  ("LTD/LTP Ratio", "ratio", 0.0, 1.5, 0.05), ("Weight Cap w_max", "wmax", 0.05, 0.60, 0.01), 
                  ("Cue Fraction", "cue_f", 0.05, 0.95, 0.05), ("Presyn. s (Phase)", "s_on", 0.1, 1.0, 0.05)]
        
        for row, (lab, key, lo, hi, st) in enumerate(spec_l): self.add_slider(lab, key, lo, hi, st, 0.16, 0.255 - row * 0.038)
        for row, (lab, key, lo, hi, st) in enumerate(spec_r): self.add_slider(lab, key, lo, hi, st, 0.55, 0.255 - row * 0.038)

        def btn(rect, text, cb):
            bt = Button(self.fig.add_axes(rect), text, color="#21262d", hovercolor="#30363d")
            bt.label.set_fontsize(8.5); bt.label.set_color("#e6edf3")
            bt.on_clicked(cb); return bt

        self.b_run = btn([0.80, 0.23, 0.085, 0.032], "Re-Simulate", lambda e: self.recompute())
        self.b_swr = btn([0.895, 0.23, 0.085, 0.032], "New SWR Roll", self.new_swr_roll)
        self.b_cue = btn([0.80, 0.19, 0.085, 0.032], "New Cue Mask", self.new_mask)
        self.b_world = btn([0.895, 0.19, 0.085, 0.032], "New World", self.new_world)

        self.chk = CheckButtons(self.fig.add_axes([0.80, 0.065, 0.18, 0.11], facecolor="#161b22"), ["Auto-run", "CREB gates plasticity", "Naive net control"], [True, True, False])
        for t in self.chk.labels: t.set_fontsize(8); t.set_color("#e6edf3")
        self.chk.on_clicked(self.on_flag)

        self.status_txt = self.fig.text(0.80, 0.03, "", fontsize=8, color="#8b949e")
        self.timer = self.fig.canvas.new_timer(interval=250)
        self.timer.single_shot = True; self.timer.add_callback(self.recompute)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def add_slider(self, label, key, lo, hi, step, x0, y):
        s = Slider(self.fig.add_axes([x0, y, 0.19, 0.02], facecolor="#21262d"), label, lo, hi, valinit=self.P[key], valstep=step, color="#388bfd")
        s.label.set_fontsize(8); s.valtext.set_fontsize(8); s.on_changed(lambda v, k=key: self.on_param(k, v))
        self.sliders[key] = s

    def on_param(self, key, v):
        self.P[key] = float(v)
        if key == "s_on": self.draw_phase(); self.fig.canvas.draw_idle(); return
        if self.flags["auto"]: self.timer.stop(); self.timer.start()

    def on_flag(self, _):
        st = self.chk.get_status()
        self.flags.update(auto=st[0], gate=st[1], naive=st[2])
        self.recompute()

    def new_swr_roll(self, _=None): self.world.swr_rng_seed += 13; self.recompute()
    def new_mask(self, _=None): self.world.u_cue = np.random.default_rng().random(M); self.recompute()
    def new_world(self, _=None): self.seed += 1; self.world = World(self.seed); self.sel = None; self.recompute()

    def on_click(self, ev):
        if ev.inaxes in (self.ax_creb, self.ax_stim, self.ax_out) and ev.xdata is not None:
            self.sel = int(np.clip(round(ev.ydata), 0, G - 1)) * G + int(np.clip(round(ev.xdata), 0, G - 1))
            self.draw_all()

    def recompute(self, *_):
        t0 = time.time()
        self.res = compute(self.world, self.P, self.flags)
        if self.sel is None:
            pick = np.where(self.res["eng"] & (self.res["mask"] == 0) & self.res["fired"])[0]
            self.sel = int(pick[len(pick) // 2]) if len(pick) else (np.where(self.res["eng"])[0][0] if self.res["eng"].any() else 0)
        self.dur = time.time() - t0
        self.draw_all()

    def draw_all(self):
        self.draw_maps(); self.draw_W(); self.draw_text(); self.draw_phase(); self.draw_trace()
        self.status_txt.set_text(f"Compute: {self.dur*1000:.0f} ms | World Seed: {self.seed}")
        self.fig.canvas.draw_idle()

    def _ring(self, ax):
        ax.scatter([self.sel % G], [self.sel // G], s=280, facecolors="none", edgecolors="#00ffcc", linewidths=2.2, zorder=6)

    def draw_maps(self):
        R, P, g = self.res, self.P, lambda z: np.asarray(z).reshape(G, G)
        ax = self.ax_creb; ax.clear(); ax.imshow(g(R["c_enc"]), cmap="magma", vmin=0, vmax=1); ax.set_title(f"1. CREB Excitability c_i", fontsize=8.5); self._ring(ax)
        ax = self.ax_stim; ax.clear(); ax.imshow(g(P["s0"] * self.world.p), cmap="inferno", vmin=0, vmax=0.45)
        ey, ex = np.where(g(R["eng"])); ax.scatter(ex, ey, s=140, marker="s", facecolors="none", edgecolors="#00e5ff", linewidths=1.5)
        ax.set_title(f"2. Sensory Input & Engram", fontsize=8.5); self._ring(ax)
        ax = self.ax_out; ax.clear()
        o = np.zeros(M, int)
        o[R["eng"] & ~R["fired"]] = 1; o[R["eng"] & R["fired"] & (R["mask"] == 1)] = 2; o[R["eng"] & R["fired"] & (R["mask"] == 0)] = 3; o[~R["eng"] & R["fired"]] = 4
        ax.imshow(g(o), cmap=CMAP, vmin=0, vmax=4); ax.set_title(f"3. Pattern Completion\nRecall: {100*(R['recall'] if not np.isnan(R['recall']) else 0):.0f}%", fontsize=8.5)
        ax.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6.5, frameon=False); self._ring(ax)
        for a_ in (self.ax_creb, self.ax_stim, self.ax_out): a_.set_xticks([]); a_.set_yticks([])

    def draw_W(self):
        R = self.res; ax = self.ax_W; ax.clear()
        order = np.r_[np.where(R["eng"])[0], np.where(~R["eng"])[0]]
        ax.imshow(R["W"][np.ix_(order, order)], cmap="viridis", vmin=0, vmax=max(self.P["wmax"], 1e-3))
        ne = int(R["eng"].sum())
        if ne > 0: ax.axhline(ne - 0.5, color="#00e5ff", lw=0.8, ls="--"); ax.axvline(ne - 0.5, color="#00e5ff", lw=0.8, ls="--")
        ax.set_title("4. Synaptic Matrix W_ij", fontsize=8.5)

    def draw_text(self):
        R, i = self.res, self.sel; ax = self.ax_txt; ax.clear(); ax.axis("off")
        syn_in = self.P["s_on"] * R["Wr"][i, R["fired"] & (np.arange(M) != i)].sum()
        tot_drive = R["I_ret"][i] + syn_in
        txt = (f"NETWORK METRICS\n──────────────────────────────────\nEngram Sparsity   : {R['eng'].sum()} / {M}\n"
               f"Uncued Targets    : {R['n_unc_eng']}\nRecall            : {100*R['recall'] if not np.isnan(R['recall']) else 0:.1f}%\n"
               f"Mean Engram ΔW    : +{R['dW']:.3f}\nSWR Peak Activity : {R['R_tr'].max():.2f}\n\n"
               f"SELECTED NEURON #{i}\n──────────────────────────────────\nCREB (enc/ret): {R['c_enc'][i]:.2f} / {R['c_ret'][i]:.2f}\n"
               f"Resting Drive : {R['I_ret'][i]:.3f}\nSynaptic In   : +{syn_in:.3f}\n"
               f"Total Drive   : {tot_drive:.3f} (Hopf = {I_HOPF:.3f})")
        ax.text(0.0, 0.98, txt, va="top", family="monospace", fontsize=8.2, color="#e6edf3")

    def draw_phase(self):
        R, s_on, i = self.res, self.P["s_on"], self.sel; ax = self.ax_ph; ax.clear()
        Ir = R["I_ret"][i]; W = R["Wr"]; act = R["fired"].copy(); act[i] = False
        I_learned = Ir + s_on * W[i, act].sum()
        vg, wg = np.meshgrid(np.linspace(-2.4, 2.4, 22), np.linspace(-1.0, 2.2, 16))
        mag = np.hypot(vg - vg ** 3 / 3 - wg + I_learned, eps * (vg + a - b * wg)) + 1e-6
        ax.quiver(vg, wg, (vg - vg ** 3 / 3 - wg + I_learned) / mag, eps * (vg + a - b * wg) / mag, mag, cmap="Blues", alpha=0.35, pivot="mid")
        ax.axvspan(vH, -vH, color="#f85149", alpha=0.10); ax.axvline(vH, color="#f85149", ls=":", lw=1.2, alpha=0.8)
        vv = np.linspace(-2.4, 2.4, 400); ax.plot(vv, (vv + a) / b, color="#ff7b72", lw=2.0)
        
        for Itot, col, ls, lw in [(Ir, "#8b949e", "--", 1.3), (Ir + s_on * W0[i, act].sum(), "#d29922", "-.", 1.6), (I_learned, "#00e5ff", "-", 2.4)]:
            ax.plot(vv, vv - vv ** 3 / 3 + Itot, color=col, ls=ls, lw=lw)
            fv, fw = fixed_point(Itot)
            ax.plot(fv, fw, "o", ms=9, mfc=col if is_stable(fv[0]) else "#161b22", mec=col, mew=2.2, zorder=8)
            
        tr = single_neuron_traj(I_learned, *fixed_point(Ir)); ax.plot(tr[:, 0], tr[:, 1], color="#3fb950", lw=1.6, alpha=0.9)
        ax.set_title(f"5. State Space (Selected Neuron) | State: {'LIMIT CYCLE' if I_learned > I_HOPF else 'FIXED POINT'}", fontsize=8.5, fontweight="bold")
        ax.set_xlim(-2.35, 2.35); ax.set_ylim(-1.05, 2.15); ax.grid(True, alpha=0.25)

    def draw_trace(self):
        R, i = self.res, self.sel; ax = self.ax_tr; ax.clear()
        tE = np.arange(len(R["Venc"])) * dt; tR = np.arange(len(R["Vret"])) * dt
        ax.axvspan(T_PRE, T_PRE + T_STIM, color="#d29922", alpha=0.18)
        ax.plot(tE, R["Venc"][:, i] + 4.2, color="#e6edf3", lw=1.0)
        ax.fill_between(tE, 1.4, 1.4 + 2.0 * R["R_tr"], color="#a371f7", alpha=0.65)
        ax.plot(tR, R["Vret"][:, i] - 1.8, color="#3fb950", lw=1.1)
        ax.set_yticks([-1.8, 1.8, 4.2]); ax.set_yticklabels(["Retrieval V", "SWR R(t)", "Encoding V"], fontsize=7.5)
        ax.set_xlim(0, T_ENC_TOTAL); ax.set_title("6. Temporal Dynamics & SWRs", fontsize=8.5)

if __name__ == "__main__":
    App()
    plt.show()