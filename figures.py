"""
Engram Formation Model: CREB allocation + SWR consolidation + Inhibitory size control
=====================================================================================

A minimal network model that combines three experimental findings:

  1. Han et al. (2007)        - neurons with higher CREB are preferentially allocated to a memory
  2. Girardeau et al. (2009)  - sharp-wave ripples (SWRs) are required for consolidation
  3. Stefanelli et al. (2016) - dentate inhibitory interneurons limit the size of the engram

MODEL COMPONENTS
----------------
  Neurons      : M=144 FitzHugh-Nagumo units (excitable, sit BELOW the Hopf bifurcation).
  CREB         : c_i in [0.1, 1] raises baseline drive:  I_base = 0.10 + kc * c_i.
                 It is ON at encoding (kc_enc) and OFF at retrieval (kc_ret = 0).
  Stimulus     : I_ext = s0 * p_i, a graded input pattern delivered 20 <= t < 110.
  Recurrence   : sparse (15%) weights W, initially weak, drive I += W @ s.
  Inhibition   : global winner-take-all feedback  I_inh' = (-I_inh + g_inh * mean(s)) / 12.
  SWR          : external Poisson pulses (rate p_swr per step, only after t > 110) that give a
                 brief strong depolarising kick g_swr * R, with R decaying over ~15 time units.
  Plasticity   : burst-threshold Hebbian rule. Traces x,y only count ABOVE theta_p, so isolated
                 spikes (sensory drive alone) do nothing; only bursts (SWR replay) change W.
                 LTP is scaled by the postsynaptic CREB c_i; soft bound wmax; LTD ratio 0.45.
  Engram       : neurons that fired >= FIRE_MIN spikes during stimulus presentation at encoding.
  Retrieval    : CREB is zeroed, only 40% of the pattern is presented as a cue, and we ask whether
                 engram neurons that received NO cue still fire (pattern completion through W).

USAGE
-----
  python engram_model_simulation.py              # full run (N=50 seeds), saves figures to ./figures
  python engram_model_simulation.py --quick      # small run for a smoke test
  python engram_model_simulation.py --seeds 100 --outdir figs

Notes on statistics: all "±" values are 95% CI half-widths of the mean (t-distribution),
NOT standard deviations. Recall comparison uses a paired test across seeds.
"""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------------------
# Fixed topology and constants
# --------------------------------------------------------------------------------------
M = 144
a, b, eps = 0.7, 0.8, 0.08
dt = 0.1
vH = -np.sqrt(1 - eps * b)
I_HOPF = (vH + a) / b - vH + vH ** 3 / 3  # drive at which a noise-free neuron starts oscillating
FIRE_MIN = 2

_rng_top = np.random.default_rng(42)
conn_mask = _rng_top.random((M, M)) < 0.15
np.fill_diagonal(conn_mask, 0)
W0 = np.where(conn_mask, 0.008 * _rng_top.random((M, M)), 0.0)

PARAMS = dict(
    kc_enc=0.18,   # CREB gain at encoding
    kc_ret=0.00,   # CREB gain at retrieval (decayed)
    s0=0.22,       # stimulus drive
    p_swr=0.004,   # SWR probability per time step
    g_swr=0.28,    # SWR drive
    g_inh=2.8,     # global inhibition gain
    eta=0.55,      # plasticity rate
    ratio=0.45,    # LTD / LTP
    wmax=0.40,     # soft weight bound
    theta_p=0.95,  # burst threshold on the plasticity traces
    cue_f=0.40,    # fraction of pattern given as cue
    sigma_v=0.04,  # membrane noise
)


# --------------------------------------------------------------------------------------
# Model core
# --------------------------------------------------------------------------------------
def fixed_point(I):
    """Resting state of the FHN neuron for constant drive I (Newton iteration)."""
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(40):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        v = v - f / (v ** 2 + (1 / b - 1))
    return v, (v + a) / b


class World:
    """One random 'animal': CREB levels, stimulus pattern, cue selection."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.c_i = rng.uniform(0.1, 1.0, M)
        self.p = rng.uniform(0.0, 1.0, M)
        self.u_cue = rng.random(M)


def simulate_epoch(W_init, I_base, I_ext, c_i, learn, P, T_total, rng, record=False):
    W = W_init.copy()
    n_steps = int(T_total / dt)
    v, w = fixed_point(I_base)
    s, x, y = np.zeros(M), np.zeros(M), np.zeros(M)
    I_inh = 0.0
    counts = np.zeros(M, int)
    dec_s, dec_t, dec_R = np.exp(-dt / 5.0), np.exp(-dt / 25.0), np.exp(-dt / 15.0)
    noise = rng.normal(0, P["sigma_v"] * np.sqrt(dt), (n_steps, M))
    R = 0.0
    rec = None
    if record:
        rec = dict(spikes=np.zeros((n_steps, M), bool), R=np.zeros(n_steps),
                   I_inh=np.zeros(n_steps), rate=np.zeros(n_steps))

    for k in range(n_steps):
        t = k * dt
        # external SWR pulses (only during the post-stimulus 'rest' period of encoding)
        if learn and t > 110.0 and rng.random() < P["p_swr"]:
            R += 1.0
        else:
            R *= dec_R
        # global inhibition
        I_inh += dt * (-I_inh + P["g_inh"] * s.mean()) / 12.0

        I = I_base + W @ s - I_inh
        if 20.0 <= t < 110.0:
            I = I + I_ext
        if R > 0.01:
            I = I + P["g_swr"] * R

        vn = np.clip(v + dt * (v - v ** 3 / 3 - w + I) + noise[k], -3.0, 3.0)
        wn = w + dt * eps * (vn + a - b * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        if 20.0 <= t < 110.0:
            counts += spk
        s = s * dec_s + spk
        x = x * dec_t + spk
        y = y * dec_t + spk

        if record:
            rec["spikes"][k] = spk
            rec["R"][k] = R
            rec["I_inh"][k] = I_inh
            rec["rate"][k] = s.mean()

        if learn and spk.any():
            sp = spk.astype(float)
            x_burst = np.maximum(x - P["theta_p"], 0.0)
            y_burst = np.maximum(y - P["theta_p"], 0.0)
            ltp = np.outer(c_i * sp, x_burst) * (P["wmax"] - W)
            ltd = np.outer(c_i * y_burst, sp) * W
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            np.fill_diagonal(W, 0.0)
    return counts, W, rec


def run_single_experiment(seed, override=None, record=False):
    P = PARAMS.copy()
    if override:
        P.update(override)
    world = World(seed)
    rng_enc = np.random.default_rng([seed, 1])
    rng_ret = np.random.default_rng([seed, 2])

    I_base_enc = 0.10 + P["kc_enc"] * world.c_i
    I_base_ret = 0.10 + P["kc_ret"] * world.c_i
    I_ext = P["s0"] * world.p

    # 1. Encoding (high CREB, SWRs active)
    cnt_enc, W_learned, rec = simulate_epoch(W0, I_base_enc, I_ext, world.c_i, True, P, 240.0, rng_enc, record)
    engram = cnt_enc >= FIRE_MIN
    n_engram = int(engram.sum())

    # 2. Retrieval (CREB decayed, partial cue, no plasticity)
    mask = (world.u_cue < P["cue_f"]).astype(float)
    cnt_ret, _, _ = simulate_epoch(W_learned, I_base_ret, I_ext * mask, world.c_i, False, P, 150.0, rng_ret)
    fired = cnt_ret >= FIRE_MIN

    # Allocation: CREB tertiles
    hi = world.c_i >= np.percentile(world.c_i, 66)
    lo = world.c_i <= np.percentile(world.c_i, 33)
    p_hi = engram[hi].mean() if hi.any() else 0.0
    p_lo = engram[lo].mean() if lo.any() else 0.0
    p1, p2 = np.clip(p_hi, 1e-6, 1 - 1e-6), np.clip(p_lo, 1e-6, 1 - 1e-6)
    odds_ratio = (p1 / (1 - p1)) / (p2 / (1 - p2))

    # Recall: uncued engram cells vs uncued non-engram cells matched on stimulus strength p
    uncued_engram = engram & (mask == 0)
    recall_e = fired[uncued_engram].mean() if uncued_engram.any() else np.nan
    mean_p = world.p[engram].mean() if engram.any() else 0.5
    matched = (~engram) & (mask == 0) & (np.abs(world.p - mean_p) <= 0.1)
    recall_c = fired[matched].mean() if matched.any() else np.nan

    dW = (W_learned[np.ix_(engram, engram)].mean() - W0[np.ix_(engram, engram)].mean()) if n_engram > 1 else 0.0

    out = dict(n_engram=n_engram, odds_ratio=odds_ratio, p_hi=p_hi, p_lo=p_lo,
               engram_recall=recall_e, control_recall=recall_c, dW=dW)
    if record:
        out.update(rec=rec, W_learned=W_learned, engram=engram, c_i=world.c_i, p=world.p,
                   mask=mask, fired=fired)
    return out


# --------------------------------------------------------------------------------------
# Ensemble + statistics
# --------------------------------------------------------------------------------------
KEYS = ["n_engram", "odds_ratio", "p_hi", "p_lo", "engram_recall", "control_recall", "dW"]


def run_ensemble(n_seeds, override=None):
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(run_single_experiment, s, override) for s in range(n_seeds)]
        res = [f.result() for f in futs]
    return {k: np.array([r[k] for r in res], float) for k in KEYS}


def mean_ci(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return (float(arr.mean()) if len(arr) else np.nan), np.nan
    return arr.mean(), stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))


def geo_mean_ci(arr):
    """Odds ratios are ratios: summarise on the log scale."""
    l = np.log(np.asarray(arr, float))
    m, c = mean_ci(l)
    return np.exp(m), np.exp(m - c), np.exp(m + c)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})
C_ENG, C_CTL, C_GREY = "#c0392b", "#7f8c8d", "#2c3e50"


def fig_components(outdir):
    """Figure 1: what each building block does, in isolation."""
    P = PARAMS
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # A. Excitable neuron + CREB shifts drive toward the Hopf point (vectorised, 20 noisy trials each)
    Is = np.linspace(0.0, 0.7, 36)
    n_tr, T = 20, 90.0
    IG = np.repeat(Is, n_tr)
    v, w = fixed_point(IG)
    rng = np.random.default_rng(0)
    cnt = np.zeros_like(IG)
    for _ in range(int(T / dt)):
        vn = np.clip(v + dt * (v - v ** 3 / 3 - w + IG) + rng.normal(0, P["sigma_v"] * np.sqrt(dt), IG.shape), -3, 3)
        w = w + dt * eps * (vn + a - b * w)
        cnt += (vn > 0) & (v <= 0)
        v = vn
    cnt = cnt.reshape(len(Is), n_tr).mean(1)
    A = ax[0, 0]
    A.plot(Is, cnt, color=C_GREY, lw=2)
    A.axvline(I_HOPF, color="k", ls="--", lw=1)
    A.text(I_HOPF + 0.01, cnt.max() * 0.05, f"Hopf\nI={I_HOPF:.2f}", fontsize=8)
    lo_b, hi_b = 0.10 + P["kc_enc"] * 0.1, 0.10 + P["kc_enc"] * 1.0
    A.axvspan(lo_b, hi_b, color=C_ENG, alpha=0.15, label="baseline range\n(low→high CREB)")
    A.axvspan(hi_b, hi_b + P["s0"], color="#2980b9", alpha=0.12, label="+ stimulus (s0·p)")
    A.set_xlabel("Constant drive I"); A.set_ylabel("Spikes in 90 time units")
    A.set_title("A. Excitable neuron; CREB raises baseline drive"); A.legend(fontsize=8, loc="upper left")

    # B. Burst-threshold plasticity gate
    B = ax[0, 1]
    t = np.arange(0, 200, dt)
    spikes_single = np.zeros_like(t); spikes_burst = np.zeros_like(t)
    for ts in (20, 80, 140):                       # isolated spikes (sensory-like)
        spikes_single[int(ts / dt)] = 1
    for ts in (20, 23, 26, 29):                    # one burst (SWR-like)
        spikes_burst[int(ts / dt)] = 1
    dec = np.exp(-dt / 25.0)
    for sp, col, lab in ((spikes_single, C_GREY, "isolated spikes"), (spikes_burst, C_ENG, "burst (4 spikes / 9 t.u.)")):
        x = np.zeros_like(t)
        for k in range(1, len(t)):
            x[k] = x[k - 1] * dec + sp[k]
        B.plot(t, np.maximum(x - P["theta_p"], 0), color=col, lw=2, label=lab)
    B.set_xlabel("Time"); B.set_ylabel("Plasticity signal  max(x − θp, 0)")
    B.set_title(f"B. Burst gate (θp = {P['theta_p']})"); B.legend(fontsize=8)

    # C. SWR pulses
    C = ax[1, 0]
    rng = np.random.default_rng(3)
    R = 0.0; Rt = []; ev = []
    for k, tt in enumerate(np.arange(0, 240, dt)):
        if tt > 110 and rng.random() < P["p_swr"]:
            R += 1.0; ev.append(tt)
        else:
            R *= np.exp(-dt / 15.0)
        Rt.append(R)
    C.plot(np.arange(0, 240, dt), np.array(Rt) * P["g_swr"], color=C_ENG)
    C.axvspan(20, 110, color="#2980b9", alpha=0.12, label="stimulus window")
    C.set_xlabel("Time"); C.set_ylabel("SWR drive  g_swr · R")
    C.set_title(f"C. Poisson SWR pulses ({len(ev)} events after t=110)"); C.legend(fontsize=8)

    # D. Inhibition: steady-state vs activity, with feedback gain
    D = ax[1, 1]
    act = np.linspace(0, 0.3, 50)
    for g in (0.2, 2.8, 6.0):
        D.plot(act, g * act, label=f"g_inh = {g}", lw=2)
    D.set_xlabel("Population activity  mean(s)"); D.set_ylabel("Steady-state inhibition  I_inh")
    D.set_title("D. Global WTA feedback"); D.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_components.png"), dpi=160); plt.close(fig)


def fig_single_trial(outdir, seed=0):
    """Figure 2: one full encode→retrieve trial."""
    r = run_single_experiment(seed, record=True)
    rec, eng, c = r["rec"], r["engram"], r["c_i"]
    order = np.argsort(c)
    sp = rec["spikes"][:, order]
    tt = np.arange(sp.shape[0]) * dt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    A = ax[0, 0]
    ti, ni = np.nonzero(sp)
    col = np.where(eng[order][ni], C_ENG, C_GREY)
    A.scatter(tt[ti], ni, s=1.5, c=col)
    A.axvspan(20, 110, color="#2980b9", alpha=0.08)
    A.set_xlabel("Time"); A.set_ylabel("Neuron (sorted by CREB, low→high)")
    A.set_title(f"A. Encoding raster (engram = red, n={r['n_engram']})")

    B = ax[0, 1]
    B.plot(tt, rec["rate"], color=C_GREY, label="population activity")
    B.plot(tt, rec["I_inh"] / PARAMS["g_inh"], color="#2980b9", label="inhibition (÷g_inh)")
    B2 = B.twinx(); B2.plot(tt, rec["R"], color=C_ENG, alpha=0.6, label="SWR  R"); B2.set_ylabel("R")
    B.axvspan(20, 110, color="#2980b9", alpha=0.08)
    B.set_xlabel("Time"); B.set_title("B. Stimulus phase, then SWR replay"); B.legend(fontsize=8, loc="upper left")

    for k, (Wm, ttl) in enumerate(((W0, "C. W before encoding"), (r["W_learned"], "D. W after encoding"))):
        axx = ax[1, k]
        idx = np.concatenate([np.where(eng)[0], np.where(~eng)[0]])
        im = axx.imshow(Wm[np.ix_(idx, idx)], cmap="viridis", vmin=0, vmax=PARAMS["wmax"])
        axx.axhline(eng.sum() - 0.5, color="w", lw=0.8); axx.axvline(eng.sum() - 0.5, color="w", lw=0.8)
        axx.set_title(ttl + " (engram block top-left)"); axx.set_xlabel("presynaptic"); axx.set_ylabel("postsynaptic")
        fig.colorbar(im, ax=axx, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_single_trial.png"), dpi=160); plt.close(fig)


def sem_ci_plot(ax, xs, ens_list, key, **kw):
    m = [mean_ci(e[key])[0] for e in ens_list]
    c = [mean_ci(e[key])[1] for e in ens_list]
    ax.errorbar(xs, m, yerr=c, marker="o", capsize=3, lw=2, **kw)


def fig_validation(outdir, n_seeds):
    """Figure 3: the three validations plus the decoupled recall test."""
    print(f"Baseline ensemble (N={n_seeds}) ...")
    base = run_ensemble(n_seeds)
    n_sw = max(20, n_seeds // 2)
    print("SWR dose-response ...")
    swr_rates = [0.0, 0.001, 0.002, 0.004, 0.008]
    swr_ens = [base if r == PARAMS["p_swr"] else run_ensemble(n_sw, dict(p_swr=r)) for r in swr_rates]
    swr_off = swr_ens[0]
    print("Inhibition sweep ...")
    inh_vals = [0.2, 1.0, 2.0, 2.8, 4.0, 6.0]
    inh_ens = [base if g == PARAMS["g_inh"] else run_ensemble(n_sw, dict(g_inh=g)) for g in inh_vals]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # A. Han 2007: allocation by CREB
    A = ax[0, 0]
    pm, pc = mean_ci(base["p_hi"]), mean_ci(base["p_lo"])
    A.bar(["low CREB\n(bottom ⅓)", "high CREB\n(top ⅓)"], [pc[0], pm[0]], yerr=[pc[1], pm[1]],
          color=[C_CTL, C_ENG], capsize=4)
    gm, glo, ghi = geo_mean_ci(base["odds_ratio"])
    A.set_ylabel("P(neuron in engram)")
    A.set_title(f"A. CREB allocation\nOR = {gm:.1f} [{glo:.1f}, {ghi:.1f}] (geo. mean, 95% CI)")

    # B. Decoupled recall, paired
    B = ax[0, 1]
    e, cc = base["engram_recall"], base["control_recall"]
    ok = ~np.isnan(e) & ~np.isnan(cc)
    for i in np.where(ok)[0]:
        B.plot([0, 1], [cc[i], e[i]], color="grey", alpha=0.3, lw=0.8)
    B.errorbar([0, 1], [cc[ok].mean(), e[ok].mean()],
               yerr=[mean_ci(cc[ok])[1], mean_ci(e[ok])[1]], color="k", marker="o", capsize=4, lw=2)
    B.set_xticks([0, 1]); B.set_xticklabels(["matched\nnon-engram", "engram"]); B.set_ylabel("Recall (uncued cells)")
    d = e[ok] - cc[ok]
    t_stat, p_t = stats.ttest_rel(e[ok], cc[ok])
    try:
        p_w = stats.wilcoxon(e[ok], cc[ok]).pvalue
    except ValueError:
        p_w = np.nan
    dz = d.mean() / d.std(ddof=1)
    B.set_title(f"B. Recall with CREB = 0 (paired)\nΔ={d.mean():.3f}, p_t={p_t:.2g}, p_W={p_w:.2g}, d_z={dz:.2f}")

    # C. SWR dW dose-response
    C = ax[0, 2]
    sem_ci_plot(C, swr_rates, swr_ens, "dW", color=C_ENG)
    C.set_xlabel("SWR rate p_swr"); C.set_ylabel("Engram-engram ΔW")
    drop = 1 - swr_off["dW"].mean() / base["dW"].mean()
    C.set_title(f"C. SWR ablation: ΔW falls {drop:.0%} at p_swr=0")

    # D. SWR consequence for recall (extension beyond original script)
    D = ax[1, 0]
    sem_ci_plot(D, swr_rates, swr_ens, "engram_recall", color=C_ENG, label="engram")
    sem_ci_plot(D, swr_rates, swr_ens, "control_recall", color=C_CTL, label="matched control")
    D.set_xlabel("SWR rate p_swr"); D.set_ylabel("Recall"); D.legend(fontsize=8)
    D.set_title("D. Does SWR loss impair recall?")

    # E. Inhibition sweep
    E = ax[1, 1]
    sem_ci_plot(E, inh_vals, inh_ens, "n_engram", color=C_GREY)
    E.set_xlabel("Inhibition gain g_inh"); E.set_ylabel("Engram size (# neurons)")
    rho, p_rho = stats.spearmanr(np.repeat(inh_vals, [len(e_["n_engram"]) for e_ in inh_ens]),
                                 np.concatenate([e_["n_engram"] for e_ in inh_ens]))
    E.set_title(f"E. Inhibition limits engram size\nSpearman ρ = {rho:.2f}, p = {p_rho:.1g}")

    # F. Summary
    F = ax[1, 2]; F.axis("off")
    lines = [
        f"Baseline (N={n_seeds}, mean ± 95% CI)",
        f"Engram size           {mean_ci(base['n_engram'])[0]:.1f} ± {mean_ci(base['n_engram'])[1]:.1f}",
        f"CREB OR (geo.)        {gm:.2f}  [{glo:.2f}, {ghi:.2f}]",
        f"Engram recall         {mean_ci(e)[0]:.1%} ± {mean_ci(e)[1]:.1%}",
        f"Control recall        {mean_ci(cc)[0]:.1%} ± {mean_ci(cc)[1]:.1%}",
        f"Paired Δ recall       {d.mean():.3f}  (p={p_t:.2g})",
        "",
        f"ΔW  SWR on / off      {base['dW'].mean():.4f} / {swr_off['dW'].mean():.4f}",
        f"Engram  g_inh {inh_vals[0]} → {inh_vals[-1]}: "
        f"{inh_ens[0]['n_engram'].mean():.1f} → {inh_ens[-1]['n_engram'].mean():.1f}",
    ]
    F.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
    F.set_title("F. Summary")

    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_validation.png"), dpi=160); plt.close(fig)
    print("\n".join(lines))


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 6
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Hopf bifurcation at I = {I_HOPF:.3f}; max baseline drive = {0.10 + PARAMS['kc_enc']:.3f} (excitable regime)")
    fig_components(args.outdir)
    fig_single_trial(args.outdir)
    fig_validation(args.outdir, args.seeds)
    print(f"Figures saved to {os.path.abspath(args.outdir)}")