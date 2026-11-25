import argparse
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.stats as stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- model constants
M = 144
a, b, eps = 0.7, 0.8, 0.08
dt = 0.1
FIRE_MIN = 2
T_ENC, T_RET = 600.0, 150.0
SENS_ON, SENS_OFF = 20.0, 110.0
TAU_R = 15.0

TUNE_SEEDS = range(1000, 1200)      # use for any parameter exploration
TEST_SEEDS = range(100000, 100200)  # touched only by --mode test

PARAMS = dict(
    kc_enc=0.30, s0=0.15, p_swr=0.0009, g_swr=0.75,
    g_inh=18.0, g_inh_ret=6.0, I_ret=0.16,
    eta=0.70, ratio=0.50, wmax=0.50,
    theta_pre=1.00, theta_post=1.10,   # pre: needs >=2 spikes; post: see docstring item 9
    tau_pre=20.0, tau_post=35.0,
    cue_f=0.40, sigma_v=0.032,
    plast_ci=False,                    # v3 multiplied plasticity by c_i; off by default now
    replay_rho=0.8, replay_cap=3.0,    # strength / cap of encoding-activity weighting of SWR drive
    conn_p=0.15, w0_scale=0.008,
    creb_flat=False,                   # True: same mean CREB drive for every cell (no c_i bias)
    match_caliper=0.15,
)

CONDITIONS = {
    "uniform_swr":       dict(swr_mode="uniform"),
    "replay_swr":        dict(swr_mode="replay"),
    "replay_shuffled":   dict(swr_mode="replay_shuffled"),
    "tonic_matched":     dict(swr_mode="tonic"),
    "swr_no_plasticity": dict(swr_mode="uniform", block=True),
    "no_swr":            dict(swr_mode="none"),
    "creb_off":          dict(swr_mode="uniform", params=dict(creb_flat=True)),
    "plast_ci_v3":       dict(swr_mode="uniform", params=dict(plast_ci=True)),
}

CONTRASTS = [  # (A, B): A minus B, paired by seed
    ("uniform_swr", "no_swr"),
    ("uniform_swr", "swr_no_plasticity"),
    ("uniform_swr", "tonic_matched"),
    ("replay_swr", "replay_shuffled"),
    ("replay_swr", "uniform_swr"),
    ("uniform_swr", "plast_ci_v3"),
]


# ---------------------------------------------------------------- model
def fixed_point(I):
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(60):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        df = v ** 2 + (1 / b - 1) + 1e-6
        v = v - f / df
    return v, (v + a) / b


class World:
    """Everything stochastic about one seed, including its own connectome."""

    def _init_(self, seed, P):
        rng = np.random.default_rng(seed)
        self.c_i = rng.uniform(0.1, 1.0, M)
        self.p = rng.uniform(0.0, 1.0, M)
        self.u_cue = rng.random(M)
        conn = rng.random((M, M)) < P["conn_p"]
        np.fill_diagonal(conn, False)
        self.conn = conn
        self.conn_f = conn.astype(float)
        self.W0 = np.where(conn, P["w0_scale"] * rng.random((M, M)), 0.0)


def simulate_epoch(W_init, conn_f, I_base, I_ext, c_i, learn, P, T_total, seed, g_inh,
                   swr_mode="uniform", block_plast_in_swr=False):
    rng = np.random.default_rng(seed)             # noise
    swr_rng = np.random.default_rng(seed + 7919)  # SWR timing/permutation: same across conditions
    W = W_init.copy()
    n_steps = int(T_total / dt)
    v, w = fixed_point(I_base)
    s, x, y = np.zeros(M), np.zeros(M), np.zeros(M)
    I_inh = 0.0
    counts = np.zeros(M, int)
    dec_s = np.exp(-dt / 5.0)
    dec_pre = np.exp(-dt / P["tau_pre"])
    dec_post = np.exp(-dt / P["tau_post"])
    dec_R = np.exp(-dt / TAU_R)
    noise = rng.normal(0, P["sigma_v"] * np.sqrt(dt), (n_steps, M))
    swr_u = swr_rng.random(n_steps)               # pre-drawn so timing is identical across modes
    R, n_swr = 0.0, 0
    swr_w, swr_ready = np.ones(M), False
    tonic = P["g_swr"] * (P["p_swr"] / dt) * TAU_R if swr_mode == "tonic" else 0.0
    gate = c_i if P["plast_ci"] else np.ones(M)

    for k in range(n_steps):
        t = k * dt
        in_sws = learn and t > SENS_OFF

        if in_sws and not swr_ready:
            swr_ready = True
            if swr_mode in ("replay", "replay_shuffled") and counts.sum() > 0:
                wt = counts / counts.mean()
                wt = np.minimum(wt, P["replay_cap"])
                wt = wt / wt.mean()
                wt = (1 - P["replay_rho"]) + P["replay_rho"] * wt   # mean-preserving
                swr_w = swr_rng.permutation(wt) if swr_mode == "replay_shuffled" else wt

        if in_sws and swr_mode in ("uniform", "replay", "replay_shuffled") and swr_u[k] < P["p_swr"]:
            R += 1.0
            n_swr += 1
        else:
            R *= dec_R

        I_inh += dt * (-I_inh + g_inh * s.mean()) / 12.0
        I = I_base + W @ s - I_inh
        if SENS_ON <= t < SENS_OFF:
            I = I + I_ext
        if R > 0.01:
            I = I + P["g_swr"] * R * swr_w
        if in_sws and tonic:
            I = I + tonic

        vn = np.clip(v + dt * (v - v ** 3 / 3 - w + I) + noise[k], -3.0, 3.0)
        wn = w + dt * eps * (vn + a - b * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        if SENS_ON <= t < SENS_OFF:
            counts += spk

        s = s * dec_s + spk
        x_old = x * dec_pre            # pre trace BEFORE this step's spikes (causal)
        x = x_old + spk
        y = y * dec_post + spk

        if learn and spk.any() and not (block_plast_in_swr and R > 0.01):
            sp = spk.astype(float)
            x_burst = np.maximum(x_old - P["theta_pre"], 0.0)
            y_burst = np.maximum(y - P["theta_post"], 0.0)
            ltp = np.outer(gate * sp * y_burst, x_burst) * (P["wmax"] - W) * conn_f
            ltd = np.outer(gate * y_burst, sp) * W
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            W *= conn_f
    return counts, W, n_swr


# ---------------------------------------------------------------- analysis helpers
def block_mean(W, conn, A, B):
    """Mean weight over EXISTING synapses from B -> A (rows post, cols pre)."""
    sub = conn[np.ix_(A, B)]
    n = sub.sum()
    return W[np.ix_(A, B)][sub].mean() if n > 0 else np.nan


def match_controls(world, engram, cued_mask, caliper):
    """Greedy 1:1 nearest-neighbour matching on (p, c_i), without replacement."""
    treated = np.where(engram & (cued_mask == 0))[0]
    pool = list(np.where(~engram & (cued_mask == 0))[0])
    feats = np.stack([world.p, world.c_i], 1)
    pairs_t, pairs_c = [], []
    # hardest-to-match first (largest distance to nearest pool member)
    if len(treated) == 0 or len(pool) == 0:
        return np.array([], int), np.array([], int)
    d_near = [np.min(np.max(np.abs(feats[pool] - feats[i]), 1)) for i in treated]
    for i in treated[np.argsort(d_near)[::-1]]:
        if not pool:
            break
        d = np.max(np.abs(feats[pool] - feats[i]), 1)
        j = int(np.argmin(d))
        if d[j] <= caliper:
            pairs_t.append(i)
            pairs_c.append(pool.pop(j))
    return np.array(pairs_t, int), np.array(pairs_c, int)


def run_single(args):
    seed, cond_name, overrides = args
    spec = CONDITIONS[cond_name]
    P = {**PARAMS, **overrides, **spec.get("params", {})}
    world = World(seed, P)

    creb = np.full(M, world.c_i.mean()) if P["creb_flat"] else world.c_i
    I_base_enc = 0.10 + P["kc_enc"] * creb
    I_ext = P["s0"] * world.p

    cnt_enc, W_l, n_swr = simulate_epoch(
        world.W0, world.conn_f, I_base_enc, I_ext, world.c_i, True, P, T_ENC,
        seed * 1000 + 1, P["g_inh"], swr_mode=spec["swr_mode"],
        block_plast_in_swr=spec.get("block", False))
    assert not W_l[~world.conn].any(), "weights appeared on non-existent synapses"
    engram = cnt_enc >= FIRE_MIN

    cue = (world.u_cue < P["cue_f"]).astype(float)
    cnt_ret, _, _ = simulate_epoch(
        W_l, world.conn_f, np.full(M, P["I_ret"]), I_ext * cue, world.c_i, False, P, T_RET,
        seed * 1000 + 2, P["g_inh_ret"], swr_mode="none")
    fired = cnt_ret >= FIRE_MIN

    # CREB allocation (log-OR, 0.5 continuity correction)
    hi = world.c_i >= np.percentile(world.c_i, 66)
    lo = world.c_i <= np.percentile(world.c_i, 33)
    A, B = engram[hi].sum() + .5, (~engram[hi]).sum() + .5
    C, D = engram[lo].sum() + .5, (~engram[lo]).sum() + .5
    log_or = np.log((A / B) / (C / D))

    t_idx, c_idx = match_controls(world, engram, cue, P["match_caliper"])
    n_pairs = len(t_idx)
    nan = np.nan
    er = fired[t_idx].mean() if n_pairs else nan
    cr = fired[c_idx].mean() if n_pairs else nan
    cued_rec = fired[engram & (cue == 1)].mean() if (engram & (cue == 1)).any() else nan

    E = np.where(engram)[0]
    Nn = np.where(~engram)[0]
    d = lambda A_, B_: block_mean(W_l, world.conn, A_, B_) - block_mean(world.W0, world.conn, A_, B_)
    out = dict(
        n_engram=engram.sum(), log_or=log_or, n_swr=n_swr, n_pairs=n_pairs,
        engram_recall=er, control_recall=cr, recall_diff=er - cr if n_pairs else nan,
        cued_recall=cued_rec,
        dW_EE=d(E, E) if len(E) > 1 else nan,
        dW_EN=d(E, Nn) if len(E) and len(Nn) else nan,
        dW_NE=d(Nn, E) if len(E) and len(Nn) else nan,
        dW_NN=d(Nn, Nn) if len(Nn) > 1 else nan,
        dW_CC=d(c_idx, c_idx) if n_pairs > 1 else nan,
        ci_eng=world.c_i[t_idx].mean() if n_pairs else nan,
        ci_ctl=world.c_i[c_idx].mean() if n_pairs else nan,
        p_eng=world.p[t_idx].mean() if n_pairs else nan,
        p_ctl=world.p[c_idx].mean() if n_pairs else nan,
    )
    out["dW_sel"] = out["dW_EE"] - out["dW_CC"]   # selectivity vs matched-control null
    return seed, cond_name, out


# ---------------------------------------------------------------- stats
def tci(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return (arr.mean() if n else np.nan), np.nan, n
    return arr.mean(), stats.t.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n), n


def paired(a_, b_):
    a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
    ok = ~(np.isnan(a_) | np.isnan(b_))
    n = int(ok.sum())
    if n < 5:
        return dict(n=n, mean=np.nan, ci=np.nan, t=np.nan, p=np.nan, dz=np.nan)
    dlt = a_[ok] - b_[ok]
    t, p = stats.ttest_rel(a_[ok], b_[ok])
    sd = dlt.std(ddof=1)
    return dict(n=n, mean=dlt.mean(), ci=stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n),
                t=t, p=p, dz=dlt.mean() / sd if sd > 0 else np.nan)


def holm(ps):
    ps = np.array(ps, float)
    order = np.argsort(np.where(np.isnan(ps), 2, ps))
    adj, running = np.full(len(ps), np.nan), 0.0
    m = int((~np.isnan(ps)).sum())
    for rank, i in enumerate(order):
        if np.isnan(ps[i]):
            continue
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def run_all(seeds, conds, overrides, workers):
    jobs = [(s, c, overrides) for s in seeds for c in conds]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(run_single, jobs, chunksize=1)
    else:
        res = [run_single(j) for j in jobs]
    table = {c: {} for c in conds}
    for seed, c, out in res:
        table[c][seed] = out
    return table


def col(table, cond, key, seeds):
    return np.array([table[cond][s][key] for s in seeds], float)


def report(table, seeds, conds):
    n = len(seeds)
    print(f"\nseeds: {n} (first {seeds[0]}, last {seeds[-1]}) | independent connectome per seed")
    print(f"{'condition':18s} {'engram%':>8s} {'SWR/trial':>9s} {'pairs':>6s} {'dropped':>7s} "
          f"{'eng rec':>8s} {'ctl rec':>8s} {'diff ±CI':>15s} {'dW_EE':>8s} {'dW_CC':>8s} {'dW_sel ±CI':>17s}")
    for c in conds:
        eng = col(table, c, "n_engram", seeds) / M
        swr = col(table, c, "n_swr", seeds)
        pairs = col(table, c, "n_pairs", seeds)
        rd, rdci, nrd = tci(col(table, c, "recall_diff", seeds))
        sel, selci, _ = tci(col(table, c, "dW_sel", seeds))
        print(f"{c:18s} {eng.mean():8.1%} {swr.mean():9.1f} {pairs.mean():6.1f} {n - nrd:7d} "
              f"{np.nanmean(col(table, c, 'engram_recall', seeds)):8.1%} "
              f"{np.nanmean(col(table, c, 'control_recall', seeds)):8.1%} "
              f"{rd:+8.1%}±{rdci:5.1%} {np.nanmean(col(table, c, 'dW_EE', seeds)):8.5f} "
              f"{np.nanmean(col(table, c, 'dW_CC', seeds)):8.5f} {sel:+9.5f}±{selci:7.5f}")

    c0 = conds[0]
    print(f"\nBlock means in '{c0}' (existing synapses only): "
          f"EE {np.nanmean(col(table, c0, 'dW_EE', seeds)):.5f} | EN {np.nanmean(col(table, c0, 'dW_EN', seeds)):.5f} | "
          f"NE {np.nanmean(col(table, c0, 'dW_NE', seeds)):.5f} | NN {np.nanmean(col(table, c0, 'dW_NN', seeds)):.5f}")
    lo_, loci, _ = tci(col(table, c0, "log_or", seeds))
    print(f"CREB allocation log-OR (high vs low c_i, 0.5 correction): {lo_:.2f} ± {loci:.2f} "
          f"(OR {np.exp(lo_):.1f}); built in by kc_enc*c_i, not evidence of consolidation")
    print(f"Matching balance in '{c0}': c_i engram {np.nanmean(col(table, c0, 'ci_eng', seeds)):.3f} vs "
          f"ctl {np.nanmean(col(table, c0, 'ci_ctl', seeds)):.3f} | p {np.nanmean(col(table, c0, 'p_eng', seeds)):.3f} vs "
          f"{np.nanmean(col(table, c0, 'p_ctl', seeds)):.3f}")

    print("\nPaired contrasts (A - B, same seeds/noise/SWR times), Holm-corrected within each metric")
    for metric in ("recall_diff", "dW_sel"):
        rows, ps = [], []
        for A_, B_ in CONTRASTS:
            if A_ in table and B_ in table:
                r = paired(col(table, A_, metric, seeds), col(table, B_, metric, seeds))
                rows.append((A_, B_, r))
                ps.append(r["p"])
        adj = holm(ps)
        print(f"  [{metric}]")
        for (A_, B_, r), pa in zip(rows, adj):
            fmt = "{:+.1%}" if metric == "recall_diff" else "{:+.5f}"
            print(f"    {A_:18s} - {B_:18s} n={r['n']:3d} Δ={fmt.format(r['mean'])}±{abs(r['ci']):.4g} "
                  f"dz={r['dz']:+.2f} p={r['p']:.4f} p_holm={pa:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "test"], required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="param=value (tune mode only)")
    ap.add_argument("--conds", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--sweep", nargs="+", help="PARAM v1 v2 ... (tune mode only; runs uniform_swr + no_swr)")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=")
        assert k in PARAMS, f"unknown param {k}"
        overrides[k] = type(PARAMS[k])(float(v)) if not isinstance(PARAMS[k], bool) else v == "True"
    if args.mode == "test" and (overrides or args.sweep):
        sys.exit("Refusing: --mode test uses frozen PARAMS. Tune on TUNE_SEEDS, edit PARAMS, then test.")

    pool = TUNE_SEEDS if args.mode == "tune" else TEST_SEEDS
    seeds = list(pool)[:args.n]
    print(f"=== v4 | mode={args.mode} | T_enc={T_ENC}, T_ret={T_RET}, FIRE_MIN={FIRE_MIN} | "
          f"time in dimensionless FHN units ===")
    print({**PARAMS, **overrides})

    if args.sweep:
        name, vals = args.sweep[0], args.sweep[1:]
        assert name in PARAMS
        for v in vals:
            ov = {**overrides, name: type(PARAMS[name])(float(v))}
            tb = run_all(seeds, ["uniform_swr", "no_swr"], ov, args.workers)
            r = paired(col(tb, "uniform_swr", "recall_diff", seeds), np.zeros(len(seeds)))
            sel = paired(col(tb, "uniform_swr", "dW_sel", seeds), col(tb, "no_swr", "dW_sel", seeds))
            print(f"{name}={v:>6s}: engram {np.mean(col(tb, 'uniform_swr', 'n_engram', seeds)) / M:6.1%} | "
                  f"recall diff {r['mean']:+.1%} | dW_sel(swr - no_swr) {sel['mean']:+.5f} (p={sel['p']:.3f})")
        return

    table = run_all(seeds, args.conds, overrides, args.workers)
    report(table, seeds, args.conds)
    print("\nInterpretation guide: a consolidation claim needs uniform_swr > no_swr AND > swr_no_plasticity "
          "AND > tonic_matched in recall_diff and dW_sel; a content claim needs replay_swr > replay_shuffled; "
          "if creb_off or plast_ci_v3 explains the effect, it is excitability, not association.")


if __name__ == "__main__":
    main()