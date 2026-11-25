"""
Externally Driven SWR Pulses, Global WTA Inhibition, and Burst-Threshold Hebbian Plasticity
Final Decoupled Headless Ensemble Script for Experimental Validation.
"""

import numpy as np
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Fixed Model Topology & Constants
M = 144
a, b, eps = 0.7, 0.8, 0.08
dt = 0.1 
vH = -np.sqrt(1 - eps * b)
I_HOPF = (vH + a) / b - vH + vH ** 3 / 3
FIRE_MIN = 2

np.random.seed(42)
conn_mask = np.random.rand(M, M) < 0.15
np.fill_diagonal(conn_mask, 0)
W0 = np.where(conn_mask, 0.008 * np.random.rand(M, M), 0.0)

# Tuned parameters for decoupled excitability and strict burst-thresholding
PARAMS = dict(
    kc_enc=0.18,      # CREB gain during encoding (active)
    kc_ret=0.00,      # CREB gain during retrieval (decayed to 0)
    s0=0.22,          # Stimulus drive
    p_swr=0.004,      # SWR Poisson rate
    g_swr=0.28,       # SWR replay drive (needs to be strong for tight bursts)
    g_inh=2.8,        # PV+ WTA Inhibition
    eta=0.55,         # Plasticity rate (increased to compensate for strict threshold)
    ratio=0.45,       # LTD/LTP ratio
    wmax=0.40,        # Soft-bound weight cap
    theta_p=0.95,     # Raised slightly to strictly gate out sensory-only plasticity
    cue_f=0.40,       # Partial cue fraction
    sigma_v=0.04      # Intrinsic membrane voltage noise
)

def fixed_point(I):
    I = np.atleast_1d(np.asarray(I, float))
    v = np.full_like(I, -1.0)
    for _ in range(40):
        f = v ** 3 / 3 + (1 / b - 1) * v + (a / b - I)
        v = v - f / (v ** 2 + (1 / b - 1))
    return v, (v + a) / b

class World:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.c_i = self.rng.uniform(0.1, 1.0, M)
        self.p = self.rng.uniform(0.0, 1.0, M)
        self.u_cue = self.rng.random(M)

def simulate_epoch(W_init, I_base, I_ext, c_i, learn, P, T_total):
    rng = np.random.default_rng()
    W = W_init.copy()
    n_steps = int(T_total / dt)
    
    v, w = fixed_point(I_base)
    s, x, y = np.zeros(M), np.zeros(M), np.zeros(M)
    I_inh = 0.0
    counts = np.zeros(M, int)
    
    dec_s = np.exp(-dt / 5.0)
    dec_t = np.exp(-dt / 25.0)
    dec_R = np.exp(-dt / 15.0) 
    
    noise = rng.normal(0, P["sigma_v"] * np.sqrt(dt), (n_steps, M))
    R = 0.0
    
    for k in range(n_steps):
        t = k * dt
        
        if learn and t > 110.0 and rng.random() < P["p_swr"]:
            R += 1.0
        else:
            R = R * dec_R
        
        I_inh += dt * (-I_inh + P["g_inh"] * s.mean()) / 12.0
        
        I = I_base + W @ s - I_inh
        if 20.0 <= t < 110.0:
            I += I_ext
        if R > 0.01:
            I += P["g_swr"] * R
            
        vn = v + dt * (v - v ** 3 / 3 - w + I) + noise[k]
        vn = np.clip(vn, -3.0, 3.0) 
        wn = w + dt * eps * (vn + a - b * w)
        spk = (vn > 0) & (v <= 0)
        v, w = vn, wn
        
        if 20.0 <= t < 110.0: 
            counts += spk
            
        s = s * dec_s + spk
        x = x * dec_t + spk
        y = y * dec_t + spk

        if learn and spk.any():
            sp = spk.astype(float)
            x_burst = np.maximum(x - P["theta_p"], 0.0)
            y_burst = np.maximum(y - P["theta_p"], 0.0)
            
            ltp = np.outer(c_i * sp, x_burst) * (P["wmax"] - W)
            ltd = np.outer(c_i * y_burst, sp) * W
            
            W += P["eta"] * (ltp - P["ratio"] * ltd)
            np.clip(W, 0.0, P["wmax"], out=W)
            np.fill_diagonal(W, 0.0)
            
    return counts, W

def run_single_experiment(seed, override_params=None):
    P = PARAMS.copy()
    if override_params:
        P.update(override_params)
        
    world = World(seed)
    
    # Decouple encoding excitability from retrieval excitability
    I_base_enc = 0.10 + P["kc_enc"] * world.c_i
    I_base_ret = 0.10 + P["kc_ret"] * world.c_i  
    
    I_ext = P["s0"] * world.p
    
    # 1. Encoding (High CREB)
    cnt_enc, W_learned = simulate_epoch(W0, I_base_enc, I_ext, world.c_i, True, P, 240.0)
    engram = cnt_enc >= FIRE_MIN
    n_engram = int(engram.sum())
    
    # 2. Retrieval (CREB decayed, relying purely on synapses)
    mask = (world.u_cue < P["cue_f"]).astype(float)
    cnt_ret, _ = simulate_epoch(W_learned, I_base_ret, I_ext * mask, world.c_i, False, P, 150.0)
    fired = cnt_ret >= FIRE_MIN
    
    # Metrics
    creb_thresh_high = np.percentile(world.c_i, 66)
    creb_thresh_low = np.percentile(world.c_i, 33)
    high_creb_rate = engram[world.c_i >= creb_thresh_high].mean() if (world.c_i >= creb_thresh_high).any() else 0.0
    low_creb_rate = engram[world.c_i <= creb_thresh_low].mean() if (world.c_i <= creb_thresh_low).any() else 0.0
    
    p1 = max(1e-6, min(1.0 - 1e-6, high_creb_rate))
    p2 = max(1e-6, min(1.0 - 1e-6, low_creb_rate))
    odds_ratio = (p1 / (1 - p1)) / (p2 / (1 - p2))
    
    uncued_engram = engram & (mask == 0)
    engram_recall_rate = fired[uncued_engram].mean() if uncued_engram.any() else np.nan
    
    mean_engram_p = world.p[engram].mean() if engram.any() else 0.5
    matched_non_engram = (~engram) & (mask == 0) & (world.p >= mean_engram_p - 0.1) & (world.p <= mean_engram_p + 0.1)
    control_recall_rate = fired[matched_non_engram].mean() if matched_non_engram.any() else np.nan
    
    dW = W_learned[np.ix_(engram, engram)].mean() - W0[np.ix_(engram, engram)].mean() if n_engram > 1 else 0.0

    return {
        "n_engram": n_engram,
        "odds_ratio": odds_ratio,
        "engram_recall": engram_recall_rate,
        "control_recall": control_recall_rate,
        "dW": dW
    }

def run_ensemble(n_seeds=50, **kwargs):
    results = {"n_engram": [], "odds_ratio": [], "engram_recall": [], "control_recall": [], "dW": []}
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_experiment, seed, kwargs) for seed in range(n_seeds)]
        for f in futures:
            res = f.result()
            for k in results.keys():
                results[k].append(res[k])
                
    metrics = {}
    for k, v in results.items():
        arr = np.array(v)
        arr = arr[~np.isnan(arr)]
        mean = np.mean(arr) if len(arr) > 0 else 0
        ci = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 0 else 0
        metrics[k] = (mean, ci)
        
    return metrics

if __name__ == "__main__":
    print("Running Baseline Ensemble (N=50)...")
    base = run_ensemble(50)
    print(f"Engram Size: {base['n_engram'][0]:.1f} ± {base['n_engram'][1]:.1f}")
    print(f"CREB Allocation OR (Han 2007): {base['odds_ratio'][0]:.2f} ± {base['odds_ratio'][1]:.2f}")
    print(f"Engram Recall: {base['engram_recall'][0]:.1%} ± {base['engram_recall'][1]:.1%}")
    print(f"Matched Control Recall: {base['control_recall'][0]:.1%} ± {base['control_recall'][1]:.1%}")
    
    print("\nRunning SWR-Ablation (Girardeau 2009)...")
    swr_off = run_ensemble(50, p_swr=0.0)
    print(f"Base dW: {base['dW'][0]:.4f} | SWR-Off dW: {swr_off['dW'][0]:.4f}")
    
    print("\nRunning Inhibition Sweep (Stefanelli 2016)...")
    low_inh = run_ensemble(30, g_inh=0.2)
    high_inh = run_ensemble(30, g_inh=6.0)
    print(f"Engram Size (Low Inh): {low_inh['n_engram'][0]:.1f} | Engram Size (High Inh): {high_inh['n_engram'][0]:.1f}")