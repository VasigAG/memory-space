"""
Headless Image Exporter for Presentation Assets.
Run this script to generate high-resolution PNGs of the model's phase space 
and temporal traces with transparent backgrounds for PowerPoint/Keynote.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Set publication plotting style
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none",
    "axes.edgecolor": "#8b949e", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#30363d",
    "font.family": "sans-serif", "font.size": 14
})

# Minimal FHN parameters to generate the curves
a, b, eps = 0.7, 0.8, 0.08
vH = -np.sqrt(1 - eps * b)
I_HOPF = (vH + a) / b - vH + vH ** 3 / 3

def export_phase_space():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Shade unstable zone
    ax.axvspan(vH, -vH, color="#f85149", alpha=0.2, label="Unstable Zone")
    ax.axvline(vH, color="#f85149", ls=":", lw=2)
    
    vv = np.linspace(-2.5, 2.5, 400)
    
    # Nullclines
    ax.plot(vv, (vv + a) / b, color="#ff7b72", lw=3, label="w-nullcline (Recovery)")
    
    # Before Learning (Rest)
    I_rest = 0.15
    w_rest = vv - vv**3/3 + I_rest
    ax.plot(vv, w_rest, color="#8b949e", ls="--", lw=2, label="Resting State")
    
    # After Learning (Pattern Completion)
    I_learned = I_rest + 0.25
    w_learned = vv - vv**3/3 + I_learned
    ax.plot(vv, w_learned, color="#00e5ff", ls="-", lw=3, label="With Synaptic Input")
    
    ax.set_title("Phase Space: Bifurcation Shift via Synaptic Input", pad=15)
    ax.set_xlabel("Membrane Potential (V)")
    ax.set_ylabel("Recovery Variable (w)")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.0, 2.2)
    ax.legend(frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("presentation_phase_space.png", dpi=300, transparent=True)
    print("Saved presentation_phase_space.png")
    plt.close()

def export_trace_mockup():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    t = np.linspace(0, 300, 1000)
    
    # Mockup SWR Envelope
    swr = np.zeros_like(t)
    swr[400:600] = np.exp(-(t[400:600] - t[400]) / 20.0)
    
    ax.fill_between(t, 0, swr, color="#a371f7", alpha=0.4, label="SWR Envelope R(t)")
    ax.plot(t, swr, color="#a371f7", lw=2)
    
    ax.set_title("Stochastic SWR Nucleation & Exponential Decay", pad=15)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([])
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig("presentation_swr_envelope.png", dpi=300, transparent=True)
    print("Saved presentation_swr_envelope.png")
    plt.close()

if __name__ == "__main__":
    print("Exporting presentation assets...")
    export_phase_space()
    export_trace_mockup()
    print("Done!")