"""
sim1_geometry.py — Geometric representations (Section 3.4).

- Verifies: (n / sum λ_i) * S_D ≈ I_n under diffuse spectrum + weak dependence.
- Non-mixing proxy: off-diagonals vanish but diagonals need not concentrate at 1.

Saves simple histograms of diagonal/off-diagonal entries after scaling.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import SpikedModel, generate_data, dual_cov, set_seed

def run(p=2000, n=20, bulk_c=1.0, sp_m=0, alphas=None, a=None, kind="gaussian", rho_mix=None, seed=2025):
    set_seed(seed)
    m = sp_m
    if alphas is None: alphas = np.array([])
    if a is None: a = np.array([])
    model = SpikedModel(p=p, n=n, m=m, alphas=alphas, a=a, c_bulk=bulk_c)
    X, lam, H = generate_data(model, kind=kind, rho_mix=rho_mix)

    SD = dual_cov(X)
    scale = n / lam.sum()
    T = scale * SD  # should be ~ I_n in mixing case

    diags = np.diag(T)
    off = T - np.diag(diags)

    print(f"[sim1] p={p}, n={n}, kind={kind}, rho_mix={rho_mix}")
    print(f"  mean(diag)={diags.mean():.3f}, std(diag)={diags.std():.3f}")
    print(f"  max|off|={np.max(np.abs(off)):.3e}, Fro(off)={np.linalg.norm(off, 'fro'):.3f}")

    # Plots
    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    plt.hist(diags, bins=20, density=True)
    plt.title("Histogram of diag entries of (n/sum λ) S_D")
    plt.xlabel("value"); plt.ylabel("density")
    plt.savefig("figures/sim1_diag_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(off.flatten(), bins=30, density=True)
    plt.title("Histogram of off-diagonals of (n/sum λ) S_D")
    plt.xlabel("value"); plt.ylabel("density")
    plt.savefig("figures/sim1_off_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Mixing (Gaussian): expect near-identity
    run(p=2000, n=20, sp_m=0, kind="gaussian", rho_mix=None, seed=2025)
    # Non-mixing proxy via dependent squares: diagonals vary more
    run(p=2000, n=20, sp_m=0, kind="t", rho_mix=0.8, seed=2026)
