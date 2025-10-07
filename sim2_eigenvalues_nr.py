"""
sim2_eigenvalues_nr.py — Eigenvalue bias and NR correction (Section 3.5).

Compares naive sample eigenvalues of S_D with noise-reduced tilde λ_j.
Varies spike exponent α and sample size n.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import SpikedModel, generate_data, dual_cov, nr_eigenvalues, set_seed

def one_run(p=5000, n=50, m=2, alphas=(0.8, 0.6), a=(1.0, 1.0), bulk=1.0, kind="gaussian", seed=1):
    set_seed(seed)
    model = SpikedModel(p=p, n=n, m=m, alphas=np.array(alphas), a=np.array(a), c_bulk=bulk)
    X, lam, H = generate_data(model, kind=kind)
    SD = dual_cov(X)
    evals, U = np.linalg.eigh(SD)
    evals = evals[::-1]

    tilde = []
    n_e = evals.size
    for j in range(n_e-1):
        mu_hat = (evals[j+1:].sum()) / (n_e - (j+1))
        tilde.append(max(evals[j] - mu_hat, 0.0))
    tilde.append(max(evals[-1], 0.0))
    tilde = np.array(tilde)

    lam_sig = lam[:m]
    print(f"[sim2] p={p} n={n} alphas={alphas} lam1={lam_sig[0]:.2e}")
    for j in range(m):
        print(f"  j={j+1}: naive_ratio={evals[j]/lam_sig[j]:.3f}, NR_ratio={tilde[j]/lam_sig[j]:.3f}")

    # Plot top-10 naive vs NR
    import os
    os.makedirs("figures", exist_ok=True)
    k = min(10, n)
    plt.figure()
    x = np.arange(1, k+1)
    plt.plot(x, evals[:k], marker="o", label="naïve eigs")
    plt.plot(x, tilde[:k], marker="x", label="NR eigs")
    plt.title("Top eigenvalues: naïve vs noise-reduced")
    plt.xlabel("index"); plt.ylabel("value")
    plt.legend()
    plt.savefig("figures/sim2_eigs.png", dpi=120, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    one_run(p=5000, n=50, m=2, alphas=(0.8, 0.6), a=(1.0, 1.0), kind="gaussian", seed=7)
    one_run(p=5000, n=30, m=2, alphas=(0.6, 0.55), a=(1.0, 1.0), kind="gaussian", seed=8)
