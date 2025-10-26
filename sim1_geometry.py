import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_T(lam, n=20, seed=0):
    rng = np.random.default_rng(seed)
    p = lam.size
    Z = rng.standard_normal((p, n))
    X = (np.sqrt(lam)[:, None] * Z)
    SD = (X.T @ X) / n
    T = (n / lam.sum()) * SD
    return T

def run(p=2000, n=20, alphas=(), seed=0, prefix="figures/sim1_geometry"):
    lam = np.ones(p)
    for idx, a in enumerate(alphas):
        lam[idx] = p**a
    T = simulate_T(lam, n=n, seed=seed)
    os.makedirs("figures", exist_ok=True)

    diag = np.diag(T)
    off = T - np.diag(diag)
    off_vals = off[np.triu_indices_from(off, k=1)]

    evals = np.linalg.eigvalsh(T)
    evals.sort()

    # Diagonals
    plt.figure(figsize=(6.0, 3.8))
    plt.hist(diag, bins=40, density=True, alpha=0.7, label="diagonals")
    plt.axvline(1.0, ls="--", lw=1, label="1")
    plt.legend(frameon=False)
    plt.title(f"Rescaled dual diagonals (p={p}, n={n})")
    plt.xlabel("value"); plt.ylabel("density")
    plt.tight_layout(); plt.savefig(prefix + "_diag.pdf", bbox_inches="tight"); plt.close()

    # Off-diagonals
    plt.figure(figsize=(6.0, 3.8))
    plt.hist(off_vals, bins=60, density=True, alpha=0.7, label="off-diagonals")
    plt.axvline(0.0, ls="--", lw=1, label="0")
    plt.legend(frameon=False)
    plt.title(f"Rescaled dual off-diagonals (p={p}, n={n})")
    plt.xlabel("value"); plt.ylabel("density")
    plt.tight_layout(); plt.savefig(prefix + "_off.pdf", bbox_inches="tight"); plt.close()

    # QQ plot
    target = np.ones_like(evals)
    plt.figure(figsize=(5.0, 5.0))
    
    # Empirical points
    plt.scatter(target, evals, s=25, color="tab:blue", alpha=0.8, label="Empirical eigenvalues")
    
    # Theoretical 45° line
    plt.plot([0.2, 1.8], [0.2, 1.8], "--", color="tab:orange", lw=1.5, label="y = x (theory)")

    plt.xlabel("Theoretical value (1)")
    plt.ylabel("Empirical eigenvalue")
    plt.title(f"QQ plot of eigenvalues (p={p}, n={n})")
    plt.legend(frameon=False, loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(prefix + "_qq.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run(p=2000, n=20, alphas=(), seed=0, prefix="figures/sim1_geometry_diffuse")
    run(p=2000, n=20, alphas=(0.8, 0.6), seed=1, prefix="figures/sim1_geometry_spiked")
