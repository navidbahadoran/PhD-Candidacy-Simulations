"""
sim0_mp_law.py — Marchenko–Pastur (MP) spectral law demo (Chapter 2).

Generates X in R^{p x n} with i.i.d. N(0,1), computes S = (1/n) X X^T,
plots histogram of eigenvalues vs. MP density with parameter gamma = p/n.
Also checks the extreme eigenvalues vs. MP edges [(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2].
"""
import numpy as np
import matplotlib.pyplot as plt

def mp_pdf(x, gamma):
    # MP density for variance 1 noise:
    # f(x) = (1/(2πγx)) sqrt((b-x)(x-a)) on [a,b], a=(1-√γ)^2, b=(1+√γ)^2
    a = (1.0 - np.sqrt(gamma))**2
    b = (1.0 + np.sqrt(gamma))**2
    y = np.zeros_like(x, dtype=float)
    mask = (x >= a) & (x <= b)
    y[mask] = np.sqrt((b - x[mask]) * (x[mask] - a)) / (2.0 * np.pi * gamma * x[mask])
    return y, a, b

def run(p=2000, n=1000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, n))
    S = (X @ X.T) / n
    evals = np.linalg.eigvalsh(S)

    gamma = p / n
    a = (1 - np.sqrt(gamma))**2
    b = (1 + np.sqrt(gamma))**2

    print(f"[MP] p={p}, n={n}, gamma={gamma:.3f}")
    print(f"     min eval={evals.min():.3f} (theory a={a:.3f})")
    print(f"     max eval={evals.max():.3f} (theory b={b:.3f})")

    # Plot histogram vs MP density
    import os
    os.makedirs("figures", exist_ok=True)
    xs = np.linspace(max(a-0.2, 1e-6), b+0.2, 500)
    pdf, a, b = mp_pdf(xs, gamma)

    plt.figure(figsize=(6,4))
    plt.hist(evals, bins=80, density=True, alpha=0.5, label="empirical eigs")
    plt.plot(xs, pdf, lw=2, label="MP density")
    plt.axvline(a, ls="--", lw=1, label="edge a")
    plt.axvline(b, ls="--", lw=1, label="edge b")
    plt.title(f"MP law: p={p}, n={n}, γ={gamma:.2f}")
    plt.xlabel("eigenvalue"); plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/sim0_mp_law.png", dpi=120)
    plt.close()

if __name__ == "__main__":
    # A couple of aspect ratios
    run(p=2000, n=1000, seed=123)  # γ=2.0
    run(p=1500, n=3000, seed=124)  # γ=0.5
