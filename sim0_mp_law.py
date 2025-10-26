import numpy as np, matplotlib.pyplot as plt, os

def mp_pdf(x, gamma):
    a = (1.0 - np.sqrt(gamma))**2
    b = (1.0 + np.sqrt(gamma))**2
    y = np.zeros_like(x, dtype=float)
    mask = (x >= a) & (x <= b)
    y[mask] = np.sqrt((b - x[mask]) * (x[mask] - a)) / (2.0 * np.pi * gamma * x[mask])
    if gamma>1:
        y[0]=1-(1/gamma)
    return y, a, b

def run(p=2000, n=1000, seed=123, fname="./figures/sim0_mp_law.pdf"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, n))
    S = (X @ X.T) / n
    evals = np.linalg.eigvalsh(S)
    gamma = p / n
    a = (1 - np.sqrt(gamma))**2
    b = (1 + np.sqrt(gamma))**2
    os.makedirs("figures", exist_ok=True)
    xs = np.linspace(max(a*0.8, 1e-6), b*1.2, 500)
    pdf, a, b = mp_pdf(xs, gamma)
    plt.figure(figsize=(6.2,4.2))
    plt.hist(evals, bins=80, density=True, alpha=0.5, label="empirical eigs")
    plt.plot(xs, pdf, lw=2, label="MP density")
    plt.axvline(a, ls="--", lw=1, label="edge a")
    plt.axvline(b, ls="--", lw=1, label="edge b")
    plt.title(f"MP law: p={p}, n={n}, γ={gamma:.2f}")
    plt.xlabel("eigenvalue"); plt.ylabel("density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run(p=2000, n=1000, seed=123, fname="./figures/sim0_mp_law_gamma2.pdf")
    run(p=1500, n=3000, seed=124, fname="./figures/sim0_mp_law_gamma0p5.pdf")
