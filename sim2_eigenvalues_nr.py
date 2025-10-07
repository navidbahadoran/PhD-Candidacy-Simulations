import numpy as np, matplotlib.pyplot as plt, os

def dual_cov(X):
    n = X.shape[1]; return (X.T @ X)/n

def nr_lambda(hat_lams, j):
    n = len(hat_lams)
    bulk_mean = (hat_lams[j+1:].sum())/(n-j) if j < n-1 else 0.0
    return hat_lams[j] - bulk_mean

def run(p=3000, n=60, alphas=(0.8,0.6), kplot=10, seed=0, prefix="figures/sim2_eigs"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p)
    for idx, a in enumerate(alphas): lam[idx] = p**a
    Z = rng.standard_normal((p, n)); X = (np.sqrt(lam)[:,None] * Z)
    SD = dual_cov(X)
    hat_lams = np.linalg.eigvalsh(SD)[::-1]
    tilde = np.array([nr_lambda(hat_lams, j) for j in range(n-1)] + [0.0])
    mu = lam[len(alphas):].mean() if len(alphas)<p else 0.0
    true_approx = np.array([lam[j] + mu if j < len(alphas) else mu for j in range(n)])
    os.makedirs("figures", exist_ok=True)
    jidx = np.arange(1, kplot+1)
    plt.figure(figsize=(6.4,4.2))
    plt.plot(jidx, hat_lams[:kplot], "o-", label=r"$\hat\lambda_j$ (naïve)")
    plt.plot(jidx, tilde[:kplot], "x-", label=r"$\tilde\lambda_j$ (NR)")
    plt.plot(jidx, true_approx[:kplot], "--", label="truth (approx)")
    plt.xlabel("component j"); plt.ylabel("eigenvalue"); plt.legend(frameon=False)
    plt.title(f"NR vs naïve eigenvalues (p={p}, n={n})")
    plt.tight_layout(); plt.savefig(prefix+".pdf", bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
