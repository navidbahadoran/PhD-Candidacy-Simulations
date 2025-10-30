import numpy as np, matplotlib.pyplot as plt, os

def dual_cov(X):
    n = X.shape[1]
    return (X.T @ X) / n

def nr_lambda_vec(hat_lams):
    n = len(hat_lams)
    out = np.zeros_like(hat_lams)
    for j in range(n-1):
        out[j] = hat_lams[j] - hat_lams[j+1:].sum()/(n-j)
    return out

def run(p=3000, alpha=0.7, n_vals=(20,40,80,120,200), reps=30, seed=1,
        fname="figures/sim4_pc_scores.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p); lam[0] = p**alpha

    nmse_naive, nmse_nr = [], []

    for n in n_vals:
        errs_naive, errs_nr = [], []
        for _ in range(reps):
            Z = rng.standard_normal((p, n))
            X = (np.sqrt(lam)[:, None] * Z)

            SD = dual_cov(X)
            evals, U = np.linalg.eigh(SD)
            idx = np.argsort(evals)[::-1]; evals = evals[idx]; U = U[:, idx]

            # true PC1 score (population direction e1)
            s_true = np.sqrt(lam[0]) * Z[0, :]

            # naïve score using sample dual eigens (PC1)
            s_hat = np.sqrt(n * evals[0]) * U[:, 0]

            # NR score using noise-reduced dual eigenvalue
            til = nr_lambda_vec(evals)
            s_til = np.sqrt(n * max(til[0], 1e-12)) * U[:, 0]

            errs_naive.append(np.mean((s_hat - s_true)**2) / lam[0])
            errs_nr.append(np.mean((s_til - s_true)**2) / lam[0])

        nmse_naive.append(float(np.mean(errs_naive)))
        nmse_nr.append(float(np.mean(errs_nr)))

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(n_vals, nmse_naive, "o-", label="Sample PC score (naive)")
    ax.plot(n_vals, nmse_nr, "x-", label="NR PC score")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("NMSE of PC1 score")
    ax.set_title(f"PC1 score NMSE vs $n$ (p={p}, α={alpha:.2f})")
    ax.legend(frameon=False, loc="upper right")

    # ---- Footer summary under the x-axis (no plot compression) ----
    spikes = 1
    bulk_mean = lam[1:].mean() if p > 1 else 0.0
    footer_text = (
        rf"{spikes} spike | "
        rf"$\alpha_1$={alpha:.2f}, $\lambda_1=p^{{\alpha_1}}$={lam[0]:.2f} | "
        rf"bulk mean $\mu \approx {bulk_mean:.2f}$ | reps={reps}"
    )
    fig.text(0.5, -0.05, footer_text, ha='center', va='top', fontsize=9)

    fig.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run()
