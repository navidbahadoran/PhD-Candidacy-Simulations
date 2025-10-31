import numpy as np, matplotlib.pyplot as plt, os

def nr_lambda_vec(hat_lams):
    n = len(hat_lams)
    out = np.zeros_like(hat_lams)
    for j in range(n-1):
        out[j] = hat_lams[j] - hat_lams[j+1:].sum()/(n-j)
    return out

def run_scores_vs_p(
    p_vals=(800,1000,1200,1400,1600,1800),
    n_train=20, n_test=20, alpha=0.80, reps=300, seed=1,
    trace_normalize=True, qr_orth=True,
    fname="figures/sim4_pc_scores.pdf"
):
    rng = np.random.default_rng(seed)
    means_naive, means_nr, ses_naive, ses_nr = [], [], [], []

    for p in p_vals:
        lam = np.ones(p); lam[0] = p**alpha
        lam1 = lam[0]

        # trace normalization so avg eigenvalue ~ 1 across p
        c = 1.0 / np.sqrt(lam.sum()) if trace_normalize else 1.0

        mse_naive, mse_nr = [], []

        for _ in range(reps):
            # -------- TRAIN --------
            Ztr = rng.standard_normal((p, n_train))
            Xtr = c * (np.sqrt(lam)[:, None] * Ztr)

            # dual eigens (sorted ↓)
            SD = (Xtr.T @ Xtr) / n_train
            evals, U = np.linalg.eigh(SD)
            idx = np.argsort(evals)[::-1]
            evals = np.maximum(evals[idx], 1e-12)
            U     = U[:, idx]

            # left singular directions (columns)
            H = (Xtr @ U) / np.sqrt(n_train*evals)
            if qr_orth:
                H, _ = np.linalg.qr(H)

            tilde = nr_lambda_vec(evals)


            # choose top indices
            j = 0


            v_naive = H[:, j]
            v_nr    = H[:, j]

            # -------- TEST (fresh) --------
            Zte = rng.standard_normal((p, n_test))
            Xte = c * (np.sqrt(lam)[:, None] * Zte)
            SD_te = (Xte.T @ Xte) / n_test
            evals_te, U_te = np.linalg.eigh(SD_te)

            # true score under e1
            s_true = c * np.sqrt(lam1) * Zte[0, :]

            # naive test score = v^T X_te
            s_hat_naive = v_naive @ Xte

            # this is the crucial scaling that the paper uses
            s_hat_nr = np.sqrt(max(tilde[j], 1e-12) / evals[j]) * (v_nr @ Xte)


            mse_naive.append(np.mean((s_hat_naive - s_true)**2) / lam1)
            mse_nr.append(   np.mean((s_hat_nr   - s_true)**2) / lam1)

        mN, mR = float(np.mean(mse_naive)), float(np.mean(mse_nr))
        sN = float(np.std(mse_naive, ddof=1)/np.sqrt(reps))
        sR = float(np.std(mse_nr,    ddof=1)/np.sqrt(reps))
        means_naive.append(mN); ses_naive.append(sN)
        means_nr.append(mR);   ses_nr.append(sR)

    # ---- plot ----
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.errorbar(p_vals, means_naive, yerr=ses_naive, fmt="o-", capsize=3, label="Naive PC1 score")
    ax.errorbar(p_vals, means_nr,    yerr=ses_nr,    fmt="s--", capsize=3, label="NR PC1 score")
    ax.set_xlabel("dimension $p$")
    ax.set_ylabel("NMSE of PC1 score")
    tn = ", trace-normalized" if trace_normalize else ""
    ax.set_title(rf"PC1 score NMSE vs p (fixed $n={n_train}$, $\alpha_1={alpha:.2f}$)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    # ---- Footer summary under the x-axis (no plot compression) ----
    spikes = 1
    bulk_mean = lam[1:].mean()
    footer_text = (
        rf"{spikes} spike | "
        rf"$\alpha_1$={alpha:.2f}, $\lambda_1=p^{{\alpha_1}}$={lam[0]:.2f} | "
        rf"bulk mean $\mu \approx {bulk_mean:.2f}$"
    )
    fig.text(0.5, -0.05, footer_text, ha='center', va='top', fontsize=9)

    fig.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run_scores_vs_p()
