import numpy as np, matplotlib.pyplot as plt, os

def run(p=2000, n_vals=(20,40,80,120), alphas=(0.8,0.6), reps=10, seed=0,
        fname="figures/sim5_inverse.pdf", qr_orthonormalize=False):
    rng = np.random.default_rng(seed)

    # population spectrum (spikes + 1's)
    lam = np.ones(p)
    for i, a in enumerate(alphas):
        lam[i] = p**a
    bulk_mean = lam[len(alphas):].mean() if len(alphas) < p else 0.0

    errs_ridge_span, errs_nr_span = [], []
    eps = 1e-12

    for n in n_vals:
        er_r_s, er_nr_s = [], []

        for _ in range(reps):
            # --- simulate data ---
            Z = rng.standard_normal((p, n))
            X = (np.sqrt(lam)[:, None] * Z)

            # primal & dual sample covariances
            S  = (X @ X.T)/n
            SD = (X.T @ X)/n

            # ridge level δ = tr(S)/n
            trS   = float(np.trace(S))
            delta = max(trS / n, eps)

            # --- dual eigensystem (sorted ↓) ---
            evals, U = np.linalg.eigh(SD)
            idx = np.argsort(evals)[::-1]
            evals = np.maximum(evals[idx], eps)   # guard tiny negatives to eps
            U     = U[:, idx]

            # left singular directions H ≈ orthonormal columns
            H = (X @ U) / np.sqrt(n*evals + eps)   # p×n
            if qr_orthonormalize:
                H, _ = np.linalg.qr(H)             # enforce H^T H ≈ I_n numerically

            # projector onto span(X)
            P = H @ H.T

            # ---- Ridge inverse (S + δ I)^{-1} ----
            Si = np.linalg.inv(S + delta*np.eye(p))

            # SPAN whitening error: || H^T Λ^{1/2} W Λ^{1/2} H - I ||_F / √n
            LhalfH = np.sqrt(lam)[:, None] * H         # p×n
            Asp_r  = LhalfH.T @ Si @ LhalfH            # n×n
            er_r_s.append(np.linalg.norm(Asp_r - np.eye(n), 'fro') / np.sqrt(n))

            # ---- NR inverse: W_nr = H diag(1/λ̄) H^T + (I-P)*(1/ω) ----
            # Noise-reduced dual eigenvalues: subtract tail mean
            tilde = np.zeros_like(evals)
            for j in range(n-1):
                tilde[j] = evals[j] - evals[j+1:].sum()/(n-j)

            # floor ω = min(tr(S)/(√p n^{1/4}), δ)
            omega   = max(min(trS / (np.sqrt(p) * (n**0.25)), delta), eps)
            lam_bar = np.maximum(tilde, omega)

            W_nr  = H @ np.diag(1.0/lam_bar) @ H.T + (np.eye(p) - P) * (1.0/omega)

            Asp_nr = LhalfH.T @ W_nr @ LhalfH
            er_nr_s.append(np.linalg.norm(Asp_nr - np.eye(n), 'fro') / np.sqrt(n))

        errs_ridge_span.append(float(np.mean(er_r_s)))
        errs_nr_span.append(float(np.mean(er_nr_s)))

    # --------- Plot ----------
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(n_vals, errs_ridge_span, "x--", label="Ridge (span whitening error)")
    ax.plot(n_vals, errs_nr_span,    "o--", label="NR (span whitening error)")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"Population Error (relative to $\Sigma_p^{-1}$)")
    ax.set_title(f"Inverse estimator (span) error vs $n$ (p={p}, δ=tr(S)/n, "
                 + ("QR on" if qr_orthonormalize else "QR off") + ")")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    # ---- Footer summary under the x-axis (no plot compression) ----
    spike_text = ", ".join(
        [rf"$\alpha_{i+1}$={a:.2f}, $\lambda_{i+1}=p^{{\alpha_i}}$={lam[i]:.2f}"
         for i, a in enumerate(alphas)]
    )
    footer = (
        rf"{len(alphas)} spike(s) | bulk mean $\mu \approx {bulk_mean:.2f}$ | "
        + spike_text
    )
    fig.text(0.5, -0.06, footer, ha='center', va='top', fontsize=9)

    fig.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run()
