import numpy as np, matplotlib.pyplot as plt, os

def run(p=2000, n_vals=(20,40,80,120), alphas=(0.8,0.6), reps=10, seed=0,
        fname="figures/sim5_inverse.pdf"):
    rng = np.random.default_rng(seed)
    # population spectrum (spikes + 1's)
    lam = np.ones(p)
    for i, a in enumerate(alphas):
        lam[i] = p**a

    errs_ridge_primal, errs_nr_primal = [], []
    errs_ridge_span,  errs_nr_span  = [], []

    for n in n_vals:
        er_r_p = []; er_nr_p = []
        er_r_s = []; er_nr_s = []

        # # ridge level for this n
        # delta = delta_global if hold_delta else lam_mean / n

        for _ in range(reps):
            Z = rng.standard_normal((p, n))
            X = (np.sqrt(lam)[:, None] * Z)

            # primal & dual sample covariances
            S  = (X @ X.T)/n
            SD = (X.T @ X)/n

            trS = float(np.trace(S))
            delta = trS / n

            # dual eigens, sorted ↓
            evals, U = np.linalg.eigh(SD)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]; U = U[:, idx]

            # left singular directions H ≈ orthonormal columns
            H = (X @ U) / np.sqrt(n*evals + 1e-12)  # p×n
            # Optional: improve orthonormality numerically
            # H, _ = np.linalg.qr(H)  # uncomment if needed

            # ---------- Ridge inverse ----------
            # (S + delta I)^{-1}
            Si = np.linalg.inv(S + delta*np.eye(p))


            # SPAN whitening error: || H^T Λ^{1/2} W Λ^{1/2} H - I ||_F / √n
            LhalfH = np.sqrt(lam)[:, None] * H         # p×n
            Asp_r  = LhalfH.T @ Si @ LhalfH            # n×n
            er_r_s.append(np.linalg.norm(Asp_r - np.eye(n), 'fro') / np.sqrt(n))

            # ---------- NR inverse (span + complement) ----------
            # NR dual eigenvalues: subtract tail mean (noise reduction)
            tilde = np.zeros_like(evals)
            for j in range(n-1):
                tilde[j] = evals[j] - evals[j+1:].sum()/(n-j)

            # floor to stabilize complement
            omega   = min(trS/(np.sqrt(p)*(n**0.25)), delta)
            lam_bar = np.maximum(tilde, omega)

            # projector onto span(X)
            P = H @ H.T

            # NR inverse W_nr = H diag(1/lam_bar) H^T + (I-P)*(1/omega)
            W_nr  = H @ np.diag(1.0/lam_bar) @ H.T + (np.eye(p) - P) * (1.0/omega)


            # SPAN error (method-specific operator on span)
            Asp_nr = LhalfH.T @ W_nr @ LhalfH
            er_nr_s.append(np.linalg.norm(Asp_nr - np.eye(n), 'fro') / np.sqrt(n))

        errs_ridge_span.append(np.mean(er_r_s))
        errs_nr_span.append(np.mean(er_nr_s))

    # --------- Plot ----------
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.4,4.2))
    # plt.plot(n_vals, errs_ridge_primal, "o-", label="ridge (primal)")
    # plt.plot(n_vals, errs_nr_primal,  "o-", label="NR (primal)")
    plt.plot(n_vals, errs_ridge_span, "x--", label="ridge")
    plt.plot(n_vals, errs_nr_span,    "o--", label="NR")
    plt.xlabel("sample size n")
    plt.ylabel("error")
    ttl_extra = ", δ = tr(S)/n"
    plt.title(f"Inverse estimator error vs n (p={p}{ttl_extra})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # try hold_delta=True to remove the artificial rise with n
    run()
