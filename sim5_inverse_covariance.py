import numpy as np, matplotlib.pyplot as plt, os

def random_orthonormal(p, rng=None):
    """Return a random orthonormal matrix H ∈ R^{p×p} via QR with sign-fix."""
    rng = np.random.default_rng(rng)
    A = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(A)
    # fix sign ambiguity so columns are uniquely oriented
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q

def make_lambda(p, spikes=None, bulk=1.0, rng=None):
    """
    Build a diagonal Λ with 'spikes' (list/array) and bulk variance.
    Example: spikes=[10, 5] puts two big eigenvalues, rest = bulk.
    """
    lam = np.full(p, float(bulk))
    if spikes:
        # m = len(spikes)
        for i, a in enumerate(spikes):
            lam[i] = p**a
        # lam[:m] = np.array(spikes, dtype=float)
    #     if permute:
    #         rng = np.random.default_rng(rng)
    #         rng.shuffle(lam)  # optional: randomize spike positions
    return lam

def make_covariance(p, spikes=None, bulk=1.0, H=None, seed=0):
    """
    Construct Σ = H Λ H^T and return (Σ, H, Λ).
    - p: dimension
    - spikes: list of spike eigenvalues (> bulk), or None
    - bulk: bulk eigenvalue (e.g., 1.0)
    - H: optional orthonormal basis; if None, generate random
    """
    rng = np.random.default_rng(seed)
    if H is None:
        H = random_orthonormal(p, rng)
    lam = make_lambda(p, spikes=spikes, bulk=bulk, rng=rng)
    # Σ = H diag(λ) H^T
    Sigma = (H * lam) @ H.T  # broadcasting columns of H by λ then H^T
    # make perfectly symmetric numerically
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma, H, lam

def nr_lambda_vec(hat_lams):
    n = len(hat_lams)
    out = np.zeros_like(hat_lams)
    for j in range(n-1):
        out[j] = hat_lams[j] - hat_lams[j+1:].sum()/(n-j)
    out[n-1] = hat_lams[n-1]
    return out

def run(p_vals=(1600,2000,2500,3000,3500,10000), n=60, alphas=(0.8,0.6), reps=10, seed=0,
        fname="figures/sim5_inverse.pdf", qr_orthonormalize=False):
    rng = np.random.default_rng(seed)

    # population spectrum (spikes + 1's)

    errs_ridge_span, errs_nr_span = [], []
    eps = 1e-12

    for p in p_vals:
        Sigma, H, lam = make_covariance(p, spikes=alphas, bulk=1.0, seed=42)
        bulk_mean = lam[len(alphas):].mean() if len(alphas) < p else 0.0
        er_r_s, er_nr_s = [], []

        for _ in range(reps):
            # --- simulate data ---
            Z = rng.standard_normal((p, n))
            X = H @ (np.sqrt(lam)[:, None] * Z)
            # center columns (theory uses centered data)
            X = X - X.mean(axis=1, keepdims=True)

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
            U = U[:, idx]

            # keep r = n-1 components
            r = n - 1

            # left singular directions H ≈ orthonormal columns
            H_hat = (X @ U) / np.sqrt(n*evals + eps)   # p×n
            P_hat = H_hat @ H_hat.T
            # if qr_orthonormalize:
            #     H, _ = np.linalg.qr(H)             # enforce H^T H ≈ I_n numerically
            
            tilde = nr_lambda_vec(evals)
            tilde = np.maximum(tilde[:r], eps)
            H_tilde = (X @ U[:,:r]) / np.sqrt(n*tilde)
            P_tilde = H_tilde @ H_tilde.T
            
            lam_hat = evals + delta
            
            Si = H_hat @ np.diag(1.0/lam_hat) @ H_hat.T + (np.eye(p) - P_hat) * (1.0/delta) #  pxp 

            # SPAN whitening error: || H^T Λ^{1/2} W Λ^{1/2} H - I ||_F / √n
            LhalfH = np.sqrt(lam)[:, None] * H.T         # p×p
            Asp_r  = LhalfH @ Si @ LhalfH.T            # pxp
            er_r_s.append(np.linalg.norm(Asp_r - np.eye(p), 'fro') / np.sqrt(p))


            # floor ω = min(tr(S)/(√p n^{1/4}), δ)
            omega   = 1 # min(trS / (np.sqrt(p) * (n**0.25)), delta)
            lam_bar = np.maximum(tilde, omega)


            W_nr  = H_tilde @ np.diag(1.0/lam_bar) @ H_tilde.T + (np.eye(p) - P_tilde) * (1.0/omega)


            # m = len(alphas)                         # number of population spikes you planted
            # H_m = H[:, :m]                          # the true spike directions (you know H, Λ)
            # Λm_half = np.sqrt(lam[:m])[:, None]     # (m×1)
            
            # # Ridge / NR inverse already built: Si, W_nr   (p×p)
            
            # V_r  = Λm_half.T @ (H_m.T @ Si   @ H_m) @ Λm_half   # m×m
            # V_nr = Λm_half.T @ (H_m.T @ W_nr @ H_m) @ Λm_half   # m×m
            
            # er_r_s.append(np.linalg.norm(V_r  - np.eye(m), 'fro') / np.sqrt(m))
            # er_nr_s.append(np.linalg.norm(V_nr - np.eye(m), 'fro') / np.sqrt(m))


            Asp_nr = LhalfH @ W_nr @ LhalfH.T
            er_nr_s.append(np.linalg.norm(Asp_nr - np.eye(p), 'fro') / np.sqrt(p))

        errs_ridge_span.append(float(np.mean(er_r_s)))
        errs_nr_span.append(float(np.mean(er_nr_s)))

    # --------- Plot ----------
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(p_vals, errs_ridge_span, "x--", label="Ridge (span whitening error)")
    ax.plot(p_vals, errs_nr_span,    "o--", label="NR (span whitening error)")
    ax.set_xlabel("Dimension $p$")
    ax.set_ylabel(r"Population Error (relative to $\Sigma_p^{-1}$)")
    ax.set_title(f"Inverse estimator (span) error vs $p$ (n={n}, δ=tr(S)/n, "+ ")")
    ax.legend(frameon=False, loc="center")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    # ---- Footer summary under the x-axis (no plot compression) ----
    spike_text = ", ".join(
        [rf"$\alpha_{i+1}$={a:.2f}, $\lambda_{i+1}=p^{{\alpha_i}}$"
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
