import numpy as np, matplotlib.pyplot as plt, os

def angle_between(u, v):
    c = abs(float(u.T @ v))
    c = max(min(c, 1.0), -1.0)  # numeric guard
    return np.arccos(c)

def bbp_alignment_sq(lam, gamma):
    # Theoretical squared alignment (Paul 2007 / spiked covariance)
    thr = 1.0 + np.sqrt(gamma)
    if lam <= thr:
        return 0.0
    return (1.0 - gamma / (lam - 1.0)**2) / (1.0 + gamma / (lam - 1.0))

def run(p=3000, alpha=0.7, n_vals=(10,20,40,80,120,200), reps=30, seed=0,
        angle_fname="figures/sim3_pc_angles.pdf",
        align_fname="figures/sim3_pc_alignment.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p); lam[0] = p**alpha        # single spike along e1
    v_true = np.zeros(p); v_true[0] = 1.0

    mean_angles = []
    mean_aligns = []
    theo_aligns = []
    errs=[]

    for n in n_vals:
        angs = []
        aligns = []

        for _ in range(reps):
            Z = rng.standard_normal((p, n))
            X = (np.sqrt(lam)[:, None] * Z)
            S = (X @ X.T) / n
            evals, vecs = np.linalg.eigh(S)
            vhat = vecs[:, -1]                 # top eigenvector

            # angle and alignment
            theta = angle_between(vhat, v_true)
            angs.append(theta)
            aligns.append(np.cos(theta)**2)    # alignment = cos^2(angle)

        angs = np.array(angs)
        mean_angles.append(np.degrees(angs).mean())
        mean_aligns.append(float(np.mean(aligns)))
        errs.append(np.degrees(angs).std())

        # theoretical alignment for this (p,n)
        gamma = p / n
        theo_aligns.append(bbp_alignment_sq(lam[0], gamma))

    os.makedirs("figures", exist_ok=True)

    # --- Plot 1: mean angle vs n ---
    plt.figure(figsize=(6.0, 4.0))
    plt.errorbar(n_vals, mean_angles, yerr=errs, fmt="o-")
    # plt.plot(n_vals, mean_angles, "o-")
    plt.xlabel("sample size n")
    plt.ylabel("mean angle (degrees)")
    plt.title(f"PC1 angle vs n (p={p}, α={alpha:.2f})")
    plt.tight_layout(); plt.savefig(angle_fname, bbox_inches="tight"); plt.close()

    # --- Plot 2: alignment vs n (empirical + theory) ---
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(n_vals, mean_aligns, "o-", label="empirical alignment")
    plt.plot(n_vals, theo_aligns, "--", label="theory (BBP/Paul)")
    plt.ylim(0.0, 1.02)
    plt.xlabel("sample size n")
    plt.ylabel(r"alignment $|\langle \hat v, v\rangle|^2 = \cos^2(\theta)$")
    plt.title(f"PC1 alignment vs n (p={p}, α={alpha:.2f})")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(align_fname, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
