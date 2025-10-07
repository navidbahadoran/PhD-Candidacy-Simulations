import numpy as np, matplotlib.pyplot as plt, os

def angle_between(u, v):
    c = abs(float(u.T @ v)); c = max(min(c,1.0), -1.0); return np.arccos(c)

def run(p=3000, alpha=0.7, n_vals=(10,20,40,80,120,200), reps=30, seed=0, fname="figures/sim3_pc_angles.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p); lam[0] = p**alpha
    means = []
    for n in n_vals:
        angs = []
        for _ in range(reps):
            Z = rng.standard_normal((p, n)); X = (np.sqrt(lam)[:,None] * Z)
            S = (X @ X.T)/n
            evals, vecs = np.linalg.eigh(S)
            vhat = vecs[:, -1]; h1 = np.zeros(p); h1[0]=1.0
            angs.append(angle_between(vhat, h1))
        means.append(np.mean(angs))
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.0,4.0))
    plt.plot(n_vals, means, "o-")
    plt.xlabel("sample size n"); plt.ylabel("mean angle (radians)")
    plt.title(f"PC1 angle vs n (p={p}, α={alpha})")
    plt.tight_layout(); plt.savefig(fname, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
