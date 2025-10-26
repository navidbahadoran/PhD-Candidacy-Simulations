import numpy as np, matplotlib.pyplot as plt, os

def dual_cov(X): n=X.shape[1]; return (X.T @ X)/n

def nr_lambda_vec(hat_lams):
    n = len(hat_lams); out = np.zeros_like(hat_lams)
    for j in range(n-1): out[j] = hat_lams[j] - hat_lams[j+1:].sum()/(n-j)
    return out

def run(p=3000, alpha=0.7, n_vals=(20,40,80,120,200), reps=30, seed=1, fname="figures/sim4_pc_scores.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p); lam[0]=p**alpha
    nmse_naive=[]; nmse_nr=[]
    for n in n_vals:
        errs_naive=[]; errs_nr=[]
        for _ in range(reps):
            Z = rng.standard_normal((p, n)); X = (np.sqrt(lam)[:,None]*Z)
            SD = dual_cov(X)
            evals, U = np.linalg.eigh(SD)
            idx = np.argsort(evals)[::-1]; evals = evals[idx]; U = U[:,idx]
            s = np.sqrt(lam[0]) * Z[0,:]
            s_hat = np.sqrt(n*evals[0]) * U[:,0]
            til = nr_lambda_vec(evals)
            s_til = np.sqrt(n*max(til[0],1e-12)) * U[:,0]
            errs_naive.append(np.mean((s_hat - s)**2)/lam[0])
            errs_nr.append(np.mean((s_til - s)**2)/lam[0])
        nmse_naive.append(np.mean(errs_naive)); nmse_nr.append(np.mean(errs_nr))
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.0,4.0))
    plt.plot(n_vals, nmse_naive, "o-", label="naive")
    plt.plot(n_vals, nmse_nr, "x-", label="NR")
    plt.xlabel("sample size n"); plt.ylabel("NMSE (component 1)")
    plt.title(f"PC score NMSE vs n (p={p}, α={alpha})")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(fname, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
