import numpy as np, matplotlib.pyplot as plt, os

def run(p=2000, n_vals=(20,40,80,120), alphas=(0.8,0.6), reps=10, seed=0, fname="figures/sim5_inverse.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p)
    for i,a in enumerate(alphas): lam[i] = p**a
    errs_ridge=[]; errs_nr=[]
    for n in n_vals:
        er_r=[]; er_nr=[]
        delta = lam.mean()/n
        for _ in range(reps):
            Z = rng.standard_normal((p, n)); X = (np.sqrt(lam)[:,None]*Z)
            S = (X @ X.T)/n
            SD = (X.T @ X)/n
            evals, U = np.linalg.eigh(SD); idx = np.argsort(evals)[::-1]; evals=evals[idx]; U=U[:,idx]
            # Ridge inverse
            Si = np.linalg.inv(S + delta*np.eye(p))
            A = (np.sqrt(lam)[:,None]*Si*np.sqrt(lam)[None,:])
            er_r.append(np.linalg.norm(A - np.eye(p), ord='fro')/np.sqrt(p))
            # NR inverse (approx via sample span)
            omega = min(lam.mean()/np.sqrt(n), delta)
            tilde = np.zeros_like(evals)
            for j in range(n-1):
                tilde[j] = evals[j] - evals[j+1:].sum()/(n-j)
            lam_bar = np.maximum(tilde, omega)
            Hhat = (X @ U) / np.sqrt(n*evals + 1e-12)
            P = Hhat @ Hhat.T
            W = Hhat @ np.diag(1.0/lam_bar) @ Hhat.T + (np.eye(p)-P)*(1.0/omega)
            A = (np.sqrt(lam)[:,None]*W*np.sqrt(lam)[None,:])
            er_nr.append(np.linalg.norm(A - np.eye(p), ord='fro')/np.sqrt(p))
        errs_ridge.append(np.mean(er_r)); errs_nr.append(np.mean(er_nr))
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.0,4.0))
    plt.plot(n_vals, errs_ridge, "o-", label="ridge inverse")
    plt.plot(n_vals, errs_nr, "x-", label="NR inverse")
    plt.xlabel("sample size n"); plt.ylabel(r"$\| \Lambda^{1/2}S^{-1}\Lambda^{1/2}-I \|_F/\sqrt{p}$")
    plt.title(f"Inverse estimator error vs n (p={p})")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(fname, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
