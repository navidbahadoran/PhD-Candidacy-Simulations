import numpy as np, matplotlib.pyplot as plt, os

def run(p=2000, n1=20, n2=20, reps=50, alpha=0.7, seed=42, fname="figures/sim6_discriminant.pdf"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p); lam[0]=p**alpha
    h1 = np.zeros(p); h1[0]=1.0
    d = 1.0
    mu1 = np.zeros(p); mu2 = d*h1
    errs_mp=[]; errs_ridge=[]; errs_nr=[]
    for _ in range(reps):
        Z1 = rng.standard_normal((p,n1)); X1 = (np.sqrt(lam)[:,None]*Z1) + mu1[:,None]
        Z2 = rng.standard_normal((p,n2)); X2 = (np.sqrt(lam)[:,None]*Z2) + mu2[:,None]
        X = np.concatenate([X1, X2], axis=1); y = np.array([0]*n1 + [1]*n2)
        m1 = X1.mean(axis=1); m2 = X2.mean(axis=1)
        S = (X - X.mean(axis=1, keepdims=True)) @ (X - X.mean(axis=1, keepdims=True)).T / (n1+n2-1)
        # MP
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        Sinv_mp = (U * (1.0/np.where(s>1e-8, s, np.inf))) @ U.T
        # ridge
        delta = np.mean(lam)/(n1+n2)
        Sinv_r = np.linalg.inv(S + delta*np.eye(p))
        # NR (approx via dual)
        SD = (X.T @ X)/(n1+n2)
        evals, Udual = np.linalg.eigh(SD); idx=np.argsort(evals)[::-1]; evals=evals[idx]; Udual=Udual[:,idx]
        tilde = np.zeros_like(evals); n = n1+n2
        for j in range(n-1): tilde[j] = evals[j] - evals[j+1:].sum()/(n-j)
        omega = min(np.mean(lam)/np.sqrt(n), np.mean(lam)/n)
        lam_bar = np.maximum(tilde, omega)
        Hhat = (X @ Udual) / np.sqrt(n*evals + 1e-12)
        P = Hhat @ Hhat.T
        Sinv_nr = Hhat @ np.diag(1.0/lam_bar) @ Hhat.T + (np.eye(p)-P) * (1.0/omega)
        # LDA-like scores
        def err_rate(Sinv):
            w = Sinv @ (m2-m1); b = -0.5*(m1+m2).T@w
            scores = w.T @ X + b
            yhat = (scores>0).astype(int)
            return np.mean(yhat!=y)
        errs_mp.append(err_rate(Sinv_mp))
        errs_ridge.append(err_rate(Sinv_r))
        errs_nr.append(err_rate(Sinv_nr))
    means = [np.mean(errs_mp), np.mean(errs_ridge), np.mean(errs_nr)]
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5.6,3.8))
    plt.bar(["MP","Ridge","NR"], means)
    plt.ylabel("misclassification rate")
    plt.title(f"HDLSS LDA (p={p}, n1=n2={n1}, α={alpha})")
    plt.tight_layout(); plt.savefig(fname, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
