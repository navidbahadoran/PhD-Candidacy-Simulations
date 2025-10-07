import numpy as np, matplotlib.pyplot as plt, os

def bbp_outlier(lambda_pop, gamma):
    return lambda_pop * (1.0 + gamma / (lambda_pop - 1.0))

def bbp_alignment_sq(lambda_pop, gamma):
    return (1.0 - gamma / (lambda_pop - 1.0)**2) / (1.0 + gamma / (lambda_pop - 1.0))

def run(p=1000, n=500, reps=30, lambdas=(1.2, 1.6, 2.0, 2.6, 3.5, 5.0), seed=777, prefix="figures/sim0_bbp_transition"):
    rng = np.random.default_rng(seed)
    gamma = p / n
    b = (1 + np.sqrt(gamma))**2
    means_eval = []; means_align = []; theo_eval = []; theo_align = []
    for lam in lambdas:
        lhat = []; align = []
        for _ in range(reps):
            z = rng.standard_normal((p, n))
            z[0, :] = np.sqrt(lam) * z[0, :]
            S = (z @ z.T) / n
            evals, vecs = np.linalg.eigh(S)
            idx = np.argmax(evals)
            lhat.append(evals[idx])
            vhat = vecs[:, idx]
            align.append((vhat[0]**2))
        mh = float(np.mean(lhat)); ma = float(np.mean(align))
        means_eval.append(mh); means_align.append(ma)
        if lam <= b:
            theo_eval.append(b); theo_align.append(0.0)
        else:
            theo_eval.append(bbp_outlier(lam, gamma))
            theo_align.append(bbp_alignment_sq(lam, gamma))
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.0,4.0))
    plt.plot(lambdas, theo_eval, lw=2, label="theory")
    plt.plot(lambdas, means_eval, "o", label="empirical")
    plt.axvline(b, ls="--", lw=1, label="threshold")
    plt.xlabel("population spike λ"); plt.ylabel("top sample eigenvalue")
    plt.title(f"BBP eigenvalue (γ={gamma:.2f})"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(prefix+"_eigs.pdf", bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6.0,4.0))
    plt.plot(lambdas, theo_align, lw=2, label="theory")
    plt.plot(lambdas, means_align, "o", label="empirical")
    plt.axvline(b, ls="--", lw=1, label="threshold")
    plt.xlabel("population spike λ"); plt.ylabel(r"alignment $|\langle \hat v, v\rangle|^2$")
    plt.title(f"BBP alignment (γ={gamma:.2f})"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(prefix+"_align.pdf", bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    run()
