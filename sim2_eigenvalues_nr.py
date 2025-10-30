import numpy as np
import matplotlib.pyplot as plt
import os

def dual_cov(X):
    n = X.shape[1]
    return (X.T @ X) / n

def nr_lambda(hat_lams, j):
    n = len(hat_lams)
    bulk_mean = (hat_lams[j+1:].sum())/(n-j) if j < n-1 else 0.0
    return hat_lams[j] - bulk_mean

def run(p=3000, n=60, alphas=(0.8,0.6), kplot=10, seed=0, prefix="figures/sim2_eigs"):
    rng = np.random.default_rng(seed)
    lam = np.ones(p)
    for idx, a in enumerate(alphas):
        lam[idx] = p**a

    # population spikes and bulk mean
    spikes = lam[:len(alphas)]
    bulk_mean = lam[len(alphas):].mean() if len(alphas) < p else 0.0

    Z = rng.standard_normal((p, n))
    X = (np.sqrt(lam)[:, None] * Z)
    SD = dual_cov(X)
    hat_lams = np.linalg.eigvalsh(SD)[::-1]

    # Noise-reduced eigenvalues
    tilde = np.array([nr_lambda(hat_lams, j) for j in range(n-1)] + [0.0])

    # True population eigenvalues (spikes + 1’s)
    true_lams = np.concatenate([lam[:n], np.ones(max(0, n - len(lam)))])

    os.makedirs("figures", exist_ok=True)
    jidx = np.arange(1, kplot+1)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))

    # Plot sample, NR, and true eigenvalues
    ax1.plot(jidx, hat_lams[:kplot], "o-", label=r"Sample $\hat\lambda_j$ (naive)")
    ax1.plot(jidx, tilde[:kplot], "x-", label=r"Noise-reduced $\tilde\lambda_j$")
    ax1.plot(jidx, true_lams[:kplot], "s--", color="tab:green", label=r"True $\lambda_j$")

    # Axis labels and legend
    ax1.set_xlabel("Eigenvalue index $j$")
    ax1.set_ylabel("Eigenvalue magnitude")
    ax1.set_title(f"Noise-reduction vs naive eigenvalues (p={p}, n={n})")
    ax1.legend(frameon=False, loc="upper right")

    # ---- Footer summary under the x-axis ----
    spike_text = ", ".join(
        [rf"$\alpha_{i+1}$={a:.2f}, $\lambda_{i+1}=p^{{\alpha_i}}$={spikes[i]:.1f}"
         for i, a in enumerate(alphas)]
    )
    footer_text = (
        rf"{len(alphas)} spikes | bulk mean $\mu\approx{bulk_mean:.2f}$ | " + spike_text
    )

    # Place footer centered below plot
    fig.text(0.5, -0.05, footer_text, ha='center', va='top', fontsize=9)

    fig.tight_layout()
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run()
