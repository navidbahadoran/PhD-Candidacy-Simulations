"""
sim4_pc_scores.py — PC scores with/without noise reduction (Section 3.7).

Computes in-sample MSE(ŝ_j) and MSE(ṡ_j) relative to s_j = sqrt(λ_j) z_j.
"""
import numpy as np
from common import SpikedModel, generate_data, dual_cov, set_seed, mse

def one_run(p=6000, n=60, alpha=0.7, seed=30):
    set_seed(seed)
    model = SpikedModel(p=p, n=n, m=1, alphas=np.array([alpha]), a=np.array([1.0]), c_bulk=1.0)
    X, lam, H = generate_data(model, kind="gaussian")
    SD = dual_cov(X)
    evals, U = np.linalg.eigh(SD)
    evals = evals[::-1]; U = U[:, ::-1]
    lam1 = lam[0]

    # population scores for j=1
    z1 = (H.T @ X / np.sqrt(lam[:, None]))[0, :]  # equals first row of Z
    s = np.sqrt(lam1) * z1

    # naïve score: sqrt(n \hat λ_1) * \hat u_1
    s_hat = np.sqrt(n * evals[0]) * U[:, 0]

    # NR score: sqrt(n \tilde λ_1) * \hat u_1
    mu_hat = (evals[1:].sum()) / (n - 1)
    lam_tilde = max(evals[0] - mu_hat, 0.0)
    s_tilde = np.sqrt(n * lam_tilde) * U[:, 0]

    print(f"[sim4] p={p} n={n} alpha={alpha} λ1={lam1:.2e}")
    print(f"  MSE naive / λ1 = {mse(s_hat, s)/lam1:.3f}")
    print(f"  MSE  NR   / λ1 = {mse(s_tilde, s)/lam1:.3f}")

if __name__ == "__main__":
    one_run(p=6000, n=60, alpha=0.7, seed=31)
    one_run(p=6000, n=40, alpha=0.55, seed=32)
