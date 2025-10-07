"""
sim3_pc_directions.py — PC direction consistency (Section 3.6).

Measures angle between sample PC direction \hat h_j and population h_j.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from common import SpikedModel, generate_data, dual_cov, set_seed, angle

def angle_experiment(p=4000, n_list=(20, 40, 80), alpha=0.8, j=1, reps=10, seed=10):
    angles = []
    for n in n_list:
        vals = []
        for r in range(reps):
            set_seed(seed + 100*r + n)
            model = SpikedModel(p=p, n=n, m=1, alphas=np.array([alpha]), a=np.array([1.0]), c_bulk=1.0)
            X, lam, H = generate_data(model, kind="gaussian")
            SD = dual_cov(X)
            evals, U = np.linalg.eigh(SD)
            U = U[:, ::-1]
            uh = U[:, j-1]
            # primal direction
            h = np.zeros(p); h[0] = 1.0  # since H=I
            # map dual eigenvector to primal PC: \hat h_j = X \hat u_j / sqrt(n \hat λ_j)
            lam_hat = np.sort(evals)[::-1][j-1]
            hhat = (X @ uh) / np.sqrt(n * lam_hat + 1e-12)
            vals.append(angle(hhat, h))
        angles.append((n, np.mean(vals), np.std(vals)))
    return angles

if __name__ == "__main__":
    res = angle_experiment(p=4000, n_list=(20, 40, 80, 120), alpha=0.8, j=1, reps=8, seed=11)
    print("[sim3] Angle(ĥ, h): mean±std (radians)")
    for n, mu, sd in res:
        print(f"  n={n:3d} -> {mu:.3f} ± {sd:.3f}")
