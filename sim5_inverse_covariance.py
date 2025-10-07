"""
sim5_inverse_covariance.py — Inverse covariance estimator S^{-1}_ω (Section 3.8).

Evaluates V_ω = Λ^{1/2} H^T S^{-1}_ω H Λ^{1/2} and its deviation from I_p on spike/bulk blocks.
"""
import numpy as np
from numpy.linalg import norm
from common import SpikedModel, generate_data, inverse_nr_estimator, set_seed

def run(p=5000, n=80, m=3, alphas=(0.8, 0.7, 0.6), a=(1.0, 1.0, 1.0), bulk=1.0, seed=40):
    set_seed(seed)
    model = SpikedModel(p=p, n=n, m=m, alphas=np.array(alphas), a=np.array(a), c_bulk=bulk)
    X, lam, H = generate_data(model, kind="gaussian")
    Soinv, aux = inverse_nr_estimator(X, omega=None)
    # V_ω
    V = np.diag(np.sqrt(lam)) @ Soinv @ np.diag(np.sqrt(lam))
    # summarize diagonal entries for spikes vs bulk
    diagV = np.diag(V)
    spikes = diagV[:m]
    bulk_diag = diagV[m:]
    print(f"[sim5] p={p} n={n} m={m} alphas={alphas}")
    print(f"  spikes diag(V_ω): mean={spikes.mean():.3f}, std={spikes.std():.3f}")
    print(f"  bulk   diag(V_ω): mean={bulk_diag.mean():.3f}, std={bulk_diag.std():.3f}")
    print(f"  ||V_ω - I||_F / sqrt(p) = {norm(V - np.eye(p), 'fro')/np.sqrt(p):.3f}")
    print(f"  omega used = {aux['omega']:.3f}")

if __name__ == "__main__":
    run(p=5000, n=80, m=3, alphas=(0.8, 0.7, 0.6), seed=41)
