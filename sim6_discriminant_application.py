"""
sim6_discriminant_application.py — Discriminant analysis demo (Section 8-style application).

Two Gaussian classes in HDLSS. Compare classification with:
- S^{-1}_ω (NR-based inverse estimator)
- Moore–Penrose inverse of S (pseudo-inverse)
- Ridge inverse (S + δI)^{-1} with δ = tr(S)/n

Reports test error rates.
"""
import numpy as np
from numpy.linalg import pinv
from common import SpikedModel, generate_data, dual_cov, inverse_nr_estimator, set_seed

def sample_two_class(model_pos, model_neg, n_test=200, seed=50):
    set_seed(seed)
    Xp, lam_p, H = generate_data(model_pos, kind="gaussian")
    Xn, lam_n, H = generate_data(model_neg, kind="gaussian")
    # class means shifted along first PC of pos class
    mu = np.zeros(model_pos.p)
    mu[0] = 2.0  # signal in class separation
    Xp = Xp + mu[:, None]
    Xn = Xn - mu[:, None]
    # Train pooled covariance on combined training data (centered by class means)
    X_train = np.hstack([Xp, Xn])
    y_train = np.array([1]*model_pos.n + [0]*model_neg.n)
    mu_p = Xp.mean(axis=1, keepdims=True)
    mu_n = Xn.mean(axis=1, keepdims=True)
    Xc = np.hstack([Xp - mu_p, Xn - mu_n])
    S = (Xc @ Xc.T) / Xc.shape[1]

    # Estimators
    # 1) NR-based inverse
    Soinv, aux = inverse_nr_estimator(Xc)
    # 2) Moore–Penrose
    S_pinv = pinv(S)
    # 3) Ridge
    delta = np.trace(S) / Xc.shape[1]
    S_ridge_inv = np.linalg.inv(S + delta*np.eye(model_pos.p))

    # LDA-like rule with given inverse: w = Sinv (mu_p - mu_n)
    w_nr = Soinv @ ((mu_p - mu_n).ravel())
    w_pinv = S_pinv @ ((mu_p - mu_n).ravel())
    w_ridge = S_ridge_inv @ ((mu_p - mu_n).ravel())

    # Test data
    def draw(model, mshift, n_draw):
        X, _, _ = generate_data(model, kind="gaussian")
        return (X + mshift[:, None])[:, :n_draw]

    Xt_pos = draw(model_pos, mu.ravel(), n_test//2)
    Xt_neg = draw(model_neg, -mu.ravel(), n_test//2)
    X_test = np.hstack([Xt_pos, Xt_neg])
    y_test = np.array([1]*(n_test//2) + [0]*(n_test//2))

    def err(w):
        scores = w @ (X_test - 0.5*(mu_p + mu_n))
        yhat = (scores > 0).astype(int)
        return (yhat != y_test).mean()

    e_nr = err(w_nr)
    e_pinv = err(w_pinv)
    e_ridge = err(w_ridge)

    print(f"[sim6] Test error rates (smaller is better):")
    print(f"  NR inverse     : {e_nr:.3f}")
    print(f"  Moore–Penrose  : {e_pinv:.3f}")
    print(f"  Ridge inverse  : {e_ridge:.3f}  (δ = tr(S)/n)")

if __name__ == "__main__":
    p, n = 4000, 40
    model_pos = SpikedModel(p=p, n=n, m=2, alphas=np.array([0.8, 0.6]), a=np.array([1.0, 1.0]), c_bulk=1.0)
    model_neg = SpikedModel(p=p, n=n, m=2, alphas=np.array([0.8, 0.6]), a=np.array([1.0, 1.0]), c_bulk=1.0)
    sample_two_class(model_pos, model_neg, n_test=200, seed=55)
