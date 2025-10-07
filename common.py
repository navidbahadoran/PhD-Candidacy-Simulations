"""
Common utilities for HDLSS simulation scripts (Chapter 4).

Implements:
- Spiked covariance generator: Sigma = H diag([a_i p^{alpha_i} (i<=m), c_j]) H^T with H=I by default.
- Sphered data Z ~ i.i.d. with optional rho-mixing along rows (AR(1) on squared entries proxy).
- Dual covariance S_D = (1/n) X^T X and primal S = (1/n) X X^T.
- Noise-reduced eigenvalues tilde_lambda_j per Yata–Aoshima NR methodology.
- Angle between vectors.
- Inverse estimator S_omega^{-1} built from NR eigenpairs (Section 3.8).

All scripts use a fixed RNG seed for reproducibility unless overridden.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# --------------------- RNG helpers ---------------------
def set_seed(seed: int = 2025) -> None:
    np.random.seed(seed)

# --------------------- Model spec ----------------------
@dataclass
class SpikedModel:
    p: int
    n: int
    m: int
    alphas: np.ndarray  # shape (m,), exponents alpha_i > 0
    a: np.ndarray       # shape (m,), positive scales
    c_bulk: float = 1.0 # bulk level (can be 1.0 or other constant)
    H: Optional[np.ndarray] = None  # optional orthogonal rotation (p x p)

    def lambda_vec(self) -> np.ndarray:
        lam = np.full(self.p, self.c_bulk, dtype=float)
        for i in range(self.m):
            lam[i] = self.a[i] * (self.p ** self.alphas[i])
        return lam

    def eigsys(self) -> Tuple[np.ndarray, np.ndarray]:
        lam = self.lambda_vec()
        if self.H is None:
            H = np.eye(self.p)
        else:
            H = self.H
        return lam, H

# --------------------- Data generation -----------------

def sample_Z(p: int, n: int, kind: str = "gaussian", df: int = 5) -> np.ndarray:
    """
    Draw Z in R^{p x n} with identity covariance.
    kind: 'gaussian' (standard normal) or 't' (standardized t with df).
    """
    if kind == "gaussian":
        Z = np.random.randn(p, n)
    elif kind == "t":
        # Student-t(df) then standardize to unit variance (df>2)
        T = np.random.standard_t(df, size=(p, n))
        Z = T / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError("Unknown kind")
    return Z

def apply_rho_mixing_proxy(Z: np.ndarray, rho: float = 0.4) -> np.ndarray:
    """
    Introduce weak dependence across rows j for each column k by AR(1)-filtering
    the *squared* entries and rescaling back to zero-mean/unit-variance proxy.
    This mimics nontrivial Cov(z_{jk}^2, z_{j'k}^2). Not exact ρ-mixing, but a proxy.
    """
    p, n = Z.shape
    Z2 = Z**2
    Z2_dep = np.empty_like(Z2)
    for k in range(n):
        x = Z2[:, k] - 1.0  # center
        y = np.empty_like(x)
        y[0] = x[0]
        for j in range(1, p):
            y[j] = rho * y[j-1] + np.sqrt(1 - rho**2) * x[j]
        # re-center and rescale y to have variance ~ Var(x)=2 (for Gaussian)
        y = y - y.mean()
        if y.std() > 0:
            y = y / y.std() * x.std()
        Z2_dep[:, k] = 1.0 + y
    # Combine magnitudes with original signs to keep mean zero and unit variance approximately
    Z_dep = np.sign(Z) * np.sqrt(np.maximum(Z2_dep, 1e-12))
    # small recentering/revariance
    Z_dep -= Z_dep.mean(axis=1, keepdims=True)
    Z_dep /= Z_dep.std(axis=1, keepdims=True) + 1e-12
    return Z_dep

def generate_data(model: SpikedModel, kind: str = "gaussian", df: int = 5, rho_mix: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X, Lambda, H. Constructs X = H diag(sqrt(lambda)) Z with chosen Z.
    If rho_mix is not None, applies row-wise dependence proxy to Z-squareds.
    """
    lam, H = model.eigsys()
    Z = sample_Z(model.p, model.n, kind=kind, df=df)
    if rho_mix is not None:
        Z = apply_rho_mixing_proxy(Z, rho=rho_mix)
    X = (H @ (np.sqrt(lam)[:, None] * Z))
    return X, lam, H

# --------------------- Covariances ----------------------
def dual_cov(X: np.ndarray) -> np.ndarray:
    return (X.T @ X) / X.shape[1]

def primal_cov(X: np.ndarray) -> np.ndarray:
    return (X @ X.T) / X.shape[1]

# --------------------- NR estimator --------------------
def nr_bulk_estimator(eigs: np.ndarray, j: int) -> float:
    """
    \hat mu_p^{(j)} = (sum_{i>j} eigs_i) / (n-j)
    eigs assumed sorted descending, length n.
    """
    n = eigs.size
    tail_sum = eigs[j:].sum()
    return tail_sum / max(n - j, 1)

def nr_eigenvalues(SD: np.ndarray) -> np.ndarray:
    """
    Returns tilde_lambda for j=1..n-1 (last is set equal to max(0, smallest correction)).
    """
    eigs, U = np.linalg.eigh(SD)
    eigs = eigs[::-1]  # descending
    # tilde for 0..n-2
    tl = np.empty_like(eigs)
    for j in range(eigs.size - 1):
        mu_hat = nr_bulk_estimator(eigs, j + 1)  # subtract average of remaining after removing top j
        tl[j] = max(eigs[j] - mu_hat, 0.0)
    tl[-1] = max(eigs[-1] - 0.0, 0.0)
    return tl

# --------------------- Angles and metrics ---------------
def angle(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (norm(u) + 1e-12)
    v = v / (norm(v) + 1e-12)
    cos = float(np.clip(np.abs(u @ v), 0.0, 1.0))
    return float(np.arccos(cos))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

# --------------------- Inverse estimator S^{-1}_omega ----
def inverse_nr_estimator(X: np.ndarray, omega: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Constructs S_omega^{-1}:
      S_omega^{-1} = sum_{j=1}^{n-1} (max(tilde_lambda_j, omega))^{-1} \tilde h_j \tilde h_j^T
                      + omega^{-1}(I - sum \tilde h_j \tilde h_j^T)
    where \tilde h_j = (1/sqrt(n max(tilde_lambda_j, omega))) X \hat u_j.
    If omega is None, use omega = min(tr(S_D)/(p^{1/2} n^{1/4}), tr(S_D)/n).
    Returns matrix (p x p) and aux dict with eigs, U, tilde_lam, omega.
    """
    p, n = X.shape
    SD = dual_cov(X)
    evals, U = np.linalg.eigh(SD)
    # descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = U[:, idx]
    # bulk scales
    trace_SD = float(np.trace(SD))
    if omega is None:
        omega = min(trace_SD / (p**0.5 * n**0.25), trace_SD / n)
    # NR eigenvalues
    tilde = np.empty_like(evals)
    for j in range(n - 1):
        mu_hat = nr_bulk_estimator(evals, j + 1)
        tilde[j] = max(evals[j] - mu_hat, 0.0)
    tilde[-1] = max(evals[-1], 0.0)
    # build S^{-1}_omega
    S_oinv = np.zeros((p, p))
    P = np.zeros((p, p))
    for j in range(n - 1):
        lj = max(tilde[j], omega)
        hj = (X @ U[:, j]) / np.sqrt(n * lj + 1e-12)
        S_oinv += (1.0 / lj) * np.outer(hj, hj)
        P += np.outer(hj, hj)
    S_oinv += (1.0 / omega) * (np.eye(p) - P)
    aux = {"evals": evals, "U": U, "tilde": tilde, "omega": omega, "P": P}
    return S_oinv, aux
