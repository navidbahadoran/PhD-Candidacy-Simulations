"""
sim0_bbp_transition.py — BBP phase transition demo (Chapter 2).

Spiked covariance: Σ = diag(λ, 1, ..., 1) with a single spike λ > 1.
Sample covariance S = (1/n) X X^T with X ~ N(0, Σ). Aspect ratio γ = p/n.

- For λ <= (1 + sqrt(γ))^2 : the top sample eigenvalue sticks to MP edge b, and eigenvector alignment ~ 0.
- For λ  > (1 + sqrt(γ))^2 : an outlier eigenvalue emerges at φ(λ) = λ (1 + γ/(λ-1)), and alignment matches theory:
      |⟨\hat v, v⟩|^2  →  (1 - γ/(λ-1)^2) / (1 + γ/(λ-1))
  (Paul 2007; spiked covariance with unit noise variance).

We sweep λ across the threshold and compare empirical means to theory.
"""
import numpy as np

def bbp_outlier(lambda_pop, gamma):
    # Outlier location mapping for covariance spike (unit noise):
    # φ(λ) = λ (1 + γ / (λ - 1)), for λ > (1 + sqrt(γ))^2
    return lambda_pop * (1.0 + gamma / (lambda_pop - 1.0))

def bbp_alignment_sq(lambda_pop, gamma):
    # Squared cosine between top sample eigenvector and population spike direction
    # valid for λ > (1 + sqrt(γ))^2
    return (1.0 - gamma / (lambda_pop - 1.0)**2) / (1.0 + gamma / (lambda_pop - 1.0))

def run(p=1000, n=500, reps=50, lambdas=(1.2, 1.6, 2.0, 3.0, 5.0), seed=777):
    rng = np.random.default_rng(seed)
    gamma = p / n
    b = (1 + np.sqrt(gamma))**2

    print(f"[BBP] p={p}, n={n}, γ={gamma:.3f}, edge b={(b):.3f}")
    print("λ_pop | mean λ̂1 | theory φ(λ) (or b) | mean alignment^2 | theory alignment^2")
    for lam in lambdas:
        lhat = []
        align = []
        for r in range(reps):
            # Draw X with covariance diag(lam, 1,...,1)
            z = rng.standard_normal((p, n))
            x = z.copy()
            x[0, :] = np.sqrt(lam) * x[0, :]
            S = (x @ x.T) / n
            evals, vecs = np.linalg.eigh(S)
            idx = np.argmax(evals)
            lhat.append(evals[idx])
            vhat = vecs[:, idx]
            # population spike direction = e1
            align.append((vhat[0]**2))

        mh = float(np.mean(lhat))
        ma = float(np.mean(align))

        if lam <= b:
            theo_l = b
            theo_a = 0.0
        else:
            theo_l = bbp_outlier(lam, gamma)
            theo_a = bbp_alignment_sq(lam, gamma)

        print(f"{lam:5.2f} | {mh:9.3f} | {theo_l:16.3f} | {ma:16.3f} | {theo_a:18.3f}")

if __name__ == "__main__":
    run(p=1000, n=500, reps=40, lambdas=(1.2, 1.6, 2.0, 2.6, 3.5, 5.0), seed=778)
