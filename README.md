# Chapter 4 — Simulation Studies and Empirical Comparisons

This folder contains clean, runnable Python scripts to reproduce the experiments in **Chapter 4**:
they illustrate the HDLSS geometric laws and evaluate the **noise-reduction (NR) methodology**
for eigenvalues, PC directions/scores, and the inverse covariance estimator.

## Environment
- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, scikit-learn (only for the LDA demo)
- No internet access required.

Set a seed via `common.set_seed(seed)` for reproducibility.

## Scripts
1. **sim1_geometry.py** — Geometric representations (Section 3.4): scaled dual covariance approaches
   the identity under diffuse spectrum + weak dependence; diagonal limit under non-mixing proxy.
2. **sim2_eigenvalues_nr.py** — Eigenvalue bias and NR correction (Section 3.5): compares naive
   `\hat λ_j` vs. noise-reduced `\tilde λ_j` across spike strengths and sample sizes.
3. **sim3_pc_directions.py** — Consistency/inconsistency of sample PC directions (Section 3.6):
   tracks the angle between `\hat h_j` and `h_j` as a function of spike exponent `α` and `n`.
4. **sim4_pc_scores.py** — PC scores with and without NR (Section 3.7): in-sample MSE relative
   to population scores.
5. **sim5_inverse_covariance.py** — Inverse estimator `S^{-1}_ω` (Section 3.8): evaluates closeness
   of `V_ω = Λ^{1/2} H^T S^{-1}_ω H Λ^{1/2}` to the identity.
6. **sim6_discriminant_application.py** — Two-class HDLSS discriminant analysis using `S^{-1}_ω` vs.
   Moore–Penrose and ridge baselines (related to paper’s applications).

Each script prints key metrics; optional Matplotlib plots are saved under `figures/`.

## How to run
From this folder:
```bash
python sim1_geometry.py
python sim2_eigenvalues_nr.py
python sim3_pc_directions.py
python sim4_pc_scores.py
python sim5_inverse_covariance.py
python sim6_discriminant_application.py
```


### Chapter 2 add-ons
0. **sim0_mp_law.py** — Marchenko–Pastur (MP) law: histogram of eigenvalues vs. MP density and edge checks.
0. **sim0_bbp_transition.py** — BBP phase transition: spiked covariance outlier location and eigenvector alignment vs. theory.
