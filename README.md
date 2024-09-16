# Replication files for "Weak-instrument-robust subvector inference in instrumental variables regression: A subvector Lagrange multiplier test and properties of subvector Anderson-Rubin confidence sets"

This repository contains code to replicate results of https://arxiv.org/abs/2407.15256.
Start by creating a conda environment with the required dependencies:

```bash
conda env create --file environment.yml
conda activate ivmodels-simulations
pip install -e .
```

## Figure 1: (Subvector) inverse Anderson-Rubin test confidence sets

```bash
python plots/figure_inverse_ar_different_alpha.py
```
will create `figures/figure_inverse_ar_different_alpha.pdf`.

## Figure 2: Power of tests for alpha=0.01 (top) and alpha=0.05 (bottom)

```bash
python experiments/guggenberger12_power_collect.py
```
collects data in a `json`. Pass `--n_cores xx` to parallelize.
```bash
python experiments/guggenberger12_power_aggregate.py
```
aggregates the data from the `json` and saves the result to `figures/guggenberger12_power/guggenberger12_power_n=1000_k=10_alphas=[0.05, 0.01].pdf`

## Figure 3: Empirical maximal rejection frequencies over tau in [0, pi) for k = 100 and Omega = Id

```bash
python experiments/kleibergen19_size_collect.py
```
collects data in an `h5`. This is computationally intensive.
```bash
python experiments/kleibergen19_size_aggregate.py
```
aggregates the data from the `h5` and saves the result to `figures/kleibergen19_size/figure_kleibergen19_identity_k100.pdf`

## Figure 5: Technical condition 2 holds / does not hold

```bash
python plots/tc2_counterexample.py
```
creates `figures/figure_tc2_counterexample.pdf`

