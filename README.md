# Replication files for "Weak-instrument-robust subvector inference in instrumental variables regression: A subvector Lagrange multiplier test and properties of subvector Anderson-Rubin confidence sets"

This repository contains code to replicate results of https://arxiv.org/abs/2407.15256.
Start by creating a conda environment with the required dependencies:

```bash
conda env create --file environment.yml
conda activate ivmodels-simulations
pip install -e .
```

## Figures

### Figure 1: (Subvector) inverse Anderson-Rubin test confidence sets

```bash
python plots/figure_inverse_ar_different_alpha.py
```
will create `figures/figure_inverse_ar_different_alpha.pdf`.

### Figure 2: Power of tests for alpha=0.01 (top) and alpha=0.05 (bottom)

```bash
python experiments/guggenberger12_power_collect.py
```
collects data in a `json`. Pass `--n_cores xx` to parallelize.
```bash
python experiments/guggenberger12_power_aggregate.py
```
aggregates the data from the `json` and saves the result to `figures/guggenberger12_power/guggenberger12_power_n=1000_k=10_alphas=[0.05, 0.01].pdf`

### Figure 3: Empirical maximal rejection frequencies over tau in [0, pi) for k = 100 and Omega = Id

```bash
python experiments/kleibergen19_size_collect.py
```
collects data in an `h5`. This is computationally intensive.
```bash
python experiments/kleibergen19_size_aggregate.py
```
aggregates the data from the `h5` and saves the result to `figures/kleibergen19_size/figure_kleibergen19_identity_k100.pdf`

### Figure 5: Technical condition 2 holds / does not hold

```bash
python plots/tc2_counterexample.py
```
creates `figures/figure_tc2_counterexample.pdf`

### Figure 7: Optimization details

```bash
python plots/optimization.py
```
creates `figures/optimization.pdf`.

### Figure 8: QQ-plots of p-values under the null hypothesis against the uniform distribution

```bash
python experiments/guggenberger12_size_collect.py --k 5
python experiments/guggenberger12_size_collect.py --k 10
python experiments/guggenberger12_size_collect.py --k 15 &&\
python experiments/guggenberger12_size_collect.py --k 20 &&\
python experiments/guggenberger12_size_collect.py --k 30 &&
```
collects data and writes this to a `json`. This data is also used to create table 3.
Only values `k=10, k=20, k=30` are necessary for figure 8.
```bash
python experiments/guggenberger12_size_aggregate.py
```
aggregates the data. This creates `figures/guggenberger12_qqplots_n=1000.pdf`.
It additionally writes `tables/guggenberger12_size/guggenberger12_type_1_error_n=1000.tex`, containing the entries for table 3.

### Figures 9 - 13:

Same as figure 3, but additionally passing `--k 5` or `--k 20` and possibly `--cov_type guggenberger12`.

### Figure 14:

See table 4

## Tables

### Table 3

See figure 8.

### Table 4

Running
```bash
python applications/card1995using/card1995using.py
```
prints table 4. It also creates `figures/card_0.pdf` (figure 14).
It also prints table 8.

### Table 5
```bash
python applications/tanaka2006risk/tanaka2006risk.py
```
print table 5.

### Tables 6, 7

Same as table 3 (figure 8), but additionally supplying `--n 50` and `--n 100`.

### Table 8

See table 4.