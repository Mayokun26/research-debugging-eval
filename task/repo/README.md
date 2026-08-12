# Readmission cohort study

This repository compares two models for 30-day readmission in a synthetic
two-site cohort. The standard model is logistic regression using the most recent
value of each laboratory measurement. The comparison model is gradient boosting
using longitudinal summaries collected during the first 48 hours.

The primary analysis uses patient-level, site-stratified five-fold cross-validation.
Preprocessing is fit within each training fold, and both models use the same inner
validation protocol for their pre-registered parameter grids. The endpoint is
area under the receiver operating characteristic curve. Uncertainty for the paired
difference is estimated by patient bootstrap.

Run the analysis from this directory:

```bash
python run.py
```

Outputs are written to `results/` and final fitted estimators to `models/`.
Run the study checks with:

```bash
python -m pytest -q
```

The bundled records are entirely synthetic and contain no patient information.
