# Leakage control experiment

> DATA SOURCE: SYNTHETIC VALIDATION DATA — NOT a NASA result, NOT a scientific finding

## The question

Public Kepler classification notebooks routinely report 98-99% accuracy.
This experiment asks what that number is measuring.

## Result

| Feature set | Best model | Features | ROC-AUC | PR-AUC | F1 | Accuracy | Scientific validity |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline` | gradient_boosting | 22 | 0.9912 | 0.9922 | 0.9659 | 0.9633 | Valid |
| `physics_informed` | random_forest | 67 | 0.9956 | 0.9959 | 0.9708 | 0.9685 | Valid |
| `leaky_control` | random_forest | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | INVALID — target leakage |

![leakage comparison](figures/fig_leakage_comparison.png)

## Reading it

The tallest bars belong to the model that must never be used.

The NASA Exoplanet Archive defines `koi_disposition`'s non-CONFIRMED values
as taken from `koi_pdisposition`, the Kepler Robovetter's verdict, and
defines a false positive as an object that failed at least one of the tests
recorded in `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co` and
`koi_fpflag_ec`. `koi_score` is the fraction of Monte Carlo Robovetter
iterations returning CANDIDATE.

```
koi_fpflag_* --(rule)--> koi_pdisposition --(copied)--> koi_disposition
                               ^
                         koi_score
```

A model given those columns is not predicting. It is recovering a
deterministic function it was handed. The near-perfect score is the
signature of that recovery, not of a better classifier.

**A very high score on a tabular KOI classifier is evidence to investigate,
not evidence of success.**

## What the physics-informed set does instead

It recovers the *discriminating power* of those flags from quantities the
archive publishes anyway, by comparing independent measurements of the same
physical property:

| Feature | Two independent measurements of | Disagreement indicates |
|---|---|---|
| `f_abs_log_rho_ratio` | stellar density: fitted from the light curve vs. the stellar catalogue | dilution, eccentricity, or a blend |
| `f_duration_ratio` | transit duration: observed vs. the geometric maximum (`T/T₀ ≤ 1`) | impossible geometry, or grazing |
| `f_depth_ratio` | transit depth: measured vs. `(Rp/Rs)²` | third light in the aperture |

## Primary model selection

Selection metric: **pr_auc**.

> Primary. Sensitive to performance on the planet-candidate class and unaffected by the abundance of easy false positives. With mild imbalance and an asymmetric cost of error, this is the metric that tracks the scientific goal.

Selected: **random_forest** on **physics_informed** (pr_auc = 0.9959).

Selection is restricted to scientifically valid feature sets. The leaky
control is structurally excluded and cannot be selected however well it
scores.

## Rules this experiment establishes

1. The leaky control is never used for prediction, ranking, the API, or deployment.
2. Every leaky figure and table carries the `LEAKY CONTROL — NOT VALID FOR SCIENTIFIC USE` label.
3. Accuracy is never a selection metric.
