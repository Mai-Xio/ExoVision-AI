"""
Model comparison and the leakage experiment.

    python -m ml.evaluation.compare_models

Reads the artifacts written by ml.training.train and produces:

    artifacts/metrics/model_comparison.csv      full grid
    artifacts/metrics/leakage_experiment.csv    the three-row headline table
    artifacts/figures/fig_leakage_comparison.png
    artifacts/figures/fig_roc_curves.png
    artifacts/figures/fig_pr_curves.png
    artifacts/figures/fig_confusion_<best>.png
    artifacts/figures/fig_calibration.png
    docs/leakage_experiment.md

Model selection
---------------
Selection uses `SETTINGS.selection_metric` (default PR-AUC), not accuracy, and
is restricted to `VALID_FEATURE_SETS`. The leaky control is structurally
excluded — it cannot be selected however well it scores, because scoring well is
the thing it was built to do without earning it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import SETTINGS
from ml.config import features as fx
from ml.evaluation import plots
from ml.evaluation.metrics import SELECTION_METRIC_RATIONALE

logger = logging.getLogger(__name__)


def load_metric_files() -> list[dict[str, Any]]:
    out = []
    for path in sorted(SETTINGS.metrics_dir.glob("*.json")):
        if path.name in {"model_comparison.json", "leakage_experiment.json"}:
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("skipping unreadable %s", path.name)
    return out


def build_comparison(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        test = r.get("metrics", {}).get("test", {})
        cv = r.get("metrics", {}).get("cv", {})
        rows.append({
            "model": r["model_name"],
            "feature_set": r["feature_set"],
            "scientifically_valid": r.get("scientifically_valid", True),
            "n_features": r.get("n_features", np.nan),
            "accuracy": test.get("accuracy", np.nan),
            "precision": test.get("precision", np.nan),
            "recall": test.get("recall", np.nan),
            "f1": test.get("f1", np.nan),
            "roc_auc": test.get("roc_auc", np.nan),
            "pr_auc": test.get("pr_auc", np.nan),
            "balanced_accuracy": test.get("balanced_accuracy", np.nan),
            "mcc": test.get("mcc", np.nan),
            "brier": test.get("brier", np.nan),
            "cv_pr_auc_mean": cv.get("pr_auc_mean", np.nan),
            "cv_pr_auc_std": cv.get("pr_auc_std", np.nan),
        })
    return pd.DataFrame(rows)


def select_primary(comparison: pd.DataFrame, metric: str) -> pd.Series:
    """
    Pick the primary model.

    Restricted to scientifically valid feature sets. Ties broken by lower
    CV standard deviation — between two models with the same mean, prefer the
    one whose performance does not depend on which host stars landed in which
    fold.
    """
    valid = comparison[comparison["scientifically_valid"]].copy()
    if valid.empty:
        raise RuntimeError("no scientifically valid models found")
    valid = valid.sort_values([metric, "cv_pr_auc_std"], ascending=[False, True])
    return valid.iloc[0]


def leakage_table(comparison: pd.DataFrame, metric: str = "pr_auc") -> pd.DataFrame:
    """One row per feature set: its best model, and an explicit validity verdict."""
    best = (comparison.sort_values(metric, ascending=False)
            .groupby("feature_set", as_index=False).first())
    order = [fs for fs in ("baseline", "physics_informed", "leaky_control")
             if fs in set(best["feature_set"])]
    best = best.set_index("feature_set").loc[order].reset_index()
    best["scientific_validity"] = np.where(
        best["scientifically_valid"], "Valid", "INVALID — target leakage")
    best["description"] = best["feature_set"].map(fx.FEATURE_SET_DESCRIPTIONS)
    return best[["feature_set", "model", "n_features", "roc_auc", "pr_auc", "f1",
                 "accuracy", "scientific_validity", "description"]]


def _load_predictions(df: pd.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Test-set (y_true, y_score) for a `feature_set__model` tag."""
    pred_path = SETTINGS.metrics_dir / f"{tag}__test_pred.csv"
    if not pred_path.exists():
        return None
    pred = pd.read_csv(pred_path)
    y_true = df.loc[pred["row_index"], "target_binary"].to_numpy()
    return y_true, pred["y_proba"].to_numpy()


def write_markdown(
    leak: pd.DataFrame, primary: pd.Series, metric: str, banner: str, out: Path
) -> Path:
    def fmt(v: Any) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v)

    lines = [
        "# Leakage control experiment",
        "",
        f"> {banner}",
        "",
        "## The question",
        "",
        "Public Kepler classification notebooks routinely report 98-99% accuracy.",
        "This experiment asks what that number is measuring.",
        "",
        "## Result",
        "",
        "| Feature set | Best model | Features | ROC-AUC | PR-AUC | F1 | Accuracy | Scientific validity |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in leak.iterrows():
        lines.append(
            f"| `{r['feature_set']}` | {r['model']} | {int(r['n_features'])} | "
            f"{fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | {fmt(r['f1'])} | "
            f"{fmt(r['accuracy'])} | {r['scientific_validity']} |"
        )

    lines += [
        "",
        "![leakage comparison](figures/fig_leakage_comparison.png)",
        "",
        "## Reading it",
        "",
        "The tallest bars belong to the model that must never be used.",
        "",
        "The NASA Exoplanet Archive defines `koi_disposition`'s non-CONFIRMED values",
        "as taken from `koi_pdisposition`, the Kepler Robovetter's verdict, and",
        "defines a false positive as an object that failed at least one of the tests",
        "recorded in `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co` and",
        "`koi_fpflag_ec`. `koi_score` is the fraction of Monte Carlo Robovetter",
        "iterations returning CANDIDATE.",
        "",
        "```",
        "koi_fpflag_* --(rule)--> koi_pdisposition --(copied)--> koi_disposition",
        "                               ^",
        "                         koi_score",
        "```",
        "",
        "A model given those columns is not predicting. It is recovering a",
        "deterministic function it was handed. The near-perfect score is the",
        "signature of that recovery, not of a better classifier.",
        "",
        "**A very high score on a tabular KOI classifier is evidence to investigate,",
        "not evidence of success.**",
        "",
        "## What the physics-informed set does instead",
        "",
        "It recovers the *discriminating power* of those flags from quantities the",
        "archive publishes anyway, by comparing independent measurements of the same",
        "physical property:",
        "",
        "| Feature | Two independent measurements of | Disagreement indicates |",
        "|---|---|---|",
        "| `f_abs_log_rho_ratio` | stellar density: fitted from the light curve vs. the stellar catalogue | dilution, eccentricity, or a blend |",
        "| `f_duration_ratio` | transit duration: observed vs. the geometric maximum (`T/T₀ ≤ 1`) | impossible geometry, or grazing |",
        "| `f_depth_ratio` | transit depth: measured vs. `(Rp/Rs)²` | third light in the aperture |",
        "",
        "## Primary model selection",
        "",
        f"Selection metric: **{metric}**.",
        "",
        f"> {SELECTION_METRIC_RATIONALE.get(metric, '')}",
        "",
        f"Selected: **{primary['model']}** on **{primary['feature_set']}** "
        f"({metric} = {fmt(primary[metric])}).",
        "",
        "Selection is restricted to scientifically valid feature sets. The leaky",
        "control is structurally excluded and cannot be selected however well it",
        "scores.",
        "",
        "## Rules this experiment establishes",
        "",
        "1. The leaky control is never used for prediction, ranking, the API, or deployment.",
        "2. Every leaky figure and table carries the `LEAKY CONTROL — NOT VALID FOR SCIENTIFIC USE` label.",
        "3. Accuracy is never a selection metric.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default=SETTINGS.selection_metric)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    SETTINGS.ensure_dirs()

    records = load_metric_files()
    if not records:
        print("No trained models found. Run: python -m ml.training.train --include-leaky",
              file=sys.stderr)
        return 1

    meta_path = SETTINGS.processed_dir / "dataset_metadata.json"
    banner = (json.loads(meta_path.read_text())["banner"]
              if meta_path.exists() else SETTINGS.data_source_banner())

    comparison = build_comparison(records).sort_values(
        ["scientifically_valid", args.metric], ascending=[False, False])
    comparison.to_csv(SETTINGS.metrics_dir / "model_comparison.csv", index=False)

    leak = leakage_table(comparison, args.metric)
    leak.to_csv(SETTINGS.metrics_dir / "leakage_experiment.csv", index=False)

    figdir = SETTINGS.artifact_figures_dir
    plots.plot_leakage_comparison(comparison, figdir / "fig_leakage_comparison.png", banner)

    # Curves for the best model of each feature set.
    df = pd.read_parquet(SETTINGS.processed_dir / "koi_clean.parquet")
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    invalid: list[str] = []
    for _, r in leak.iterrows():
        tag = f"{r['feature_set']}__{r['model']}"
        loaded = _load_predictions(df, tag)
        if loaded is None:
            continue
        label = f"{r['feature_set']} / {r['model']}"
        curves[label] = loaded
        if r["scientific_validity"].startswith("INVALID"):
            invalid.append(label)

    if curves:
        plots.plot_roc_curves(curves, figdir / "fig_roc_curves.png", invalid, banner=banner)
        plots.plot_pr_curves(curves, figdir / "fig_pr_curves.png", invalid, banner=banner)
        valid_curves = {k: v for k, v in curves.items() if k not in invalid}
        if valid_curves:
            plots.plot_calibration(valid_curves, figdir / "fig_calibration.png", banner=banner)

    primary = select_primary(comparison, args.metric)
    primary_tag = f"{primary['feature_set']}__{primary['model']}"
    loaded = _load_predictions(df, primary_tag)
    if loaded is not None:
        y_true, y_score = loaded
        plots.plot_confusion(y_true, (y_score >= 0.5).astype(int),
                             figdir / f"fig_confusion_{primary_tag}.png",
                             title=f"Confusion matrix — {primary_tag}")

    # Record the primary model so downstream modules and the API agree on it.
    (SETTINGS.metrics_dir / "primary_model.json").write_text(json.dumps({
        "tag": primary_tag,
        "model": primary["model"],
        "feature_set": primary["feature_set"],
        "selection_metric": args.metric,
        "selection_rationale": SELECTION_METRIC_RATIONALE.get(args.metric, ""),
        "score": float(primary[args.metric]),
        "artifact": str((SETTINGS.models_dir / f"{primary_tag}.joblib").name),
        "data_source_banner": banner,
    }, indent=2), encoding="utf-8")

    md = write_markdown(leak, primary, args.metric, banner,
                        SETTINGS.repo_root / "docs" / "leakage_experiment.md")

    print("=" * 78)
    print(banner)
    print("=" * 78)
    print("\nLEAKAGE EXPERIMENT")
    print(leak.drop(columns=["description"]).round(4).to_string(index=False))
    print(f"\nPRIMARY MODEL: {primary_tag}  ({args.metric} = {primary[args.metric]:.4f})")
    print(f"  rationale: {SELECTION_METRIC_RATIONALE.get(args.metric, '')}")
    print(f"\nfigures -> {figdir}\ndoc -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
