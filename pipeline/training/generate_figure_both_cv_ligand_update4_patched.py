#!/usr/bin/env python3
"""
Generate Comprehensive ML Analysis Figure

This script trains ML models and generates the comprehensive 6-panel figure
showing model comparison, ROC curves, PR curves, confusion matrix, and key findings.

This is the SAME figure as ml_comprehensive_analysis.png - the one you liked!

Usage:
    python generate_comprehensive_figure.py \
        --features fingerprint_summary_with_components.csv \
        --output comprehensive_ml_results.png
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import warnings

# Table III generator: prefer project implementation, fall back to a local implementation.
try:
    from pipeline.training.generate_table_iii_median_iqr import generate_table_iii  # type: ignore
except Exception:
    def _median_iqr(vals: list[float], decimals: int = 1) -> str:
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return ""
        med = np.median(a)
        q1 = np.percentile(a, 25)
        q3 = np.percentile(a, 75)
        return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"

    def generate_table_iii(
        results_by_strategy_fpr: dict,
        frozen_model_by_strategy: dict,
        operating_points: list[tuple[int, float]],
        output_csv: str,
        output_latex: str,
        debug: bool = False,
        long_ns: float = 20.0,
    ) -> pd.DataFrame:
        """Generate Table III (operating point recommendations) from fold_ops.

        This implementation is leakage-safe: thresholds are selected on the TRAIN split
        for each fold at the requested FPR target, then evaluated on the TEST split.
        It summarizes fold-level metrics as median [Q1, Q3].

        Expected input shape:
          results_by_strategy_fpr[fpr_key][strategy][time_ns][model] contains:
            - 'fold_ops': list of dicts with keys:
                time_ns, fpr_target, recall, achieved_fpr, triage_rate, cost_saved, model
        """
        # Map strategies to deployment regimes + use-case labels (paper-facing names)
        strategy_to_regime = {
            "protein_logo": "Novel proteins, primary deployment",
            "pocket_gkf": "Robustness: within-protein generalization to unseen pockets (known protein, new pocket)",
        }

        rows = []
        # iterate both strategies (ignore fpr_key duplication; we filter by op['fpr_target'])
        any_fpr_key = next(iter(results_by_strategy_fpr.keys()))
        strategies = results_by_strategy_fpr[any_fpr_key].keys()

        for strategy in strategies:
            model_name = frozen_model_by_strategy.get(strategy, "Logistic Regression")
            by_time = results_by_strategy_fpr[any_fpr_key][strategy]

            for (t_ns, fpr_target) in operating_points:
                # Locate fold_ops for this timescale+model
                fold_ops = []
                if t_ns in by_time and model_name in by_time[t_ns]:
                    fold_ops = by_time[t_ns][model_name].get("fold_ops", []) or []
                # Filter to this operating point and model
                ops = [op for op in fold_ops
                       if float(op.get("fpr_target", -1)) == float(fpr_target)
                       and op.get("model") == model_name
                       and int(op.get("time_ns", -1)) == int(t_ns)]
                if not ops:
                    if debug:
                        log.warning(f"No fold_ops for {strategy} {model_name} t={t_ns} fpr={fpr_target}")
                    continue

                # Collect fold-level metrics; convert to percent for table display
                recall_pct = [100.0 * float(op.get("recall", np.nan)) for op in ops]
                term_pct   = [100.0 * float(op.get("triage_rate", np.nan)) for op in ops]  # terminate = predicted unstable
                cost_pct   = [100.0 * float(op.get("cost_saved", np.nan)) for op in ops]
                fpr_pct    = [100.0 * float(op.get("achieved_fpr", np.nan)) for op in ops]

                rows.append({
                    "Deployment regime": strategy_to_regime.get(strategy, strategy),
                    "Operating point": f"{int(t_ns)} ns @ FPR $\\le$ {int(round(100*fpr_target))}\\%" ,
                    "Recall (%)": _median_iqr(recall_pct, 1),
                    "Terminated (%)": _median_iqr(term_pct, 1),
                    "Cost saved (%)": _median_iqr(cost_pct, 1),
                    "Achieved FPR (%)": _median_iqr(fpr_pct, 1),
                    "_strategy": strategy,
                    "_time_ns": int(t_ns),
                    "_fpr_target": float(fpr_target),
                })

        df = pd.DataFrame(rows)
        # Stable sort for paper readability
        df = df.sort_values(by=["_strategy", "_time_ns", "_fpr_target"]).drop(columns=["_strategy","_time_ns","_fpr_target"])

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        # LaTeX export: compact 7-column table (paper-safe)
        def _latex_escape(s: str) -> str:
            return s.replace("&", "\\&").replace("%", "\\%")

        with open(output_latex, "w") as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{FPR-calibrated operating-point performance for early MD triage (Ligand~1). Values are median [Q1, Q3] across cross-validation folds.}\n")
            f.write("\\label{tab:operationaltriage}\n")
            f.write("\\renewcommand{\\arraystretch}{1.15}\n")
            f.write("\\small\n")
            f.write("\\begin{tabular}{l l c c c c}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Operating point} & \\textbf{Use case} & \\textbf{Recall (\\%)} & \\textbf{Terminated (\\%)} & \\textbf{Cost saved (\\%)} & \\textbf{Achieved FPR (\\%)}\\\\\n")
            f.write("\\midrule\n")

            # Group by deployment regime
            for regime, sub in df.groupby("Deployment regime", sort=False):
                f.write(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{_latex_escape(regime)}}}}}\\\\\n")
                for _, r in sub.iterrows():
                    # Use-case text can be filled by user in manuscript if desired; keep blank here
                    f.write(f"{_latex_escape(r['Operating point'])} &  & {r['Recall (%)']} & {r['Terminated (%)']} & {r['Cost saved (%)']} & {r['Achieved FPR (%)']}\\\\\n")
                f.write("\\midrule\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")

        if debug:
            log.info(f"✓ Wrote Table III CSV: {output_csv}")
            log.info(f"✓ Wrote Table III LaTeX: {output_latex}")
        return df


warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",   # Windows / macOS
        "Times",             # macOS fallback
        "Nimbus Roman",      # Linux
        "DejaVu Serif"       # Matplotlib default (always exists)
    ],
    "font.size": 12,                 # base font size
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
})

RESULTS = []

TIMESCALES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
CV_SETTINGS = {
    "protein": {
        "group_by": "protein",
        "cv_mode": "logo",
        "label": "Cross-protein (LOPO)"
    },
    "pocket": {
        "group_by": "pocket",
        "cv_mode": "gkf",
        "label": "Cross-pocket (protein|pocket)"
    }
}
FEATURE_SETS = {
    # Core feature sets
    "F5_struct_energy": ["slope", "rmsd_var", "mean_disp", "var_disp", "energy_std"],  # BASE (full)
    "F4_struct": ["slope", "rmsd_var", "mean_disp", "var_disp"],  # STRUCT4 (no energy)
    "F1_energy": ["energy_std"],  # Energy alone

    # Leave-one-feature-out (LOFO) variants for detailed ablation
    "LOFO_no_slope": ["rmsd_var", "mean_disp", "var_disp", "energy_std"],
    "LOFO_no_rmsd_var": ["slope", "mean_disp", "var_disp", "energy_std"],
    "LOFO_no_mean_disp": ["slope", "rmsd_var", "var_disp", "energy_std"],
    "LOFO_no_var_disp": ["slope", "rmsd_var", "mean_disp", "energy_std"],
    "LOFO_no_energy_std": ["slope", "rmsd_var", "mean_disp", "var_disp"],  # Same as F4_struct

    # Single-feature models (for interpretability)
    "SINGLE_slope": ["slope"],
    "SINGLE_rmsd_var": ["rmsd_var"],
    "SINGLE_mean_disp": ["mean_disp"],
    "SINGLE_var_disp": ["var_disp"],
}

# Ablation experiment configuration
ABLATION_FEATURE_SETS = {
    # Minimal ablation: BASE vs STRUCT4 vs ENERGY1
    "minimal": ["F5_struct_energy", "F4_struct", "F1_energy"],
    # Full LOFO ablation
    "lofo": ["F5_struct_energy", "LOFO_no_slope", "LOFO_no_rmsd_var",
             "LOFO_no_mean_disp", "LOFO_no_var_disp", "LOFO_no_energy_std"],
    # Single-feature ablation
    "single": ["SINGLE_slope", "SINGLE_rmsd_var", "SINGLE_mean_disp",
               "SINGLE_var_disp", "F1_energy"],
    # Complete ablation (all variants)
    "full": list(FEATURE_SETS.keys()),
}
FEATURE_LABELS = {
    "slope": "RMSD Drift (β)",
    "rmsd_var": "RMSD Variance",
    "mean_disp": "Mean Cα Displacement",
    "var_disp": "Cα Displacement Variance",
    "energy_std": "Interaction Energy Fluctuation (σE)",
}


# Default feature columns (can be overridden by --feature_set argument)
FEATURE_COLS = [
    'slope',
    'rmsd_var',
    'mean_disp',
    'var_disp',
]

def get_feature_cols(feature_set: str = "F4_struct"):
    """
    Return the appropriate feature columns based on feature_set choice.

    Args:
        feature_set: Either "F4_struct" (4 features) or "F5_struct_energy" (5 features)

    Returns:
        List of feature column names
    """
    if feature_set == "F5_struct_energy":
        return ["slope", "rmsd_var", "mean_disp", "var_disp", "energy_std"]
    else:  # Default to F4_struct
        return ["slope", "rmsd_var", "mean_disp", "var_disp"]


def _apply_core_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent filtering across all functions.

    This ensures dataset size (N) is identical in training, evaluation,
    and all reporting/plotting functions.

    Filters applied:
    1. Exclude problematic pockets (MDR2_TRIRC pocket9, SNQ2_CANGA pocket2)
    2. Remove rows where label_unstable is NaN
    3. Replace inf/-inf with NaN (for numeric safety)

    Returns:
        Filtered DataFrame (copy)
    """
    exclude = (
             ((df["protein"] == "MDR2_TRIRC") & (df["pocket"] == "pocket9")) |
             ((df["protein"] == "SNQ2_CANGA") & (df["pocket"] == "pocket2"))
    )
    df = df[~exclude].copy()
    df = df[df["label_unstable"].notna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

from sklearn.base import clone

def select_best_time_and_model(results_by_timescale, metric="auc"):
    """
    Return (best_time, best_model_name, best_score) across all timescales and models.
    """
    best_time = None
    best_model = None
    best_score = -np.inf

    for t, models in results_by_timescale.items():
        for model_name, r in models.items():
            score = r.get(metric, None)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_time = t
                best_model = model_name

    return best_time, best_model, best_score


def load_data(features_path, timescale, group_by='protein', feature_cols=None):

    """
    Load and prepare data.

    Args:
        features_path: Path to CSV
        timescale: Timescale to extract
        group_by: 'protein' (leave-one-protein-out),
                  'pocket' (pocket-level CV with protein|pocket groups), or
                  'ligand_pocket' (protein|ligand|pocket groups - if ligand_name exists)
        feature_cols: List of feature column names to use. If None, uses global FEATURE_COLS.
    """
    # Use provided feature_cols or fall back to global default
    if feature_cols is None:
        feature_cols = FEATURE_COLS.copy()
    else:
        feature_cols = list(feature_cols)  # Make a copy to avoid modifying the original

    df = pd.read_csv(features_path)

    # Check if ligand_name column exists
    has_ligand = 'ligand_name' in df.columns

    # Apply consistent core filters (pocket exclusions, label notna, inf handling)
    df = _apply_core_filters(df)

    df_t = df[df['time_ns'] == timescale].copy()

    missing = [c for c in feature_cols if c not in df_t.columns]
    if missing:
        # If the only missing column is energy, fall back to F4.
        # Otherwise, raise so you don't silently train on wrong inputs.
        if set(missing) == {"energy_std"}:
            log.warning(f"energy_std column not found, falling back to 4-feature set")
            feature_cols = [c for c in feature_cols if c != "energy_std"]
        else:
            raise ValueError(f"Missing required feature columns: {missing}")

    # Drop rows with NaN in features (after inf replacement in _apply_core_filters)
    df_t = df_t.dropna(subset=feature_cols + ['label_unstable', 'protein'])

    X = df_t[feature_cols].values
    y = df_t['label_unstable'].values


    if group_by == 'ligand_pocket':
        if has_ligand:
            # Group by protein|ligand|pocket to keep replicas together
            groups = (df_t['protein'].astype(str) + '|' +
                      df_t['ligand_name'].astype(str) + '|' +
                      df_t['pocket'].astype(str)).values
        else:
            log.warning("group_by='ligand_pocket' but 'ligand_name' column missing; "
                        "falling back to protein|pocket grouping.")
            groups = (df_t['protein'].astype(str) + '|' + df_t['pocket'].astype(str)).values
    elif group_by == 'pocket':
        # Group by protein|pocket to keep replicas together
        groups = (df_t['protein'].astype(str) + '|' + df_t['pocket'].astype(str)).values
    else:
        # Original: group by protein only
        groups = df_t['protein'].values

    return X, y, groups


def get_models():
    """Return models to evaluate."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', C=3, random_state=42

        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=3,
            class_weight='balanced', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, random_state=42
        ),
        # 'Neural Network': MLPClassifier(
        #     hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        #     max_iter=500, random_state=42, early_stopping=True
        # )
    }


def train_model_cv(X, y, groups, model, cv_mode='logo', target_fpr=0.10,
                   fpr_targets=None, t_ns=None, long_ns=20.0):
    """
    Train model with group-aware cross-validation.

    Args:
        cv_mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold with 5 splits)
        fpr_targets: list of FPR targets for operating point evaluation (e.g., [0.10, 0.20])
        t_ns: current timescale in ns (needed for cost_saved calculation)
        long_ns: baseline simulation length for cost calculation

    Returns:
        y_true_all, y_prob_all, y_pred_all, fold_aucs, fold_pr_aucs, fold_ops
        where fold_ops is a list of per-fold operating point metrics
    """
    from sklearn.model_selection import GroupKFold

    if fpr_targets is None:
        fpr_targets = [0.10, 0.20]

    if cv_mode == 'gkf':
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    else:
        cv = LeaveOneGroupOut()

    scaler = StandardScaler()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    fold_aucs = []
    fold_pr_aucs = []
    fold_ops = []  # NEW: per-fold operating point metrics

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        cloned_model = clone(model)
        cloned_model.fit(X_train_scaled, y_train)

        # Probabilities on train (for threshold selection) and test
        y_prob_train = cloned_model.predict_proba(X_train_scaled)[:, 1]
        y_prob_test = cloned_model.predict_proba(X_test_scaled)[:, 1]

        # Default threshold using target_fpr (backward compatible)
        thr, _, _ = _threshold_at_target_fpr(y_train, y_prob_train, target_fpr=target_fpr)
        y_pred_test = (y_prob_test >= thr).astype(int)

        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob_test)
        y_pred_all.extend(y_pred_test)

        if len(np.unique(y_test)) > 1:
            fold_aucs.append(roc_auc_score(y_test, y_prob_test))
            fold_pr_aucs.append(average_precision_score(y_test, y_prob_test))

        # === NEW: Operating-point evaluation per fold ===
        for fpr_target in fpr_targets:
            # Select threshold on TRAIN set at this FPR target
            tau, _, _ = _threshold_at_target_fpr(y_train, y_prob_train, fpr_target)

            if not np.isfinite(tau):
                continue

            y_hat = (y_prob_test >= tau).astype(int)

            # Confusion matrix elements
            tp = ((y_hat == 1) & (y_test == 1)).sum()
            fn = ((y_hat == 0) & (y_test == 1)).sum()
            fp = ((y_hat == 1) & (y_test == 0)).sum()
            tn = ((y_hat == 0) & (y_test == 0)).sum()

            # Metrics
            recall = tp / (tp + fn + 1e-8)
            achieved_fpr = fp / (fp + tn + 1e-8)
            triage_rate = y_hat.mean()

            # Cost saved (only if t_ns is provided)
            cost_saved = np.nan
            if t_ns is not None:
                cost_saved = triage_rate * (long_ns - t_ns) / long_ns

            fold_ops.append({
                "fold": fold_id,
                "time_ns": t_ns,
                "fpr_target": fpr_target,
                "recall": float(recall),
                "achieved_fpr": float(achieved_fpr),
                "triage_rate": float(triage_rate),
                "cost_saved": float(cost_saved),
                "n_test": len(y_test),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "tn": int(tn),
            })

    return (np.array(y_true_all), np.array(y_prob_all), np.array(y_pred_all),
            np.array(fold_aucs), np.array(fold_pr_aucs), fold_ops)


def _log_cv_fold_summary(y, groups, cv, header: str = ""):
    """Log per-fold test-set sizes and class counts (useful for LOPO small-n interpretation)."""
    y = np.asarray(y).astype(int)
    uniq_groups = np.unique(groups)
    fold_sizes = []
    if header:
        log.info(header)
    for fold_id, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        y_te = y[test_idx]
        n_te = len(test_idx)
        n_pos = int(y_te.sum())
        n_neg = int(n_te - n_pos)
        held_out = np.unique(groups[test_idx])
        held_out_str = held_out[0] if len(held_out) == 1 else f"{len(held_out)} groups"
        fold_sizes.append(n_te)
        log.info(
            f"  Fold {fold_id:2d}: n_test={n_te:2d} (unstable={n_pos:2d}, stable={n_neg:2d}), held_out={held_out_str}"
        )
    if fold_sizes:
        fold_sizes = np.asarray(fold_sizes)
        log.info(
            f"  Test-set size across folds: min={fold_sizes.min()}, median={int(np.median(fold_sizes))}, max={fold_sizes.max()} (n_folds={len(fold_sizes)}, n_groups={len(uniq_groups)})"
        )


def evaluate_all_models(
        features_path: Path,
        timescales: list,
        group_by: str = "protein",
        cv_mode: str = "logo",
        GOAL_FPR: float = 0.10,
        fpr_targets: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        long_ns: float = 20.0,
        print_fold_summary: bool = False,
        fold_summary_timescale: float = 2.0,
        feature_cols: list = None,
        model_whitelist: list = None,
):
    """
    Train and evaluate all models at all timescales.

    Metrics are reported as mean ± std across CV folds.
    Now also collects per-fold operating point metrics for Table III.

    Args:
        features_path: Path to features CSV
        timescales: List of timescales to evaluate
        group_by: Grouping strategy for CV
        cv_mode: CV mode ('logo' or 'gkf')
        GOAL_FPR: Target FPR for threshold selection
        fpr_targets: Tuple of FPR targets for operating point evaluation
        long_ns: Baseline simulation length for cost calculation
        print_fold_summary: Whether to print per-fold test-set sizes
        fold_summary_timescale: Timescale at which to print fold summary
        feature_cols: List of feature columns to use (4 or 5 features)
        model_whitelist: If provided, only run these models (e.g., ["Logistic Regression", "Random Forest"])
    """
    # Use default if not provided
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    log.info("=" * 80)
    log.info("TRAINING MODELS")
    log.info("=" * 80)
    log.info(f"Grouping strategy: {group_by}")
    log.info(f"CV mode: {cv_mode}")
    log.info(f"Feature set: {feature_cols}")

    results_by_timescale = {}
    all_fold_ops = []  # NEW: collect all fold-level operating points
    models_dict = get_models()

    # Filter models if whitelist provided
    if model_whitelist is not None:
        models_dict = {k: v for k, v in models_dict.items() if k in set(model_whitelist)}
        log.info(f"Model whitelist: {list(models_dict.keys())}")

    for t in timescales:
        log.info(f"\nTimescale: {t} ns")
        X, y, groups = load_data(features_path, t, group_by=group_by, feature_cols=feature_cols)

        log.info(
            f"  Samples: {len(X)}, "
            f"Unstable: {int(y.sum())} ({100 * y.mean():.1f}%)"
        )
        log.info(f"  Unique groups: {len(np.unique(groups))}")

        # Optional: print per-fold test-set sizes (helpful for LOPO small-n interpretation)
        if print_fold_summary and float(t) == float(fold_summary_timescale):
            from sklearn.model_selection import GroupKFold
            if cv_mode == 'gkf':
                cv_dbg = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            else:
                cv_dbg = LeaveOneGroupOut()
            _log_cv_fold_summary(
                y=y,
                groups=groups,
                cv=cv_dbg,
                header=f"Per-fold test-set sizes at t={t} ns ({CV_SETTINGS.get('protein' if cv_mode=='logo' else 'pocket', {}).get('label', cv_mode)}; group_by={group_by})"
            )

        results_by_timescale[t] = {}

        for model_name, model in models_dict.items():
            try:
                (
                    y_true,
                    y_prob,
                    y_pred,
                    fold_aucs,
                    fold_pr_aucs,
                    fold_ops,  # NEW: per-fold operating point metrics
                ) = train_model_cv(
                    X,
                    y,
                    groups,
                    model,
                    cv_mode=cv_mode,
                    target_fpr=GOAL_FPR,
                    fpr_targets=list(fpr_targets),
                    t_ns=t,
                    long_ns=long_ns,
                )

                # Guard against degenerate cases
                if len(fold_aucs) == 0:
                    log.warning(
                        f"    {model_name:20s}: "
                        "Skipped (no valid CV folds with both classes)"
                    )
                    continue

                auc_mean = float(np.mean(fold_aucs))
                auc_std = float(np.std(fold_aucs))
                pr_mean = float(np.mean(fold_pr_aucs))
                pr_std = float(np.std(fold_pr_aucs))

                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # Add model name to fold_ops for later aggregation
                for op in fold_ops:
                    op["model"] = model_name
                all_fold_ops.extend(fold_ops)

                results_by_timescale[t][model_name] = {
                    "auc": auc_mean,
                    "auc_std": auc_std,
                    "pr_auc": pr_mean,
                    "pr_auc_std": pr_std,
                    "n_folds": len(fold_aucs),
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                    "fold_ops": fold_ops,  # NEW: store fold-level ops
                }

                log.info(
                    f"    {model_name:20s}: "
                    f"AUC={auc_mean:.3f}±{auc_std:.3f}, "
                    f"PR={pr_mean:.3f}±{pr_std:.3f}, "
                    f"Recall={recall:.3f}"
                )

            except Exception as e:
                log.warning(f"    {model_name}: Failed - {e}")

    # Store all fold ops in results for later aggregation
    results_by_timescale["_fold_ops"] = all_fold_ops

    return results_by_timescale

#figure 1
def plot_timescale_horizon(results_dict, output="figure1_timescale_horizon.png",
                           frozen_model=None):
    """
    Plot AUC and PR-AUC vs timescale.

    Args:
        results_dict: Dict of results by strategy
        output: Output file path
        frozen_model: If provided, use this model for all timescales (e.g., "Logistic Regression").
                      If None, selects best model per timescale (for exploration only).
    """
    plt.figure(figsize=(6.5, 4.8))

    colors = {
        "protein_logo": "#4C72B0",   # muted blue (Cross-protein)
        "pocket_gkf":   "#DD8452",   # muted orange (Cross-pocket)
    }


    for key, results in results_dict.items():
        times, roc_vals, pr_vals, roc_stds, pr_stds = [], [], [], [], []

        for t in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]:
            if t not in results:
                continue

            # Use frozen model if provided, otherwise select best per timescale
            model_t = frozen_model if frozen_model else select_best_model(results[t], metric="auc")

            if model_t is None or model_t not in results[t]:
                continue

            r = results[t][model_t]
            times.append(t)
            roc_vals.append(r["auc"])
            roc_stds.append(r["auc_std"])
            pr_vals.append(r["pr_auc"])
            pr_stds.append(r["pr_auc_std"])
            print(
                f"{key}  t={t}  model={model_t}  "
                f"AUC={r['auc']:.3f}±{r['auc_std']:.3f}  "
                f"PR={r['pr_auc']:.3f}±{r['pr_auc_std']:.3f}"
            )

        label = "Cross-protein CV" if key == "protein_logo" else "Cross-pocket CV"

        plt.plot(times, roc_vals, marker="o", linewidth=1.8,
                 color=colors[key], label=f"{label} – ROC–AUC")
        plt.plot(times, pr_vals, marker="s", linestyle="--", linewidth=1.8,
                 color=colors[key], label=f"{label} – PR–AUC")

    plt.axvspan(2, 7, color="#999999", alpha=0.08)
    plt.text(6.0, 0.84, "Predictive horizon\n(2–7 ns)",
             ha="center", va="top", fontsize=10, style="italic", color="#444")

    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("ROC–AUC / PR–AUC")
    plt.title("")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


from sklearn.metrics import roc_curve

def recall_at_fpr(y_true, y_prob, target_fpr=0.2):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = tpr[fpr <= target_fpr]
    return valid.max() if len(valid) > 0 else 0.0

def select_best_model(results_at_timescale, metric="auc"):
    """
    Select the best model for a single timescale based on a metric.
    Default: highest ROC-AUC.
    Returns model_name or None.
    """
    best_model = None
    best_score = -np.inf

    for model_name, r in results_at_timescale.items():
        if metric in r and r[metric] is not None:
            if r[metric] > best_score:
                best_score = r[metric]
                best_model = model_name

    return best_model

def plot_triage_efficiency(results_dict, output="figure2_triage_efficiency.png",
                           fpr_targets=(0.10, 0.20), frozen_model=None):
    """
    Plot recall vs timescale for different FPR targets.

    Uses fold_ops (train-calibrated thresholds) to avoid leakage.

    Args:
        results_dict: Dict of results by strategy
        output: Output file path
        fpr_targets: FPR targets to plot
        frozen_model: If provided (e.g., "Logistic Regression"), use only this model.
                      If None, selects best model per timescale (for exploration only,
                      not recommended for paper figures as it's model selection on test).
    """
    plt.figure(figsize=(6.5, 4.5))

    styles = {
        0.20: dict(linestyle="-",  marker="o"),
        0.10: dict(linestyle="--", marker="s"),
    }

    for key, results in results_dict.items():
        label_base = "Cross-protein" if key == "protein_logo" else "Cross-pocket"

        for fpr_t in fpr_targets:
            times, recalls = [], []

            for t in TIMESCALES:
                if t not in results:
                    continue

                if frozen_model:
                    # Use frozen model (recommended for paper)
                    if frozen_model not in results[t]:
                        continue
                    r = results[t][frozen_model]
                    fold_ops = r.get("fold_ops", [])
                    if fold_ops:
                        rec, _, _ = _metrics_from_fold_ops(fold_ops, fpr_t)
                    else:
                        rec = recall_at_fpr(r["y_true"], r["y_prob"], fpr_t)

                    if np.isfinite(rec):
                        times.append(t)
                        recalls.append(rec)
                else:
                    # Choose best model at this timescale (exploration only)
                    best_model = None
                    best_recall = -1

                    for model_name, r in results[t].items():
                        fold_ops = r.get("fold_ops", [])

                        if fold_ops:
                            rec, _, _ = _metrics_from_fold_ops(fold_ops, fpr_t)
                        else:
                            rec = recall_at_fpr(r["y_true"], r["y_prob"], fpr_t)

                        if np.isfinite(rec) and rec > best_recall:
                            best_recall = rec
                            best_model = model_name

                    if best_model is not None:
                        times.append(t)
                        recalls.append(best_recall)

            model_label = f" ({frozen_model})" if frozen_model else " (best)"
            st = styles.get(fpr_t, dict(linestyle="-", marker="o"))
            plt.plot(times, recalls, linewidth=2.5,
                     label=f"{label_base}{model_label}, FPR ≤ {int(fpr_t*100)}%",
                     **st)

    plt.axvspan(2, 7, color="gray", alpha=0.15, label="Early MD (2–7 ns)")
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Recall")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_triage_efficiency_bar(
        results_dict,
        output="figure2_triage_efficiency.png",
        target_fpr=0.2,
        frozen_model=None,
):
    """
    Bar plot of recall vs timescale.

    Uses fold_ops (train-calibrated thresholds) to avoid leakage.

    Args:
        results_dict: Dict of results by strategy
        output: Output file path
        target_fpr: FPR target
        frozen_model: If provided, use only this model. If None, selects best per timescale.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.5))

    times = [t for t in TIMESCALES if 2 <= t <= 14]

    # Bar layout
    bar_width = 0.35
    x = np.arange(len(times))

    for i, (key, results) in enumerate(results_dict.items()):
        recalls = []

        for t in times:
            if t not in results:
                recalls.append(np.nan)
                continue

            if frozen_model:
                # Use frozen model (recommended for paper)
                if frozen_model not in results[t]:
                    recalls.append(np.nan)
                    continue
                r = results[t][frozen_model]
            else:
                # Select best model (exploration only)
                best_model = select_best_model(results[t], metric="auc")
                if best_model is None:
                    recalls.append(np.nan)
                    continue
                r = results[t][best_model]

            fold_ops = r.get("fold_ops", [])

            if fold_ops:
                recall, _, _ = _metrics_from_fold_ops(fold_ops, target_fpr)
            else:
                recall = recall_at_fpr(r["y_true"], r["y_prob"], target_fpr)

            recalls.append(recall)

        label = "Cross-protein" if key == "protein_logo" else "Cross-pocket"

        plt.bar(
            x + i * bar_width,
            recalls,
            width=bar_width,
            label=label,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            )

    # Highlight early-screening regime
    early_idx = [i for i, t in enumerate(times) if 2 <= t <= 7]
    if early_idx:
        plt.axvspan(
            min(early_idx) - 0.5,
            max(early_idx) + 0.5,
            color="gray",
            alpha=0.12,
            label="Early MD (2–7 ns)",
            )

    model_label = f" ({frozen_model})" if frozen_model else " (best)"
    plt.xticks(x + bar_width / 2, times)
    plt.xlabel("MD time window (ns)")
    plt.ylabel(f"Recall at FPR = {target_fpr}{model_label}")
    plt.title("")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    log.info(f"✓ Saved {output}")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# already in your file:
# from scipy.stats import mannwhitneyu

def _format_p(p: float) -> str:
    if p < 1e-4:
        return "p<1e-4"
    if p < 1e-3:
        return "p<1e-3"

    return f"p={p:.3f}"

def _bh_fdr(pvals: list[float]) -> list[float]:
    """
    Benjamini–Hochberg FDR correction (no extra deps).
    Returns q-values in original order.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()

def plot_early_time_separation_compact(
        features_path: Path,
        timescale=2,
        output="early_separation_compact_2ns.png",
        feature_cols=None,
        show_pvalues=True,
        fdr_correct=True,
        save_stats_csv=True
):
    """
    Plot violin plots showing feature separation between stable/unstable at a given timescale.
    Optionally annotate each panel with a two-sided Mann–Whitney U p-value (and BH-FDR q-value).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = pd.read_csv(features_path)
    df = _apply_core_filters(df)
    df = df[df["time_ns"] == timescale].copy()
    df = df.dropna(subset=feature_cols + ["label_unstable"])
    df["Stability"] = df["label_unstable"].map({0: "Stable", 1: "Unstable"})

    log.info(f"  plot_early_time_separation_compact: N={len(df)} at t={timescale}ns")

    # Compute per-feature p-values (Stable vs Unstable)
    stats_rows = []
    pvals = []
    for feat in feature_cols:
        stable = df.loc[df["Stability"] == "Stable", feat].astype(float).values
        unstable = df.loc[df["Stability"] == "Unstable", feat].astype(float).values

        # guard: if either class is empty
        if len(stable) == 0 or len(unstable) == 0:
            p = np.nan
        else:
            # Mann–Whitney U is robust for heavy-tailed / non-normal distributions
            _, p = mannwhitneyu(stable, unstable, alternative="two-sided")

        pvals.append(p)
        stats_rows.append({
            "time_ns": timescale,
            "feature": feat,
            "n_stable": len(stable),
            "n_unstable": len(unstable),
            "median_stable": np.nanmedian(stable) if len(stable) else np.nan,
            "median_unstable": np.nanmedian(unstable) if len(unstable) else np.nan,
            "p_mwu": p
        })

    # Optional BH-FDR correction across the 4 panels
    if show_pvalues and fdr_correct:
        qvals = _bh_fdr([pv if np.isfinite(pv) else 1.0 for pv in pvals])
        for r, q in zip(stats_rows, qvals):
            r["q_bh_fdr"] = q
    else:
        for r in stats_rows:
            r["q_bh_fdr"] = np.nan

    # Plot grid
    n_features = len(feature_cols)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    FEATURE_LABELS = {
        "slope": "RMSD Drift (β)",
        "rmsd_var": "RMSD Variance",
        "mean_disp": "Mean Cα Displacement",
        "var_disp": "Cα Displacement Variance",
        "energy_std": "Interaction Energy Fluctuation (σE)",
    }

    # Use your ACM-ish palette (current screenshot look)
    palette = ["#D9D9D9", "#7F7F7F"]  # Stable, Unstable

    for i, feat in enumerate(feature_cols):
        if i >= len(axes):
            break
        ax = axes[i]

        sns.violinplot(
            data=df,
            x="Stability",
            y=feat,
            palette=palette,
            cut=0,
            inner="box",
            linewidth=1,
            ax=ax
        )

        title = FEATURE_LABELS.get(feat, feat)

        # Add p-value annotation
        if show_pvalues:
            p = stats_rows[i]["p_mwu"]
            q = stats_rows[i]["q_bh_fdr"]
            if np.isfinite(p):
                txt = _format_p(p)
                if fdr_correct and np.isfinite(q):
                    txt += f"\nq={q:.3f}"
            else:
                txt = "p=NA"

            # place at top-left inside axes
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10
            )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    log.info(f"✓ Saved {output}")

    if save_stats_csv:
        stats_df = pd.DataFrame(stats_rows)
        stats_out = str(output).replace(".png", "_stats.csv")
        stats_df.to_csv(stats_out, index=False)
        log.info(f"✓ Saved {stats_out}")

from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, mannwhitneyu


def generate_feature_importance_table_for_model(
        features_path: Path,
        timescale: float,
        model_name: str,
        output_csv: Path,
        group_by: str = 'protein',
        feature_cols: list = None
):
    """
    Compute feature importance for the selected best model at the selected timescale.
    Uses permutation importance (AUC drop) for comparability across models.
    For LR adds standardized coefficients; for tree models adds feature_importances_.

    Args:
        features_path: Path to features CSV
        timescale: Timescale to use
        model_name: Name of model to evaluate
        output_csv: Output CSV path
        group_by: Grouping strategy (for reporting only)
        feature_cols: List of feature columns to use. If None, uses global FEATURE_COLS.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    log.info(f"\nGenerating feature importance at {timescale}ns using {model_name}")
    log.info(f"  Feature set: {feature_cols}")

    # Load data with consistent filtering
    df = pd.read_csv(features_path)
    df = _apply_core_filters(df)
    df_t = df[df['time_ns'] == timescale].copy()

    # Drop NaN in features
    df_t = df_t.dropna(subset=feature_cols)

    X = df_t[feature_cols].values
    y = df_t['label_unstable'].values

    # Groups only used for reporting/consistency; importance is fit on all data
    if group_by == 'pocket':
        groups = (df_t['protein'].astype(str) + '|' + df_t['pocket'].astype(str)).values
    else:
        groups = df_t['protein'].values

    # Standardize (keep consistent with training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build the same model from your pool
    models_dict = get_models()
    if model_name not in models_dict:
        raise ValueError(f"Unknown model_name={model_name}. Must be one of {list(models_dict.keys())}")

    model = clone(models_dict[model_name])
    model.fit(X_scaled, y)

    # Permutation importance (AUC drop) — comparable across models
    perm = permutation_importance(
        model,
        X_scaled,
        y,
        scoring='roc_auc',
        n_repeats=50,
        random_state=42
    )

    # Optional grounding: correlation with rmsd_late_20ns if present
    has_rmsd_late = 'rmsd_late_20ns' in df_t.columns

    rows = []
    for i, feat in enumerate(feature_cols):
        rho = np.nan
        if has_rmsd_late:
            rho, _ = spearmanr(df_t[feat], df_t['rmsd_late_20ns'])

        row = {
            "feature": feat,
            "perm_auc_drop": perm.importances_mean[i],
            "perm_auc_drop_std": perm.importances_std[i],
            "spearman_rmsd_late": rho
        }

        # Model-specific importance
        if hasattr(model, "coef_"):  # Logistic Regression
            row["coef"] = float(model.coef_[0][i])
            row["coef_abs"] = float(abs(model.coef_[0][i]))
        if hasattr(model, "feature_importances_"):  # RF / GB
            row["tree_importance"] = float(model.feature_importances_[i])

        rows.append(row)

    importance_df = pd.DataFrame(rows)

    # Sort by permutation importance (primary, comparable across models)
    importance_df = importance_df.sort_values("perm_auc_drop", ascending=False)

    importance_df.to_csv(output_csv, index=False)

    log.info(f"✓ Saved feature importance table: {output_csv}")
    log.info("Top drivers (by permutation ΔAUC):")
    for _, r in importance_df.head(10).iterrows():
        log.info(f"  {r['feature']:12s} | ΔAUC={r['perm_auc_drop']:.3f}")

    return importance_df


def _threshold_at_target_fpr(y_true, y_prob, target_fpr: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)

    fpr, tpr, thr = roc_curve(y_true, y_prob)

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return (np.nan, np.nan, np.nan)

    # Use as much FPR budget as possible
    best_fpr_idx = valid[np.argmax(fpr[valid])]

    # If multiple thresholds share this FPR, choose highest TPR
    same_fpr = valid[np.isclose(fpr[valid], fpr[best_fpr_idx], atol=1e-6)]
    best = same_fpr[np.argmax(tpr[same_fpr])]

    return float(thr[best]), float(tpr[best]), float(fpr[best])



def _triage_rate_and_actual_fpr(y_true, y_prob, thr: float):
    """
    Given a threshold, return:
      triage_rate = P(pred_unstable)  (how many are rejected early)
      actual_fpr  = FP / (FP + TN)
    """
    if not np.isfinite(thr):
        return (np.nan, np.nan)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    triage_rate = float(np.mean(y_pred == 1))

    # confusion_matrix: rows=true (0,1), cols=pred (0,1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    actual_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
    return (triage_rate, actual_fpr)


def _cost_saved(triage_rate: float, t_ns: float, long_ns: float):
    """
    Simple single-decision policy: run t_ns for everyone, then stop early for predicted-unstable.
    Baseline always runs long_ns.
    """
    if not np.isfinite(triage_rate):
        return np.nan
    return float(triage_rate * (long_ns - t_ns) / long_ns)


# ============================================================================
# TABLE III: Operating Point Aggregation Functions
# ============================================================================

def summarize_operating_points(fold_ops, model_name="Logistic Regression"):
    """
    Aggregate fold-level operating point metrics for Table III.

    Args:
        fold_ops: list of dicts with keys: fold, time_ns, fpr_target, recall,
                  achieved_fpr, triage_rate, cost_saved, model
        model_name: which model to filter for (default: Logistic Regression)

    Returns:
        dict: (time_ns, fpr_target) -> {recall_mean, recall_std, triage_mean, ...}
    """
    summary = {}

    # Filter to selected model
    ops = [r for r in fold_ops if r.get("model") == model_name]

    # Get unique (time_ns, fpr_target) combinations
    combos = set((r["time_ns"], r["fpr_target"]) for r in ops if r["time_ns"] is not None)

    for (t_ns, fpr) in sorted(combos):
        rows = [r for r in ops if r["time_ns"] == t_ns and r["fpr_target"] == fpr]

        if len(rows) == 0:
            continue

        # Use sample std (ddof=1) for proper fold-level reporting
        n = len(rows)
        summary[(t_ns, fpr)] = {
            "n_folds": n,
            "recall_mean": np.mean([r["recall"] for r in rows]),
            "recall_std": np.std([r["recall"] for r in rows], ddof=1) if n > 1 else 0.0,
            "triage_mean": np.mean([r["triage_rate"] for r in rows]),
            "triage_std": np.std([r["triage_rate"] for r in rows], ddof=1) if n > 1 else 0.0,
            "cost_mean": np.mean([r["cost_saved"] for r in rows]),
            "cost_std": np.std([r["cost_saved"] for r in rows], ddof=1) if n > 1 else 0.0,
            "fpr_mean": np.mean([r["achieved_fpr"] for r in rows]),
            "fpr_std": np.std([r["achieved_fpr"] for r in rows], ddof=1) if n > 1 else 0.0,
        }

    return summary

# Convenience function to format a single metric as median[IQR]
def format_median_iqr(values, decimals=1):
    """
    Format an array of values as median[Q1, Q3].

    Args:
        values: array-like of numeric values
        decimals: number of decimal places

    Returns:
        str: formatted string like "60.8 [32.1, 89.4]"
    """
    values = np.array(values)
    median = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return f"{median:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"

def export_screening_report_csv(
        results_dict: dict,
        output_csv: str,
        long_ns: float = 20.0,
        fpr_targets=(0.10, 0.20),
):
    """
    Export screening report using fold_ops (train-calibrated thresholds applied to test).

    This version avoids leakage by using the per-fold operating point metrics
    computed during CV (threshold selected on train, evaluated on test) rather
    than re-selecting thresholds on pooled out-of-fold predictions.

    results_dict can be:
      - results_by_timescale (dict[t][model] = metrics)
      - OR dict[strategy_name] -> results_by_timescale (when group_by == 'both')
    """
    rows = []

    # Normalize input shape:
    # If values look like dict[timescale], treat as single strategy.
    is_multi_strategy = True
    if len(results_dict) > 0:
        any_key = next(iter(results_dict.keys()))
        # crude check: if top-level keys are timescales (numbers), it's single strategy
        if isinstance(any_key, (int, float, np.integer, np.floating)):
            is_multi_strategy = False

    strategies = results_dict.items() if is_multi_strategy else [("single", results_dict)]

    for strategy_name, by_time in strategies:
        # Try to parse "protein_logo" or "pocket_gkf"
        group_by = None
        cv_mode = None
        if strategy_name != "single" and "_" in strategy_name:
            parts = strategy_name.split("_")
            if len(parts) >= 2:
                group_by, cv_mode = parts[0], parts[1]

        # Filter to only numeric timescale keys (skip '_fold_ops' and other metadata)
        timescale_keys = [k for k in by_time.keys() if isinstance(k, (int, float, np.integer, np.floating))]

        for t_ns in sorted(timescale_keys):
            for model_name, r in by_time[t_ns].items():
                fold_ops = r.get("fold_ops", [])

                # Fall back to old behavior if fold_ops is empty
                if not fold_ops:
                    y_true = r.get("y_true", None)
                    y_prob = r.get("y_prob", None)
                    if y_true is None or y_prob is None:
                        continue

                    row = {
                        "strategy": strategy_name,
                        "group_by": group_by,
                        "cv_mode": cv_mode,
                        "timescale_ns": float(t_ns),
                        "model": model_name,
                        "auc": r.get("auc", np.nan),
                        "pr_auc": r.get("pr_auc", np.nan),
                        "recall": r.get("recall", np.nan),
                        "precision": r.get("precision", np.nan),
                        "f1": r.get("f1", np.nan),
                    }

                    # Old behavior (threshold on pooled) - only used if fold_ops missing
                    for fpr_t in fpr_targets:
                        thr, tpr_at, fpr_at = _threshold_at_target_fpr(y_true, y_prob, target_fpr=fpr_t)
                        triage_rate, actual_fpr = _triage_rate_and_actual_fpr(y_true, y_prob, thr)
                        cost_saved = _cost_saved(triage_rate, t_ns=float(t_ns), long_ns=float(long_ns))

                        row[f"recall@fpr{fpr_t:.2f}"] = tpr_at
                        row[f"triage_rate@fpr{fpr_t:.2f}"] = triage_rate
                        row[f"cost_saved@fpr{fpr_t:.2f}"] = cost_saved
                        row[f"actual_fpr@fpr{fpr_t:.2f}"] = actual_fpr
                        row["source"] = "pooled_y_prob"

                    rows.append(row)
                    continue

                # Preferred: use fold_ops (train-calibrated thresholds, no leakage)
                row = {
                    "strategy": strategy_name,
                    "group_by": group_by,
                    "cv_mode": cv_mode,
                    "timescale_ns": float(t_ns),
                    "model": model_name,
                    "auc": r.get("auc", np.nan),
                    "auc_std": r.get("auc_std", np.nan),
                    "pr_auc": r.get("pr_auc", np.nan),
                    "pr_auc_std": r.get("pr_auc_std", np.nan),
                    "source": "fold_ops",
                }

                for fpr_t in fpr_targets:
                    # Filter fold_ops for this FPR target
                    ops = [op for op in fold_ops if float(op.get("fpr_target", -1)) == float(fpr_t)]
                    if not ops:
                        row[f"recall@fpr{fpr_t:.2f}"] = np.nan
                        row[f"triage_rate@fpr{fpr_t:.2f}"] = np.nan
                        row[f"cost_saved@fpr{fpr_t:.2f}"] = np.nan
                        row[f"actual_fpr@fpr{fpr_t:.2f}"] = np.nan
                        row[f"n_folds@fpr{fpr_t:.2f}"] = 0
                        continue

                    # Pool counts across folds (consistent with Table III)
                    tp = sum(int(op.get("tp", 0)) for op in ops)
                    fn = sum(int(op.get("fn", 0)) for op in ops)
                    fp = sum(int(op.get("fp", 0)) for op in ops)
                    tn = sum(int(op.get("tn", 0)) for op in ops)
                    n_test = sum(int(op.get("n_test", 0)) for op in ops)

                    recall = tp / (tp + fn + 1e-8)
                    achieved_fpr = fp / (fp + tn + 1e-8)
                    triage_rate = (tp + fp) / (n_test + 1e-8)
                    cost_saved = triage_rate * (long_ns - float(t_ns)) / long_ns

                    row[f"recall@fpr{fpr_t:.2f}"] = float(recall)
                    row[f"triage_rate@fpr{fpr_t:.2f}"] = float(triage_rate)
                    row[f"cost_saved@fpr{fpr_t:.2f}"] = float(cost_saved)
                    row[f"actual_fpr@fpr{fpr_t:.2f}"] = float(achieved_fpr)
                    row[f"n_folds@fpr{fpr_t:.2f}"] = len(ops)

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

def generate_comparison_figure(results_dict, output_path, features_path, feature_cols=None):
    """Generate figure comparing protein-level vs pocket-level CV.

    Args:
        results_dict: Dict of results by strategy
        output_path: Output file path
        features_path: Path to features CSV
        feature_cols: List of feature columns (for consistent filtering)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    log.info("\n" + "="*80)
    log.info("GENERATING COMPARISON FIGURE")
    log.info("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'protein_logo': 'steelblue', 'pocket_gkf': 'darkgreen'}
    labels = {'protein_logo': 'Leave-one-protein-out', 'pocket_gkf': 'Pocket-level GroupKFold'}

    # Panel 1: AUC comparison
    ax = axes[0]
    for key, results in results_dict.items():
        aucs = []
        times = []
        for t in TIMESCALES:
            if t not in results:
                continue

            best_model = select_best_model(results[t], metric="auc")
            if best_model is None:
                continue

            aucs.append(results[t][best_model]['auc'])
            times.append(t)


        if aucs:
            ax.plot(times, aucs, marker='o', linewidth=2.5, markersize=8,
                    label=labels[key], color=colors[key], alpha=0.8)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
    ax.set_xlabel('Simulation Time (ns)', fontsize=13)
    ax.set_ylabel('ROC-AUC (best model per timescale)', fontsize=13)
    ax.set_title('Model Performance: Protein vs Pocket CV', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    # Panel 2: Performance table at 2ns
    ax2 = axes[1]
    ax2.axis('off')

    # Load and filter data consistently with training
    df = pd.read_csv(features_path)
    df = _apply_core_filters(df)
    df_2ns = df[df['time_ns'] == 2].copy()
    # Drop NaNs in features to match training N exactly
    df_2ns = df_2ns.dropna(subset=feature_cols + ['label_unstable', 'protein'])

    table_data = [['CV Strategy', '2ns AUC', 'Recall', 'Precision', 'Groups']]

    for key, results in results_dict.items():
        if 2 not in results:
            continue

        best_model = select_best_model(results[2], metric="auc")
        if best_model is None:
            continue

        r = results[2][best_model]

        # Compute group counts from filtered data
        if key == 'protein_logo':
            n_groups = df_2ns['protein'].nunique()
        else:
            n_groups = (df_2ns['protein'].astype(str) + '|' +
                        df_2ns['pocket'].astype(str)).nunique()

        table_data.append([
            labels[key],
            f"{r['auc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            str(n_groups)
        ])


    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                      bbox=[0, 0.3, 1, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('2ns Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f"✓ Saved comparison figure: {output_path}")
    plt.close()

def load_results_from_pkl(pkl_path: Path):
    log.info(f"✓ Loading results from {pkl_path}")
    with open(pkl_path, "rb") as f:
        results_dict = pickle.load(f)

    required = {"protein_logo", "pocket_gkf"}
    missing = required - set(results_dict.keys())
    if missing:
        raise ValueError(f"Results file missing keys: {missing}")

    return results_dict

def compute_dataset_stats(features_path: Path, timescales=None):
    """
    Compute dataset statistics per timescale.

    Uses _apply_core_filters to ensure consistent N with training/evaluation.
    """
    if timescales is None:
        timescales = [1]
    df = pd.read_csv(features_path)
    # Apply consistent core filters
    df = _apply_core_filters(df)

    rows = []
    for t in timescales:
        df_t = df[df['time_ns'] == t]
        n_total = len(df_t)
        n_unstable = int(df_t['label_unstable'].sum())
        n_stable = n_total - n_unstable

        # Also report unique proteins/pockets for context
        n_proteins = df_t['protein'].nunique() if 'protein' in df_t.columns else np.nan
        n_pockets = df_t['pocket'].nunique() if 'pocket' in df_t.columns else np.nan

        rows.append({
            "timescale_ns": t,
            "n_total": n_total,
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "unstable_frac": n_unstable / n_total if n_total > 0 else np.nan,
            "n_proteins": n_proteins,
            "n_pockets": n_pockets,
        })

    return pd.DataFrame(rows)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def _metrics_from_y_pred(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    triage_rate = float(np.mean(np.array(y_pred) == 1))  # fraction flagged for early stop
    actual_fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return recall, triage_rate, actual_fpr

# Note: _cost_saved is defined earlier in the file (around line 922)
# Removed duplicate definition here to avoid confusion

def _metrics_from_fold_ops(fold_ops, fpr_target):
    """
    Aggregate operating-point metrics at a given fpr_target using fold_ops.
    Uses pooled counts across folds (more stable than unweighted averaging).
    """
    if not fold_ops:
        return np.nan, np.nan, np.nan

    ops = [op for op in fold_ops if float(op.get("fpr_target", -1)) == float(fpr_target)]
    if not ops:
        return np.nan, np.nan, np.nan

    tp = sum(int(op.get("tp", 0)) for op in ops)
    fn = sum(int(op.get("fn", 0)) for op in ops)
    fp = sum(int(op.get("fp", 0)) for op in ops)
    tn = sum(int(op.get("tn", 0)) for op in ops)
    n_test = sum(int(op.get("n_test", 0)) for op in ops)

    recall = tp / (tp + fn + 1e-8)
    triage_rate = (tp + fp) / (n_test + 1e-8)
    achieved_fpr = fp / (fp + tn + 1e-8)
    return float(recall), float(triage_rate), float(achieved_fpr)

def plot_triage_and_cost_frozen_model(
        results_by_strategy_fpr,          # dict: target_fpr -> results_dict
        frozen_model_by_strategy,         # dict: strategy_key -> model_name
        TIMESCALES,
        long_ns=20.0,
        out_recall="triage_recall_frozen.png",
        out_cost=f"triage_cost_frozen.png",
):
    labels = {'protein_logo': 'Cross-protein CV', 'pocket_gkf': 'Cross-pocket CV'}
    colors = {
        'protein_logo': '#4C72B0',   # muted blue
        'pocket_gkf':   '#DD8452',   # muted orange
    }


    # Style: FPR=0.20 is solid 'o-', FPR=0.10 is dashed 's--'
    styles = {
        (0.20, "protein_logo"): dict(linestyle="-",  marker="o", color=colors['protein_logo'], linewidth=2, markersize=8),
        (0.20, "pocket_gkf"):   dict(linestyle="-",  marker="o", color=colors['pocket_gkf'],   linewidth=2, markersize=8),
        (0.10, "protein_logo"): dict(linestyle="--", marker="s", color=colors['protein_logo'], linewidth=1.5, markersize=6, alpha=0.7),
        (0.10, "pocket_gkf"):   dict(linestyle="--", marker="s", color=colors['pocket_gkf'],   linewidth=1.5, markersize=6, alpha=0.7),
    }


    # -------- Recall plot --------
    plt.figure(figsize=(6.5, 4.5))
    for target_fpr, results_dict in results_by_strategy_fpr.items():
        for strategy_key, by_time in results_dict.items():
            model_name = frozen_model_by_strategy[strategy_key]
            fpr_label = f"FPR≤{int(target_fpr*100)}%"

            xs, ys = [], []
            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "fold_ops" not in r:
                    continue

                recall, _, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                if np.isfinite(recall):
                    xs.append(t)
                    ys.append(recall * 100)  # Convert to percentage

            st = styles.get((float(target_fpr), strategy_key), {})
            plt.plot(xs, ys, label=f"{labels[strategy_key]} ({fpr_label})", **st)

    plt.axvspan(2, 7, color="#999999", alpha=0.08)
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Recall (%)")
    plt.ylim(0, 100)
    plt.xlim(0, 15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_recall, dpi=300, bbox_inches="tight")
    plt.close()

    colors = {
        'protein_logo': '#4C72B0',   # muted blue
        'pocket_gkf':   '#DD8452',   # muted orange
    }

    # Style: FPR=0.20 is solid 'o-', FPR=0.10 is dashed 's--'
    styles = {
        (0.20, "protein_logo"): dict(linestyle="-",  marker="o", color=colors['protein_logo'], linewidth=2, markersize=8),
        (0.20, "pocket_gkf"):   dict(linestyle="-",  marker="o", color=colors['pocket_gkf'],   linewidth=2, markersize=8),
        (0.10, "protein_logo"): dict(linestyle="--", marker="s", color=colors['protein_logo'], linewidth=1.5, markersize=6, alpha=0.7),
        (0.10, "pocket_gkf"):   dict(linestyle="--", marker="s", color=colors['pocket_gkf'],   linewidth=1.5, markersize=6, alpha=0.7),
    }
    # -------- Cost-saved plot --------
    plt.figure(figsize=(6.5, 4.5))
    for target_fpr, results_dict in results_by_strategy_fpr.items():
        for strategy_key, by_time in results_dict.items():
            model_name = frozen_model_by_strategy[strategy_key]
            fpr_label = f"FPR≤{int(target_fpr*100)}%"

            xs, ys = [], []
            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "fold_ops" not in r:
                    continue

                _, triage_rate, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                cs = _cost_saved(triage_rate, t_ns=t, long_ns=long_ns)
                if np.isfinite(cs):
                    xs.append(t)
                    ys.append(cs * 100)  # Convert to percentage

            st = styles.get((float(target_fpr), strategy_key), {})
            plt.plot(xs, ys, label=f"{labels[strategy_key]} ({fpr_label})", **st)

    plt.axvspan(2, 7, color="#999999", alpha=0.08)
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Computational Cost Saved (%)")
    plt.ylim(0, 55)
    plt.xlim(0, 15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_cost, dpi=300, bbox_inches="tight")
    plt.close()

    # -------- Print summary table --------
    print("\n" + "="*90)
    print("TRIAGE PERFORMANCE SUMMARY")
    print("="*90)
    print(f"{'Strategy':<18} {'FPR':<6} {'t(ns)':<6} {'Recall%':<10} {'CostSaved%':<12} {'AUC±std':<16} {'PR-AUC±std':<16}")
    print("-"*90)

    for target_fpr in sorted(results_by_strategy_fpr.keys()):
        results_dict = results_by_strategy_fpr[target_fpr]
        for strategy_key in ['protein_logo', 'pocket_gkf']:
            if strategy_key not in results_dict:
                continue
            by_time = results_dict[strategy_key]
            model_name = frozen_model_by_strategy[strategy_key]

            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "y_true" not in r or "y_pred" not in r:
                    continue

                recall, triage_rate, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                cs = _cost_saved(triage_rate, t_ns=t, long_ns=long_ns)
                auc = r.get("auc", np.nan)
                auc_std = r.get("auc_std", np.nan)
                pr_auc = r.get("pr_auc", np.nan)
                pr_auc_std = r.get("pr_auc_std", np.nan)

                print(f"{labels[strategy_key]:<18} {target_fpr:<6.2f} {t:<6} "
                      f"{recall*100:>6.1f}%    {cs*100:>6.1f}%      "
                      f"{auc:.3f}±{auc_std:.3f}      {pr_auc:.3f}±{pr_auc_std:.3f}")
        print("-"*90)

    print("="*90 + "\n")


# ============================================================================
# ABLATION EXPERIMENT FUNCTIONS
# ============================================================================

def run_ablation_experiment(
        features_path: Path,
        ablation_type: str = "minimal",
        timescales: list = None,
        cv_regimes: list = None,
        models_to_run: list = None,
        fpr_targets: tuple = (0.10, 0.20),
        long_ns: float = 20.0,
        output_dir: Path = None,
        date_tag: str = None,
):
    """
    Run ablation experiment comparing different feature sets.

    Args:
        features_path: Path to features CSV
        ablation_type: Type of ablation ("minimal", "lofo", "single", "full")
        timescales: List of timescales to evaluate (default: [2, 3, 5])
        cv_regimes: List of CV regimes to use (default: ["protein_logo", "pocket_gkf"])
        models_to_run: List of model names (default: ["Random Forest", "Logistic Regression"])
        fpr_targets: FPR targets for operating point evaluation
        long_ns: Baseline simulation length for cost calculation
        output_dir: Output directory for results
        date_tag: Date tag for output files

    Returns:
        Dict containing ablation results
    """
    if timescales is None:
        timescales = [2, 3, 5]  # Focused horizons for ablation
    if cv_regimes is None:
        cv_regimes = ["protein_logo", "pocket_gkf"]
    if models_to_run is None:
        models_to_run = ["Random Forest", "Logistic Regression"]
    if output_dir is None:
        output_dir = Path("./outputs")
    if date_tag is None:
        date_tag = "ablation"

    # Get feature sets for this ablation type
    feature_set_names = ABLATION_FEATURE_SETS.get(ablation_type, ABLATION_FEATURE_SETS["minimal"])

    log.info("\n" + "="*80)
    log.info(f"ABLATION EXPERIMENT: {ablation_type.upper()}")
    log.info("="*80)
    log.info(f"Feature sets: {feature_set_names}")
    log.info(f"Timescales: {timescales}")
    log.info(f"CV regimes: {cv_regimes}")
    log.info(f"Models: {models_to_run}")
    log.info(f"FPR targets: {fpr_targets}")

    # Store all results
    ablation_results = {
        "config": {
            "ablation_type": ablation_type,
            "feature_sets": feature_set_names,
            "timescales": timescales,
            "cv_regimes": cv_regimes,
            "models": models_to_run,
            "fpr_targets": fpr_targets,
        },
        "results": {},
        "fold_ops_all": [],
    }

    # Run evaluation for each feature set
    for fs_name in feature_set_names:
        if fs_name not in FEATURE_SETS:
            log.warning(f"Unknown feature set: {fs_name}, skipping")
            continue

        feature_cols = FEATURE_SETS[fs_name]
        log.info(f"\n{'='*60}")
        log.info(f"Feature set: {fs_name} ({len(feature_cols)} features)")
        log.info(f"Features: {feature_cols}")
        log.info(f"{'='*60}")

        ablation_results["results"][fs_name] = {}

        # Run each CV regime
        for cv_regime in cv_regimes:
            if cv_regime == "protein_logo":
                group_by = "protein"
                cv_mode = "logo"
            elif cv_regime == "pocket_gkf":
                # Use protein|pocket grouping for cross-pocket CV
                # (consistent with CV_SETTINGS description)
                group_by = "pocket"
                cv_mode = "gkf"
            else:  # ligand_pocket_gkf (if needed)
                group_by = "ligand_pocket"
                cv_mode = "gkf"

            log.info(f"\n  CV regime: {cv_regime} (group_by={group_by}, cv_mode={cv_mode})")

            try:
                results = evaluate_all_models(
                    features_path=features_path,
                    timescales=timescales,
                    group_by=group_by,
                    cv_mode=cv_mode,
                    GOAL_FPR=fpr_targets[0],
                    fpr_targets=fpr_targets,
                    long_ns=long_ns,
                    feature_cols=feature_cols,
                    model_whitelist=models_to_run,  # Pass model whitelist
                )

                # Store results
                ablation_results["results"][fs_name][cv_regime] = results

                # Collect fold_ops with feature set info
                if "_fold_ops" in results:
                    for op in results["_fold_ops"]:
                        op["feature_set"] = fs_name
                        op["n_features"] = len(feature_cols)
                        op["cv_regime"] = cv_regime
                    ablation_results["fold_ops_all"].extend(results["_fold_ops"])

            except Exception as e:
                log.error(f"  Error running {fs_name} / {cv_regime}: {e}")
                ablation_results["results"][fs_name][cv_regime] = None

    # Save raw results
    results_path = output_dir / f"ablation_results_{ablation_type}_{date_tag}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(ablation_results, f)
    log.info(f"\n✓ Saved ablation results: {results_path}")

    return ablation_results


def generate_ablation_table(
        ablation_results: dict,
        output_csv: Path,
        output_latex: Path = None,
        reference_set: str = "F5_struct_energy",
        frozen_model: str = "Logistic Regression",
        fpr_target: float = 0.20,
        timescale: float = 3,
):
    """
    Generate ablation comparison table.

    Args:
        ablation_results: Results from run_ablation_experiment
        output_csv: Path to output CSV
        output_latex: Path to output LaTeX (optional)
        reference_set: Feature set to use as baseline for delta computations
        frozen_model: Model to use for comparison
        fpr_target: FPR target to report
        timescale: Timescale to report

    Returns:
        DataFrame with ablation table
    """
    log.info("\n" + "="*80)
    log.info("GENERATING ABLATION TABLE")
    log.info("="*80)
    log.info(f"Reference set: {reference_set}")
    log.info(f"Model: {frozen_model}")
    log.info(f"FPR target: {fpr_target}")
    log.info(f"Timescale: {timescale} ns")

    rows = []
    reference_metrics = {}

    results = ablation_results["results"]

    # First pass: compute reference metrics
    if reference_set in results:
        for cv_regime in results[reference_set]:
            if results[reference_set][cv_regime] is None:
                continue
            by_time = results[reference_set][cv_regime]
            if timescale not in by_time:
                continue
            if frozen_model not in by_time[timescale]:
                continue

            r = by_time[timescale][frozen_model]
            fold_ops = r.get("fold_ops", [])

            if fold_ops:
                recall, triage_rate, achieved_fpr = _metrics_from_fold_ops(fold_ops, fpr_target)
                cost_saved = _cost_saved(triage_rate, t_ns=timescale, long_ns=20.0)

                reference_metrics[cv_regime] = {
                    "recall": recall,
                    "triage_rate": triage_rate,
                    "achieved_fpr": achieved_fpr,
                    "cost_saved": cost_saved,
                    "auc": r.get("auc", np.nan),
                    "pr_auc": r.get("pr_auc", np.nan),
                }

    # Second pass: compute metrics for all feature sets
    for fs_name, fs_results in results.items():
        feature_cols = FEATURE_SETS.get(fs_name, [])

        for cv_regime, by_time in fs_results.items():
            if by_time is None:
                continue
            if timescale not in by_time:
                continue
            if frozen_model not in by_time[timescale]:
                continue

            r = by_time[timescale][frozen_model]
            fold_ops = r.get("fold_ops", [])

            if not fold_ops:
                continue

            recall, triage_rate, achieved_fpr = _metrics_from_fold_ops(fold_ops, fpr_target)
            cost_saved = _cost_saved(triage_rate, t_ns=timescale, long_ns=20.0)

            # Compute deltas vs reference
            ref = reference_metrics.get(cv_regime, {})
            delta_recall = recall - ref.get("recall", recall)
            delta_cost = cost_saved - ref.get("cost_saved", cost_saved)
            delta_auc = r.get("auc", np.nan) - ref.get("auc", r.get("auc", np.nan))

            row = {
                "feature_set": fs_name,
                "n_features": len(feature_cols),
                "features": ", ".join(feature_cols),
                "cv_regime": cv_regime,
                "timescale_ns": timescale,
                "model": frozen_model,
                "fpr_target": fpr_target,
                "auc": r.get("auc", np.nan),
                "auc_std": r.get("auc_std", np.nan),
                "pr_auc": r.get("pr_auc", np.nan),
                "pr_auc_std": r.get("pr_auc_std", np.nan),
                "recall": recall,
                "triage_rate": triage_rate,
                "cost_saved": cost_saved,
                "achieved_fpr": achieved_fpr,
                "delta_recall_vs_base": delta_recall,
                "delta_cost_vs_base": delta_cost,
                "delta_auc_vs_base": delta_auc,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by feature set type and CV regime
    df = df.sort_values(["cv_regime", "n_features", "feature_set"], ascending=[True, False, True])

    # Save CSV
    df.to_csv(output_csv, index=False)
    log.info(f"✓ Saved ablation table CSV: {output_csv}")

    # Generate LaTeX if requested
    if output_latex:
        _generate_ablation_latex(df, output_latex, reference_set)

    # Print summary
    _print_ablation_summary(df, reference_set)

    return df


def _generate_ablation_latex(df: pd.DataFrame, output_path: Path, reference_set: str):
    """Generate LaTeX table for ablation results."""

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation Study: Feature Set Comparison}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Feature Set & $n$ & AUC & PR-AUC & Recall\% & Cost\% & $\Delta$Recall & $\Delta$Cost \\",
        r"\midrule",
    ]

    for cv_regime in df["cv_regime"].unique():
        cv_label = "Cross-protein" if cv_regime == "protein_logo" else "Cross-pocket"
        latex_lines.append(r"\multicolumn{8}{l}{\textbf{" + cv_label + r"}} \\")

        df_cv = df[df["cv_regime"] == cv_regime]

        for _, row in df_cv.iterrows():
            fs_name = row["feature_set"]
            is_ref = fs_name == reference_set

            # Format values
            auc_str = f"{row['auc']:.3f}"
            pr_str = f"{row['pr_auc']:.3f}"
            recall_str = f"{row['recall']*100:.1f}"
            cost_str = f"{row['cost_saved']*100:.1f}"

            if is_ref:
                delta_r = "---"
                delta_c = "---"
            else:
                delta_r = f"{row['delta_recall_vs_base']*100:+.1f}"
                delta_c = f"{row['delta_cost_vs_base']*100:+.1f}"

            # Bold reference row
            if is_ref:
                fs_name = r"\textbf{" + fs_name + r"}"

            latex_lines.append(
                f"  {fs_name} & {row['n_features']} & {auc_str} & {pr_str} & "
                f"{recall_str} & {cost_str} & {delta_r} & {delta_c} \\\\"
            )

        latex_lines.append(r"\midrule")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    log.info(f"✓ Saved ablation table LaTeX: {output_path}")


def _print_ablation_summary(df: pd.DataFrame, reference_set: str):
    """Print ablation summary to console."""

    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)

    for cv_regime in df["cv_regime"].unique():
        cv_label = "Cross-protein CV" if cv_regime == "protein_logo" else "Cross-pocket CV"
        print(f"\n{cv_label}")
        print("-"*90)
        print(f"{'Feature Set':<25} {'n':>3} {'AUC':>8} {'PR-AUC':>8} {'Recall%':>8} "
              f"{'Cost%':>8} {'ΔRecall':>8} {'ΔCost':>8}")
        print("-"*90)

        df_cv = df[df["cv_regime"] == cv_regime]

        for _, row in df_cv.iterrows():
            fs_name = row["feature_set"]
            is_ref = fs_name == reference_set

            delta_r = "---" if is_ref else f"{row['delta_recall_vs_base']*100:+.1f}%"
            delta_c = "---" if is_ref else f"{row['delta_cost_vs_base']*100:+.1f}%"

            marker = " *" if is_ref else ""

            print(f"{fs_name:<25}{marker:>2} {row['n_features']:>3} "
                  f"{row['auc']:>8.3f} {row['pr_auc']:>8.3f} "
                  f"{row['recall']*100:>7.1f}% {row['cost_saved']*100:>7.1f}% "
                  f"{delta_r:>8} {delta_c:>8}")

    print("="*100)
    print("* = reference (baseline) feature set")
    print()


def plot_ablation_bars(
        ablation_results,
        output_path,
        frozen_model="Logistic Regression",
        fpr_target=0.20,
):
    import numpy as np
    import matplotlib.pyplot as plt

    results = ablation_results["results"]
    timescales = ablation_results["config"]["timescales"]

    feature_sets = ["F5_struct_energy", "F4_struct", "F1_energy"]
    colors = {
        "F5_struct_energy": "#1f77b4",
        "F4_struct": "#ff7f0e",
        "F1_energy": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, cv_regime in enumerate(["protein_logo", "pocket_gkf"]):
        ax = axes[ax_idx]

        x = np.arange(len(timescales))
        width = 0.25

        for i, fs_name in enumerate(feature_sets):
            recalls = []

            if fs_name not in results:
                continue
            if cv_regime not in results[fs_name]:
                continue

            by_time = results[fs_name][cv_regime]

            for t in timescales:
                if t not in by_time:
                    recalls.append(np.nan)
                    continue

                r = by_time[t].get(frozen_model)
                if not r:
                    recalls.append(np.nan)
                    continue

                recall, _, _ = _metrics_from_fold_ops(
                    r["fold_ops"], fpr_target
                )
                recalls.append(recall * 100)

            ax.bar(
                x + i * width,
                recalls,
                width=width,
                label=fs_name,
                color=colors.get(fs_name, "gray"),
                edgecolor="black",
                linewidth=0.6,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(timescales)
        ax.set_xlabel("Simulation Time (ns)")
        ax.set_title(
            "Cross-protein CV" if cv_regime == "protein_logo"
            else "Cross-pocket CV"
        )
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(f"Recall (%) at FPR ≤ {int(fpr_target*100)}%")
    axes[0].set_ylim(0, 100)

    axes[0].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_ablation_report(
        ablation_results: dict,
        output_dir: Path,
        date_tag: str,
        frozen_model: str = "Logistic Regression",
):
    """
    Generate comprehensive ablation report with tables and figures.

    Args:
        ablation_results: Results from run_ablation_experiment
        output_dir: Output directory
        date_tag: Date tag for output files
        frozen_model: Model to use for reporting
    """
    log.info("\n" + "="*80)
    log.info("GENERATING ABLATION REPORT")
    log.info("="*80)

    # Generate tables for different FPR targets and timescales
    for fpr_target in [0.10, 0.20]:
        for timescale in ablation_results["config"]["timescales"]:
            csv_path = output_dir / f"ablation_table_t{timescale}ns_fpr{int(fpr_target*100)}_{date_tag}.csv"
            latex_path = output_dir / f"ablation_table_t{timescale}ns_fpr{int(fpr_target*100)}_{date_tag}.tex"

            generate_ablation_table(
                ablation_results=ablation_results,
                output_csv=csv_path,
                output_latex=latex_path,
                reference_set="F5_struct_energy",
                frozen_model=frozen_model,
                fpr_target=fpr_target,
                timescale=timescale,
            )

    # Generate curves
    for fpr_target in [0.10, 0.20]:
        curves_path = output_dir / f"ablation_curves_fpr{int(fpr_target*100)}_{date_tag}.png"
        plot_ablation_bars(
            ablation_results=ablation_results,
            output_path=curves_path,
            frozen_model=frozen_model,
            fpr_target=fpr_target,
        )

    log.info("\n✓ Ablation report complete")

def plot_risk_sweep_alpha_curve(
        results_dict_by_strategy: dict,
        timescale_ns: float = 2.0,
        frozen_model_by_strategy: dict = None,
        alphas: list = None,
        long_ns: float = 20.0,
        out_png: str = "./outputs/figure_risk_sweep_alpha.png",
):
    """
    Plot α-sweep (FPR budget sweep) at a FIXED timescale.
    Uses fold_ops (train-calibrated thresholds per fold) to avoid leakage.

    Produces two curves per strategy:
      - Recall vs α
      - Cost saved (%) vs α

    Args:
        results_dict_by_strategy: dict like {"protein_logo": results_by_time, "pocket_gkf": results_by_time}
        timescale_ns: which t to evaluate (e.g., 2.0)
        frozen_model_by_strategy: dict like {"protein_logo": "Logistic Regression", "pocket_gkf": "Logistic Regression"}
        alphas: list of α values (FPR targets). If None, will infer from fold_ops present at (t, model).
        long_ns: baseline simulation length for cost_saved calculation
        out_png: output file path
    """
    import numpy as np
    import matplotlib.pyplot as plt

    labels = {'protein_logo': 'Cross-protein CV', 'pocket_gkf': 'Cross-pocket CV'}
    colors = {
        'protein_logo': '#4C72B0',   # muted blue
        'pocket_gkf': '#DD8452'      # muted orange
    }


    # Collect available α values from fold_ops if user didn't supply them
    if alphas is None:
        alpha_set = set()
        for strategy_key, by_time in results_dict_by_strategy.items():
            if timescale_ns not in by_time:
                continue
            model_name = None
            if frozen_model_by_strategy and strategy_key in frozen_model_by_strategy:
                model_name = frozen_model_by_strategy[strategy_key]
            else:
                # Fallback: pick best model by AUC at this timescale (exploration only)
                model_name = select_best_model(by_time[timescale_ns], metric="auc")

            r = by_time[timescale_ns].get(model_name, {})
            fold_ops = r.get("fold_ops", [])
            for op in fold_ops:
                if "fpr_target" in op:
                    alpha_set.add(float(op["fpr_target"]))
        alphas = sorted(alpha_set)

    if not alphas:
        raise ValueError("No α values found. Did you run CV with a non-empty fpr_targets list?")

    plt.figure(figsize=(6.8, 4.8))
    ax = plt.gca()

    for strategy_key, by_time in results_dict_by_strategy.items():
        if timescale_ns not in by_time:
            continue

        if frozen_model_by_strategy and strategy_key in frozen_model_by_strategy:
            model_name = frozen_model_by_strategy[strategy_key]
        else:
            model_name = select_best_model(by_time[timescale_ns], metric="auc")

        r = by_time[timescale_ns].get(model_name)
        if not r or "fold_ops" not in r:
            continue

        fold_ops = r["fold_ops"]

        recs, costs = [], []
        for a in alphas:
            recall, triage_rate, achieved_fpr = _metrics_from_fold_ops(fold_ops, a)
            recs.append(recall * 100 if np.isfinite(recall) else np.nan)

            cs = _cost_saved(triage_rate, t_ns=float(timescale_ns), long_ns=float(long_ns))
            costs.append(cs * 100 if np.isfinite(cs) else np.nan)

        c = colors.get(strategy_key, None)
        lab = labels.get(strategy_key, strategy_key)

        ax.plot(alphas, recs, marker="o", linewidth=2.0, color=c,
                label=f"{lab} — Recall")
        ax.plot(alphas, costs, marker="s", linestyle="--", linewidth=1.8, color=c, alpha=0.9,
                label=f"{lab} — Cost saved")

    ax.set_xlabel(r"Risk-budget $\alpha$ (FPR target)")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 80)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"✓ Saved α-sweep figure: {out_png}")

def main():
    date_tag = "20251225"
    date_tag = "20260214"
    version = "2.all"
    parser = argparse.ArgumentParser(
        description="Generate comprehensive ML analysis figure with different CV strategies"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(f"outputs/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"),
        help="Path to fingerprint_summary_with_components.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"./outputs/comprehensive_ml_both_cv_{date_tag}.png"),
        help="Output figure path"
    )
    parser.add_argument(
        "--timescales",
        type=float,
        nargs="+",
        default=TIMESCALES,
        help="Timescales to evaluate (default: )"
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default='both',
        choices=['protein', 'pocket', 'ligand_pocket', 'both'],
        help="Grouping strategy: 'protein' (protein-level), 'pocket' (pocket-level), 'ligand_pocket' (protein|ligand|pocket), or 'both'"
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default='logo',
        choices=['logo', 'gkf'],
        help="CV mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold)"
    )
    parser.add_argument(
        "--load_results",
        type=Path,
        default=Path(f'./outputs/results_both_cv_{date_tag}_{version}.pkl') ,
        help="Path to existing results .pkl file (skip training if provided)"
    )
    parser.add_argument(
        "--plot_only",
        default= False,
        help="Only generate plots from existing results (no training)"
    )

    parser.add_argument(
        "--print_fold_summary",
        default=False,
        help="Print per-fold test-set sizes and class counts at a selected timescale (useful for LOPO small-n interpretation)."
    )
    parser.add_argument(
        "--fold_summary_timescale",
        type=float,
        default=2.0,
        help="Timescale (ns) at which to print per-fold CV test-set sizes when --print_fold_summary is set (default: 2)."
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="F4_struct",
        choices=["F4_struct", "F5_struct_energy"],
        help="Feature set to use: 'F4_struct' (4 features: slope, rmsd_var, mean_disp, var_disp) or 'F5_struct_energy' (adds energy_std)"
    )

    # Ablation experiment arguments
    parser.add_argument(
        "--run_ablation",
        default= False ,
        help="Run ablation experiment comparing different feature sets"
    )
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="full",
        choices=["minimal", "lofo", "single", "full"],
        help="Type of ablation: 'minimal' (BASE vs STRUCT4 vs ENERGY1), 'lofo' (leave-one-feature-out), 'single' (single features), 'full' (all variants)"
    )
    parser.add_argument(
        "--ablation_timescales",
        type=float,
        nargs="+",
        default=[2, 3, 5],
        help="Timescales for ablation experiment (default: 2 3 5)"
    )
    parser.add_argument(
        "--load_ablation",
        type=Path,
        default=None,
        help="Path to existing ablation results .pkl file (skip training, just generate reports)"
    )

    args = parser.parse_args()

    # Get feature columns based on feature_set argument
    feature_cols = get_feature_cols(args.feature_set)
    log.info(f"Using feature set: {args.feature_set} -> {feature_cols}")

    # =========================================================================
    # ABLATION EXPERIMENT MODE
    # =========================================================================
    if args.run_ablation or args.load_ablation:
        log.info("="*80)
        log.info("ABLATION EXPERIMENT MODE")
        log.info("="*80)

        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)

        if args.load_ablation and args.load_ablation.exists():
            # Load existing ablation results
            log.info(f"Loading ablation results from: {args.load_ablation}")
            with open(args.load_ablation, "rb") as f:
                ablation_results = pickle.load(f)
        else:
            # Run ablation experiment
            ablation_results = run_ablation_experiment(
                features_path=args.features,
                ablation_type=args.ablation_type,
                timescales=args.ablation_timescales,
                cv_regimes=["protein_logo", "pocket_gkf"],
                models_to_run=["Random Forest", "Logistic Regression"],
                fpr_targets=(0.10, 0.20),
                output_dir=output_dir,
                date_tag=date_tag,
            )

        # Generate ablation report
        generate_ablation_report(
            ablation_results=ablation_results,
            output_dir=output_dir,
            date_tag=date_tag,
            frozen_model="Logistic Regression",
        )

        log.info("\n" + "="*80)
        log.info("ABLATION EXPERIMENT COMPLETE")
        log.info("="*80)
        return  # Exit after ablation

    # =========================================================================
    # STANDARD ANALYSIS MODE
    # =========================================================================
    log.info("="*80)
    log.info("COMPREHENSIVE ML FIGURE GENERATION WITH MULTIPLE CV STRATEGIES")
    log.info("="*80)

    results_dict = {}
    if args.plot_only:
        if args.load_results is None:
            raise ValueError("--plot_only requires --load_results")

        results_dict = load_results_from_pkl(args.load_results)
        generate_comparison_figure(results_dict, args.output, args.features, feature_cols=feature_cols)

    elif args.group_by == 'both':
        # Run both strategies for comparison
        log.info("\n" + "="*80)
        log.info("EVALUATION 1: Leave-one-protein-out (protein generalization)")
        log.info("="*80)

        results_protein = evaluate_all_models(args.features, args.timescales,
                                              group_by='protein', cv_mode='logo',
                                              print_fold_summary=args.print_fold_summary,
                                              fold_summary_timescale=args.fold_summary_timescale,
                                              fpr_targets = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                                              feature_cols=feature_cols)
        results_dict['protein_logo'] = results_protein

        log.info("\n" + "="*80)
        log.info("EVALUATION 2: Pocket-level GroupKFold (pocket generalization)")
        log.info("="*80)

        results_pocket = evaluate_all_models(args.features, args.timescales,
                                             group_by='ligand_pocket', cv_mode='gkf',
                                             print_fold_summary=args.print_fold_summary,
                                             fold_summary_timescale=args.fold_summary_timescale,
                                             fpr_targets = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                                             feature_cols=feature_cols)
        results_dict['pocket_gkf'] = results_pocket
        # Generate comparison figure
        generate_comparison_figure(results_dict, args.output, args.features, feature_cols=feature_cols)
        results_dict = {
            "protein_logo": results_protein,
            "pocket_gkf": results_pocket,
        }

        with open(f"./outputs/results_both_cv_{date_tag}_{version}.pkl", "wb") as f:
            pickle.dump(results_dict, f)

        log.info("✓ Saved combined results: results_both_cv.pkl")
        # After results_dict is populated:
        report_path = f"./outputs/screening_report_both_cv_{date_tag}_{version}.csv" if args.group_by == "both" \
            else f"./outputs/screening_report_{args.group_by}_{args.cv_mode}_{date_tag}_{version}.csv"

        export_screening_report_csv(
            results_dict=results_dict,
            output_csv=report_path,
            long_ns=20.0,
            fpr_targets=(0.10, 0.20),
        )
        log.info(f"✓ Saved screening report CSV: {report_path}")

    compute_dataset_stats(features_path=args.features)

    dataset_stats = compute_dataset_stats(features_path=args.features, timescales=args.timescales)
    dataset_stats.to_csv(
        f"./outputs/dataset_size_by_timescale_{date_tag}_{version}.csv",
        index=False
    )
    # Use frozen_model for paper-quality figures
    plot_timescale_horizon(
        results_dict,
        output=f"./outputs/figure1_timescale_horizon_{date_tag}_{version}.png",
        frozen_model="Logistic Regression"
    )


    # Use frozen_model for paper-quality figures
    plot_triage_efficiency(
        results_dict,
        output=f"./outputs/figure2_triage_efficiency_{date_tag}_{version}.png",
        frozen_model="Logistic Regression"
    )

    plot_early_time_separation_compact(
        features_path=args.features,
        timescale=2,
        output= f"./outputs/early_time_separation_2ns_{date_tag}_{version}.png",
        feature_cols=feature_cols
    )

    # strategy_key examples in your CSV: "protein_logo" and "pocket_gkf"
    # Check for cached FPR results
    fpr_results_path = Path(f"./outputs/results_fpr_strategies_{date_tag}_{version}.pkl")

    if fpr_results_path.exists():
        log.info(f"Loading cached FPR results from {fpr_results_path}")
        with open(fpr_results_path, "rb") as f:
            results_by_strategy_fpr = pickle.load(f)
    else:
        log.info("Computing FPR results (this may take a while)...")

        # FIX C: Run evaluate_all_models ONCE per strategy with both FPR targets
        # Each call already computes fold_ops for all fpr_targets=[0.10, 0.20]
        # We use GOAL_FPR=0.10 for threshold selection during CV, but fold_ops
        # contains metrics for BOTH FPR targets

        results_protein = evaluate_all_models(
            args.features, args.timescales,
            group_by="protein", cv_mode="logo",
            GOAL_FPR=0.10,  # Default threshold for y_pred
            fpr_targets = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),  # Both targets computed in fold_ops
            print_fold_summary=args.print_fold_summary,
            fold_summary_timescale=args.fold_summary_timescale,

            feature_cols=feature_cols,
        )

        results_pocket = evaluate_all_models(
            args.features, args.timescales,
            group_by="ligand_pocket", cv_mode="gkf",
            GOAL_FPR=0.10,
            fpr_targets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
            print_fold_summary=args.print_fold_summary,
            fold_summary_timescale=args.fold_summary_timescale,
            feature_cols=feature_cols,
        )

        # Structure results by FPR target for backward compatibility with plotting functions
        # Note: The y_pred in results is based on GOAL_FPR, but fold_ops has both
        results_by_strategy_fpr = {
            0.10: {
                "protein_logo": results_protein,
                "pocket_gkf": results_pocket,
            },
            0.20: {
                "protein_logo": results_protein,
                "pocket_gkf": results_pocket,
            },
        }

        # Save FPR results for future use
        with open(fpr_results_path, "wb") as f:
            pickle.dump(results_by_strategy_fpr, f)
        log.info(f"✓ Saved FPR results to {fpr_results_path}")

    # Freeze one model per strategy (example choices)
    frozen_model_by_strategy = {
        "protein_logo": "Logistic Regression",
        "pocket_gkf": "Logistic Regression",
    }

    plot_triage_and_cost_frozen_model(
        results_by_strategy_fpr,
        frozen_model_by_strategy,
        TIMESCALES=[1,2,3,4,5,6,7,8,10,12,14],
        long_ns=20.0,
        out_recall=f"./outputs/triage_out_recall_{date_tag}_{version}.png",
        out_cost=f"./outputs/triage_cost_frozen_{date_tag}_{version}.png"

    )

    # ============================================================================
    # Generate Table III: Operating Point Recommendations
    # ============================================================================
    log.info("\n" + "="*80)
    log.info("GENERATING TABLE III: OPERATING POINT RECOMMENDATIONS")
    log.info("="*80)

    # Print operating point summary first
    # print_operating_point_summary(results_by_strategy_fpr, frozen_model_by_strategy)
    #
    # Generate the full Table III with CSV and LaTeX outputs
    # Fix D: Only the 3 recommended operating points for the paper
    table_iii_df = generate_table_iii(
        results_by_strategy_fpr,
        frozen_model_by_strategy,
        operating_points=[(2, 0.20), (3, 0.20), (3, 0.10),(4, 0.20), (4, 0.10),(5, 0.20), (5, 0.10), (6, 0.20), (6, 0.10)],  # Paper's recommended modes
        output_csv=f"./outputs/table_iii_operating_points_{date_tag}_{version}.csv",
        output_latex=f"./outputs/table_iii_operating_points_{date_tag}_{version}.tex",
        debug=True
    )

    log.info("\n" + "="*80)
    log.info("COMPLETE!")
    log.info("="*80)

    plot_risk_sweep_alpha_curve(
        results_dict_by_strategy=results_dict,
        timescale_ns=2.0,
        frozen_model_by_strategy={"protein_logo": "Logistic Regression",
                                  "pocket_gkf": "Logistic Regression"},
        long_ns=20.0,
        out_png="./outputs/figure_alpha_sweep_2ns.png",
    )


if __name__ == "__main__":
    main()

# Example usage:
# python generate_comprehensive_figure.py \
#     --features fingerprint_summary_with_components.csv \
#     --output comprehensive_ml_results.png