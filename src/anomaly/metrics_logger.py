"""
metrics_logger.py
-----------------
Centralised evaluation-metrics logger and visual reporting system for the anomaly-detection pipeline.

Generates evaluation artifacts for both Static and Cyclic equipment:
  evaluation_results/confusion_matrix_binary_<name>.png     - Visual Healthy vs Anomaly heatmap
  evaluation_results/confusion_matrix_per_fault_<name>.png  - Visual Multi-class fault heatmap
  evaluation_results/evaluation_results_<name>.html         - Visual HTML dashboard with heatmaps
  evaluation_results/evaluation_results_<name>.md           - Markdown evaluation summary
  evaluation_results/evaluation_results.html                - Unified visual dashboard (both models)
  logs/training_metrics_<YYYYMMDD_HHMMSS>.log               - Timestamped log file
  reports/training_metrics_<name>_<ts>.json                 - Machine-readable full dump

Metrics computed:
  Split Statistics        : Train rows, Calib rows, Test rows, Healthy rows | Unhealthy rows
  Threshold / Calibration : Threshold value, Calib mean, Calib std, Calib min/max, Implied percentile
  Hold-out Test Metrics   : Accuracy, Precision, Recall, F1 Score, False Alarm Rate, ROC-AUC, PR-AUC
  Per Anomaly Fault Type  : Accuracy, Precision, Recall, F1 Score, False Alarm Rate, ROC-AUC, PR-AUC, Support
"""

import base64
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import abs_path, load_config

# module-level logger
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(val, decimals=4):
    """Round a float/nan safely for JSON serialisation and reporting."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def _safe_str(val, decimals=4, default="n/a"):
    """Format float/None safely for human-readable display without LaTeX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return f"{float(val):.{decimals}f}"


def _div(a, b):
    """Zero-safe division."""
    return float(a) / float(b) if b else 0.0


# ---------------------------------------------------------------------------
# Public API  -  called from model.py, pipeline.py, and standalone evaluation
# ---------------------------------------------------------------------------

def log_training_metrics(name, art, df):
    """
    Compute, log, and render all evaluation metrics for equipment type *name*.

    Parameters
    ----------
    name : str
        Equipment-type key, e.g. "static_rotating" or "cyclic_sts".
    art  : dict
        Artefact dict returned by model.train() or model.load().
    df   : pd.DataFrame
        The labelled dataframe used for training/evaluation (contains _healthy,
        _labelled, _fault columns).

    Returns
    -------
    dict  Full metrics payload.
    """
    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = ts.strftime("%Y%m%d_%H%M%S")

    log.info("=" * 70)
    log.info("[METRICS] %s  |  evaluation run at %s", name.upper(), ts_str)
    log.info("=" * 70)

    metrics = {
        "equipment_type": name,
        "run_timestamp": ts_str,
        "trained_at": art.get("trained_at"),
        "method": art.get("method"),
    }

    # 1. Split statistics
    split_keys = art.get("split_keys")
    counts = art.get("counts", {})

    log.info("[METRICS] -- Split Statistics --")
    train_count = counts.get("train", 0)
    # If val exists and was carved from train, report the full training partition
    if counts.get("val", 0) > 0 and (train_count + counts.get("val", 0) + counts.get("calib", 0) + counts.get("test", 0) == len(df)):
        train_count += counts.get("val", 0)
    log.info("[METRICS]   Train rows   : %d", train_count)
    log.info("[METRICS]   Calib rows   : %d", counts.get("calib", 0))
    log.info("[METRICS]   Test  rows   : %d", counts.get("test", 0))

    healthy_total = int(df["_healthy"].sum()) if "_healthy" in df.columns else 0
    unhealthy_total = int((~df["_healthy"]).sum()) if "_healthy" in df.columns else 0
    log.info("[METRICS]   Healthy rows : %d  |  Unhealthy rows : %d",
             healthy_total, unhealthy_total)

    metrics["split"] = {
        "train": train_count,
        "calib": counts.get("calib", 0),
        "test": counts.get("test", 0),
        "total_healthy": healthy_total,
        "total_unhealthy": unhealthy_total,
    }

    # 2. Threshold / calibration stats
    thr = art.get("threshold")
    calib_scores = art.get("calib_scores", np.array([]))
    thr_pct = None
    if len(calib_scores):
        thr_pct = _safe(float(np.mean(calib_scores <= thr)) * 100, 1)

    log.info("[METRICS] -- Threshold / Calibration --")
    log.info("[METRICS]   Threshold value        : %.6f",
             thr if thr is not None else float("nan"))
    log.info("[METRICS]   Calib score  mean      : %.6f",
             np.mean(calib_scores) if len(calib_scores) else float("nan"))
    log.info("[METRICS]   Calib score  std        : %.6f",
             np.std(calib_scores) if len(calib_scores) else float("nan"))
    log.info("[METRICS]   Calib score  min / max  : %.6f / %.6f",
             np.min(calib_scores) if len(calib_scores) else float("nan"),
             np.max(calib_scores) if len(calib_scores) else float("nan"))
    log.info("[METRICS]   Implied percentile      : %s%%", thr_pct)

    metrics["threshold"] = {
        "value": _safe(thr, 6),
        "implied_percentile_pct": thr_pct,
        "calib_score_mean": _safe(np.mean(calib_scores) if len(calib_scores) else None),
        "calib_score_std": _safe(np.std(calib_scores) if len(calib_scores) else None),
        "calib_score_min": _safe(np.min(calib_scores) if len(calib_scores) else None),
        "calib_score_max": _safe(np.max(calib_scores) if len(calib_scores) else None),
        "n_calib_scores": int(len(calib_scores)),
    }

    # 3. Hold-out test data evaluation metrics
    perf_metrics = _compute_performance(name, art, df, split_keys)
    metrics["performance"] = {k: v for k, v in perf_metrics.items() if not k.startswith("_")}

    # 4. Per anomaly fault type metrics (hold-out test set)
    per_class = _compute_per_class(name, art, df, split_keys)
    metrics["per_class"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_class]

    # 5. Visual confusion matrix in evaluation_results/
    visual_files = _generate_visual_confusion_matrix(
        name=name,
        art=art,
        perf_metrics=perf_metrics,
    )
    metrics["visual_artifacts"] = visual_files

    # Persist log and JSON report files
    _write_log_file(name, ts_file, metrics)
    _write_json_file(name, ts_file, metrics)

    log.info("[METRICS] -- Done  (%s) --", name)
    log.info("=" * 70)

    return metrics


def log_evaluation_metrics(name, summary_df, by_fault_df):
    """Post-detection evaluation logging removed as requested."""
    pass


# ---------------------------------------------------------------------------
# Private computation helpers
# ---------------------------------------------------------------------------

def _compute_performance(name, art, df, split_keys):
    """Binary precision / recall / F1 / ROC-AUC / PR-AUC / Accuracy / False Alarm Rate on hold-out test split."""
    result = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "false_alarm_rate": None,
        "false_alarm_rate_holdout": None,
        "roc_auc": None,
        "pr_auc": None,
        "note": "labels unavailable",
    }

    if "_labelled" not in df.columns or not df["_labelled"].any():
        log.info("[METRICS] -- Performance  : no labels - skipped")
        return result

    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        iso = art.get("iso")
        thr = art.get("threshold")

        if split_keys is None:
            raise ValueError(
                f"[{name}] split_keys is None. Evaluation must be performed strictly on hold-out test data; fallback to all rows is prohibited."
            )
        o = df.merge(split_keys, on=["object_id", "timestamp"], how="left")
        if "split" not in o.columns or not (o["split"] == "test").any():
            raise ValueError(
                f"[{name}] No hold-out test split found in split_keys. "
                "Evaluation must be performed strictly on hold-out test data; fallback to all rows is prohibited."
            )

        test_mask = (o["split"] == "test").values
        truth = (~o["_healthy"]).values.astype(int)

        scores_all = None
        method = art.get("method", "")
        params = list(art["params"])
        et_cfg = art.get("et", {
            "parameters": {p: {} for p in params},
            "mode": "timeseries",
            "rolling_window": 1,
            "rolling_windows": [1],
        })

        if iso is not None and method == "xgboost":
            from . import features
            from .model import anomaly_score as _anomaly_score

            try:
                X_feat, C_feat, _ = features.build(df, et_cfg)
                if "healthy_mu" in art and "healthy_cov_inv" in art and art["healthy_mu"] is not None:
                    X_feat = features.add_mahalanobis(
                        X_feat, art["healthy_mu"], art["healthy_cov_inv"], params
                    )
                Z_feat = art["baseline"].residuals(X_feat, C_feat)
                scores_all, _ = _anomaly_score(
                    art["method"], iso, Z_feat, params, X=X_feat
                )
            except Exception as re_err:
                log.warning(
                    "[METRICS][%s] XGBoost re-scoring failed (%s) - using flag proxy",
                    name, re_err,
                )

        elif method == "max_z":
            from . import features
            try:
                X_feat, C_feat, _ = features.build(df, et_cfg)
                Z_feat = art["baseline"].residuals(X_feat, C_feat)
                scores_all = Z_feat[params].abs().max(axis=1).values
            except Exception as re_err:
                log.warning(
                    "[METRICS][%s] max_z re-scoring failed (%s) - cannot compute performance",
                    name, re_err,
                )

        if scores_all is not None and len(scores_all) == len(o):
            pred_prob = scores_all
        elif "flag" in o.columns:
            pred_prob = o["flag"].astype(float).values
        else:
            log.info(
                "[METRICS] -- Performance  : cannot compute (no scores or flag column)"
            )
            return result

        pred_binary = (
            (pred_prob >= thr).astype(int)
            if thr is not None
            else (pred_prob > 0.5).astype(int)
        )

        eval_mask = test_mask
        eval_note = "hold-out test set"

        truth_test = truth[eval_mask]
        pred_bin_test = pred_binary[eval_mask]
        pred_prob_test = pred_prob[eval_mask]

        acc = _safe(accuracy_score(truth_test, pred_bin_test))
        prec = _safe(precision_score(truth_test, pred_bin_test, zero_division=0))
        rec = _safe(recall_score(truth_test, pred_bin_test, zero_division=0))
        f1 = _safe(f1_score(truth_test, pred_bin_test, zero_division=0))
        try:
            auc = _safe(roc_auc_score(truth_test, pred_prob_test))
        except Exception:
            auc = None
        try:
            ap = _safe(average_precision_score(truth_test, pred_prob_test))
        except Exception:
            ap = None

        test_healthy = eval_mask & (truth == 0)
        far_ho = (
            _safe(pred_binary[test_healthy].mean())
            if test_healthy.any()
            else 0.0
        )

        result.update({
            "test_rows": int(eval_mask.sum()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_alarm_rate": far_ho,
            "false_alarm_rate_holdout": far_ho,
            "roc_auc": auc,
            "pr_auc": ap,
            "note": f"computed on {eval_note}",
            "_arrays": {
                "truth_test": truth_test,
                "pred_bin_test": pred_bin_test,
                "pred_prob_test": pred_prob_test,
            },
        })

        log.info("[METRICS] -- Hold-out Test Data Evaluation Metrics (N=%d) --", int(eval_mask.sum()))
        log.info("[METRICS]   Accuracy               : %s", acc)
        log.info("[METRICS]   Precision              : %s", prec)
        log.info("[METRICS]   Recall                 : %s", rec)
        log.info("[METRICS]   F1 Score               : %s", f1)
        log.info("[METRICS]   False Alarm Rate       : %s", far_ho)
        log.info("[METRICS]   ROC-AUC                : %s", auc)
        log.info("[METRICS]   PR-AUC                 : %s", ap)

    except ValueError:
        raise
    except Exception as exc:
        log.warning(
            "[METRICS][%s] Performance computation failed: %s", name, exc
        )
        log.debug(traceback.format_exc())

    return result


def _compute_per_class(name, art, df, split_keys):
    """
    Accuracy, Precision, Recall, F1, False Alarm Rate, ROC-AUC, PR-AUC, and Support
    for each anomaly fault type on the hold-out test set.
    """
    if art.get("method") != "xgboost" or art.get("iso") is None:
        return []
    if "_fault" not in df.columns or "_labelled" not in df.columns:
        return []
    if not df["_labelled"].any():
        return []

    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        iso = art["iso"]
        le = iso._label_encoder
        params = list(art["params"])

        from . import features
        from .model import anomaly_score as _anomaly_score

        et_cfg = art.get("et", {
            "parameters": {p: {} for p in params},
            "mode": "timeseries",
            "rolling_window": 1,
            "rolling_windows": [1],
        })
        X_feat, C_feat, _ = features.build(df, et_cfg)
        if "healthy_mu" in art and "healthy_cov_inv" in art and art["healthy_mu"] is not None:
            X_feat = features.add_mahalanobis(
                X_feat, art["healthy_mu"], art["healthy_cov_inv"], params
            )
        Z_feat = art["baseline"].residuals(X_feat, C_feat)
        score_res = _anomaly_score("xgboost", iso, Z_feat, params, X=X_feat, return_proba=True)
        anomaly_prob, pred_classes, proba, class_names = score_res

        if split_keys is None:
            raise ValueError(
                f"[{name}] split_keys is None. Per-class breakdown must be performed strictly on hold-out test data; fallback to all rows is prohibited."
            )
        o_split = df.merge(split_keys, on=["object_id", "timestamp"], how="left")
        if "split" not in o_split.columns or not (o_split["split"] == "test").any():
            raise ValueError(
                f"[{name}] No hold-out test split found in split_keys. "
                "Per-class breakdown must be performed strictly on hold-out test data; fallback to all rows is prohibited."
            )

        eval_mask = (o_split["split"] == "test").values
        y_true_str = df["_fault"].values[eval_mask]
        y_pred_str = pred_classes[eval_mask]
        proba_test = proba[eval_mask] if proba is not None else None

        # Build ordered list of classes: Normal first, then sorted anomaly faults
        all_classes = sorted(list(set(y_true_str) | set(y_pred_str)))
        ordered_classes = ["Normal"] if "Normal" in all_classes else []
        ordered_classes += [c for c in all_classes if c != "Normal"]

        per_class_out = []
        log.info("[METRICS] -- Per Anomaly Fault Type Metrics (Hold-out Test Set, N=%d) --", int(eval_mask.sum()))
        header = (
            f"  {'Fault Type':<26}  {'Accuracy':>9}  {'Precision':>9}  "
            f"{'Recall':>9}  {'F1 Score':>9}  {'False Alarm':>12}  "
            f"{'ROC-AUC':>9}  {'PR-AUC':>9}  {'Support':>8}"
        )
        log.info("[METRICS] %s", header)
        log.info("[METRICS]   %s", "-" * (len(header) - 2))

        acc_list, p_list, r_list, f1_list, far_list, auc_list, pr_list = [], [], [], [], [], [], []

        for label in ordered_classes:
            y_true_c = (y_true_str == label).astype(int)
            y_pred_c = (y_pred_str == label).astype(int)
            supp = int(y_true_c.sum())

            acc_c = _safe(accuracy_score(y_true_c, y_pred_c))
            p_c = _safe(precision_score(y_true_c, y_pred_c, zero_division=0))
            r_c = _safe(recall_score(y_true_c, y_pred_c, zero_division=0))
            f1_c = _safe(f1_score(y_true_c, y_pred_c, zero_division=0))

            # False Alarm Rate (False Positive Rate) for this class = FP / (FP + TN)
            fp_c = int(((y_pred_c == 1) & (y_true_c == 0)).sum())
            tn_c = int(((y_pred_c == 0) & (y_true_c == 0)).sum())
            far_c = _safe(fp_c / (fp_c + tn_c)) if (fp_c + tn_c) > 0 else 0.0

            # ROC-AUC & PR-AUC for this class
            auc_c = None
            pr_c = None
            if proba_test is not None and label in class_names and len(np.unique(y_true_c)) > 1:
                col_idx = class_names.index(label)
                p_vec = proba_test[:, col_idx]
                try:
                    auc_c = _safe(roc_auc_score(y_true_c, p_vec))
                except Exception:
                    auc_c = None
                try:
                    pr_c = _safe(average_precision_score(y_true_c, p_vec))
                except Exception:
                    pr_c = None

            row_data = {
                "class": label,
                "accuracy": acc_c,
                "precision": p_c,
                "recall": r_c,
                "f1": f1_c,
                "false_alarm_rate": far_c,
                "roc_auc": auc_c,
                "pr_auc": pr_c,
                "support": supp,
            }
            per_class_out.append(row_data)

            if acc_c is not None:
                acc_list.append(acc_c)
            if p_c is not None:
                p_list.append(p_c)
            if r_c is not None:
                r_list.append(r_c)
            if f1_c is not None:
                f1_list.append(f1_c)
            if far_c is not None:
                far_list.append(far_c)
            if auc_c is not None:
                auc_list.append(auc_c)
            if pr_c is not None:
                pr_list.append(pr_c)

            auc_disp = f"{auc_c:.4f}" if auc_c is not None else "n/a"
            pr_disp = f"{pr_c:.4f}" if pr_c is not None else "n/a"
            log.info(
                "[METRICS]   %-26s  %9.4f  %9.4f  %9.4f  %9.4f  %12.4f  %9s  %9s  %8d",
                label,
                acc_c or 0.0,
                p_c or 0.0,
                r_c or 0.0,
                f1_c or 0.0,
                far_c or 0.0,
                auc_disp,
                pr_disp,
                supp,
            )

        log.info("[METRICS]   %s", "-" * (len(header) - 2))
        macro_acc = _safe(np.mean(acc_list)) if acc_list else 0.0
        macro_p = _safe(np.mean(p_list)) if p_list else 0.0
        macro_r = _safe(np.mean(r_list)) if r_list else 0.0
        macro_f1 = _safe(np.mean(f1_list)) if f1_list else 0.0
        macro_far = _safe(np.mean(far_list)) if far_list else 0.0
        macro_auc = _safe(np.mean(auc_list)) if auc_list else None
        macro_pr = _safe(np.mean(pr_list)) if pr_list else None

        macro_auc_disp = f"{macro_auc:.4f}" if macro_auc is not None else "n/a"
        macro_pr_disp = f"{macro_pr:.4f}" if macro_pr is not None else "n/a"

        log.info(
            "[METRICS]   %-26s  %9.4f  %9.4f  %9.4f  %9.4f  %12.4f  %9s  %9s  %8d",
            "Macro Average",
            macro_acc,
            macro_p,
            macro_r,
            macro_f1,
            macro_far,
            macro_auc_disp,
            macro_pr_disp,
            int(eval_mask.sum()),
        )

        class PerClassList(list):
            def __init__(self, items, metadata=None):
                super().__init__(items)
                self.metadata = metadata or {}

        return PerClassList(per_class_out, metadata={
            "y_true_str": y_true_str,
            "y_pred_str": y_pred_str,
            "classes": ordered_classes,
            "macro": {
                "accuracy": macro_acc,
                "precision": macro_p,
                "recall": macro_r,
                "f1": macro_f1,
                "false_alarm_rate": macro_far,
                "roc_auc": macro_auc,
                "pr_auc": macro_pr,
            }
        })

    except ValueError:
        raise
    except Exception as exc:
        log.warning(
            "[METRICS][%s] Per-class computation failed: %s", name, exc
        )
        log.debug(traceback.format_exc())
        return []


# ---------------------------------------------------------------------------
# Visual Confusion Matrix Generation (Matching User Reference Layout)
# ---------------------------------------------------------------------------

def _generate_visual_confusion_matrix(name, art, perf_metrics):
    """
    Generate a 2x2 confusion matrix PNG matching the user's exact visual format:
      - Clean 3x3 table grid with solid black borders
      - Soft ice blue header backgrounds (#eef6fc)
      - True Positive (TP) & True Negative (TN): soft pastel green (#d9ead3)
      - False Negative (FN) & False Positive (FP): soft pastel red/pink (#fadbd8)
      - High readability, generous breathing room, no text overflow

    Saves directly to evaluation_results/ as:
      - confusion_matrix_static.png / confusion_matrix_cyclic.png
    """
    eval_dir = abs_path("evaluation_results")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if "static" not in name and "cyclic" not in name:
        return {}

    from sklearn.metrics import confusion_matrix
    import matplotlib.patches as patches

    bin_arrays = perf_metrics.get("_arrays", {})
    truth_test = bin_arrays.get("truth_test")
    pred_bin_test = bin_arrays.get("pred_bin_test")

    if truth_test is None or pred_bin_test is None:
        return {}

    cm = confusion_matrix(truth_test, pred_bin_test, labels=[0, 1])
    # Label 0 = Healthy (Negative), Label 1 = Anomaly (Positive)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    acc = float(perf_metrics.get("accuracy") or 0.0)
    prec = float(perf_metrics.get("precision") or 0.0)
    rec = float(perf_metrics.get("recall") or 0.0)
    f1 = float(perf_metrics.get("f1") or 0.0)
    far = float(perf_metrics.get("false_alarm_rate") or 0.0)

    # 12 x 6.8 inches, 300 DPI for crystal clear typography and spacious presentation
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    disp_title = "Static Rotating Equipment - Confusion Matrix" if "static" in name else "Cyclic STS Crane - Confusion Matrix"
    ax.text(6.0, 6.30, disp_title, ha="center", va="center",
            fontsize=18, fontweight="bold", color="#111827")

    # Table layout: 3 columns x 3 rows with balanced spacing
    x0, w0, w1, w2 = 0.5, 3.4, 3.8, 3.8
    x1 = x0 + w0
    x2 = x1 + w1
    x_end = x2 + w2

    y_neg, h_data, h_hdr = 1.30, 1.65, 0.90
    y_pos = y_neg + h_data
    y_hdr = y_pos + h_data
    y_top = y_hdr + h_hdr

    c_blue = "#eef6fc"   # Soft ice blue header background (matching user image)
    c_green = "#d9ead3"  # Soft pastel green for TP & TN (matching user image)
    c_red = "#fadbd8"    # Soft pastel red/pink for FN & FP (matching user image)

    # Draw colored cell backgrounds
    ax.add_patch(patches.Rectangle((x0, y_hdr), w0, h_hdr, facecolor=c_blue, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x1, y_hdr), w1, h_hdr, facecolor=c_blue, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x2, y_hdr), w2, h_hdr, facecolor=c_blue, edgecolor="none"))

    ax.add_patch(patches.Rectangle((x0, y_pos), w0, h_data, facecolor=c_blue, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x1, y_pos), w1, h_data, facecolor=c_green, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x2, y_pos), w2, h_data, facecolor=c_red, edgecolor="none"))

    ax.add_patch(patches.Rectangle((x0, y_neg), w0, h_data, facecolor=c_blue, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x1, y_neg), w1, h_data, facecolor=c_red, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x2, y_neg), w2, h_data, facecolor=c_green, edgecolor="none"))

    # Draw crisp, solid black grid lines
    lw = 2.4
    grid_color = "#000000"
    for y_line in [y_neg, y_pos, y_hdr, y_top]:
        ax.plot([x0, x_end], [y_line, y_line], color=grid_color, linewidth=lw, solid_capstyle="projecting")
    for x_line in [x0, x1, x2, x_end]:
        ax.plot([x_line, x_line], [y_neg, y_top], color=grid_color, linewidth=lw, solid_capstyle="projecting")

    # Headers
    ax.text(x1 + w1 / 2, y_hdr + 0.58, "Predicted Positive", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#000000")
    ax.text(x1 + w1 / 2, y_hdr + 0.26, "(Anomaly)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#4b5563")

    ax.text(x2 + w2 / 2, y_hdr + 0.58, "Predicted Negative", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#000000")
    ax.text(x2 + w2 / 2, y_hdr + 0.26, "(Healthy)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#4b5563")

    ax.text(x0 + w0 / 2, y_pos + h_data * 0.62, "Actual Positive", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#000000")
    ax.text(x0 + w0 / 2, y_pos + h_data * 0.34, "(Anomaly)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#4b5563")

    ax.text(x0 + w0 / 2, y_neg + h_data * 0.62, "Actual Negative", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#000000")
    ax.text(x0 + w0 / 2, y_neg + h_data * 0.34, "(Healthy)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#4b5563")

    # Cell Data
    # TP (True Positive)
    tp_pct = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    ax.text(x1 + w1 / 2, y_pos + 1.20, "True Positive (TP)", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#14532d")
    ax.text(x1 + w1 / 2, y_pos + 0.74, f"{tp:,}", ha="center", va="center",
            fontsize=19, fontweight="bold", color="#000000")
    ax.text(x1 + w1 / 2, y_pos + 0.32, f"({tp_pct:.1f}% of Anomalies)", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#166534")

    # FN (False Negative)
    fn_pct = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0
    ax.text(x2 + w2 / 2, y_pos + 1.20, "False Negative (FN)", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#991b1b")
    ax.text(x2 + w2 / 2, y_pos + 0.74, f"{fn:,}", ha="center", va="center",
            fontsize=19, fontweight="bold", color="#000000")
    ax.text(x2 + w2 / 2, y_pos + 0.32, f"({fn_pct:.1f}% Missed)", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#7f1d1d")

    # FP (False Positive)
    fp_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    ax.text(x1 + w1 / 2, y_neg + 1.20, "False Positive (FP)", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#991b1b")
    ax.text(x1 + w1 / 2, y_neg + 0.74, f"{fp:,}", ha="center", va="center",
            fontsize=19, fontweight="bold", color="#000000")
    ax.text(x1 + w1 / 2, y_neg + 0.32, f"({fp_pct:.1f}% False Alarms)", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#7f1d1d")

    # TN (True Negative)
    tn_pct = (tn / (fp + tn) * 100) if (fp + tn) > 0 else 0
    ax.text(x2 + w2 / 2, y_neg + 1.20, "True Negative (TN)", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#14532d")
    ax.text(x2 + w2 / 2, y_neg + 0.74, f"{tn:,}", ha="center", va="center",
            fontsize=19, fontweight="bold", color="#000000")
    ax.text(x2 + w2 / 2, y_neg + 0.32, f"({tn_pct:.1f}% of Healthy)", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#166534")

    # Summary metrics footer
    total_test = tp + fn + fp + tn
    metrics_str = (
        f"Hold-out Test (N={total_test:,})   |   "
        f"Accuracy: {acc:.4f}   |   Precision: {prec:.4f}   |   "
        f"Recall: {rec:.4f}   |   F1 Score: {f1:.4f}   |   "
        f"False Alarm Rate: {far:.4f}"
    )
    ax.text(6.0, 0.55, metrics_str, ha="center", va="center", fontsize=11, fontweight="bold",
            color="#0f172a", bbox=dict(boxstyle="round,pad=0.55", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2))

    short_name = "static" if "static" in name else ("cyclic" if "cyclic" in name else name)
    out_png_short = eval_dir / f"confusion_matrix_{short_name}.png"

    plt.savefig(out_png_short, bbox_inches="tight")
    plt.close(fig)

    log.info("[METRICS][%s] Confusion matrix saved: %s", name, out_png_short.name)

    return {
        "confusion_matrix_png": str(out_png_short)
    }


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_log_file(name, ts_file, metrics):
    """Append human-readable metrics block to logs/training_metrics_<ts>.log."""
    try:
        log_dir = abs_path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"training_metrics_{ts_file}.log"

        perf = metrics.get("performance", {})
        split = metrics.get("split", {})
        thresh = metrics.get("threshold", {})

        lines = [
            "=" * 72,
            f"  TRAINING METRICS  |  {metrics['equipment_type'].upper()}",
            f"  Run timestamp     : {metrics['run_timestamp']}",
            f"  Trained at        : {metrics.get('trained_at', 'n/a')}",
            f"  Method            : {metrics.get('method', 'n/a')}",
            "=" * 72,
            "",
            "-- Split Statistics --------------------------------------------",
            f"  Train rows            : {split.get('train', 0)}",
            f"  Calibration rows      : {split.get('calib', 0)}",
            f"  Test  rows            : {split.get('test', 0)}",
            f"  Healthy rows          : {split.get('total_healthy', 0)}  |  Unhealthy rows : {split.get('total_unhealthy', 0)}",
            "",
            "-- Threshold / Calibration -------------------------------------",
            f"  Threshold value       : {thresh.get('value')}",
            f"  Calib score mean      : {thresh.get('calib_score_mean')}",
            f"  Calib score std       : {thresh.get('calib_score_std')}",
            f"  Calib score min/max   : {thresh.get('calib_score_min')} / {thresh.get('calib_score_max')}",
            f"  Implied percentile    : {thresh.get('implied_percentile_pct')}%",
            "",
            "-- Hold-out Test Data Evaluation Metrics -----------------------",
            f"  Accuracy              : {perf.get('accuracy')}",
            f"  Precision             : {perf.get('precision')}",
            f"  Recall                : {perf.get('recall')}",
            f"  F1 Score              : {perf.get('f1')}",
            f"  False Alarm Rate      : {perf.get('false_alarm_rate')}",
            f"  ROC-AUC               : {perf.get('roc_auc')}",
            f"  PR-AUC                : {perf.get('pr_auc')}",
            "",
        ]

        per_class = metrics.get("per_class", [])
        if per_class:
            lines.append("-- Per Anomaly Fault Type Metrics (Hold-out Test Set) -----------")
            col_fmt = "  %-26s  %9s  %9s  %9s  %9s  %16s  %9s  %9s  %8s"
            lines.append(
                col_fmt % ("Fault Type", "Accuracy", "Precision", "Recall", "F1 Score", "False Alarm", "ROC-AUC", "PR-AUC", "Support")
            )
            lines.append("  " + "-" * 110)
            for r in per_class:
                if not isinstance(r, dict) or "class" not in r:
                    continue
                auc_s = f"{r['roc_auc']:.4f}" if r.get('roc_auc') is not None else "n/a"
                pr_s = f"{r['pr_auc']:.4f}" if r.get('pr_auc') is not None else "n/a"
                acc_s = f"{r['accuracy']:.4f}" if r.get('accuracy') is not None else "n/a"
                p_s = f"{r['precision']:.4f}" if r.get('precision') is not None else "n/a"
                rec_s = f"{r['recall']:.4f}" if r.get('recall') is not None else "n/a"
                f_s = f"{r['f1']:.4f}" if r.get('f1') is not None else "n/a"
                far_s = f"{r['false_alarm_rate']:.4f}" if r.get('false_alarm_rate') is not None else "n/a"
                lines.append(
                    col_fmt % (
                        r['class'],
                        acc_s,
                        p_s,
                        rec_s,
                        f_s,
                        far_s,
                        auc_s,
                        pr_s,
                        str(r.get('support', 0)),
                    )
                )
            lines.append("")

        lines += ["=" * 72, ""]

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        log.info("[METRICS] Log file written: %s", log_path)

    except Exception as exc:
        log.warning("[METRICS] Could not write metrics log file: %s", exc)


def _write_json_file(name, ts_file, metrics):
    """Write the clean metrics dict as JSON to reports/training_metrics_<name>_<ts>.json."""
    try:
        rep_dir = abs_path("reports")
        rep_dir.mkdir(parents=True, exist_ok=True)
        json_path = rep_dir / f"training_metrics_{name}_{ts_file}.json"

        # Sanitize internal array references from JSON output
        clean_metrics = {}
        for k, v in metrics.items():
            if k == "performance" and isinstance(v, dict):
                clean_metrics[k] = {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
            elif k == "per_class" and isinstance(v, list):
                clean_metrics[k] = [{pk: pv for pk, pv in item.items() if not pk.startswith("_")} for item in v]
            else:
                clean_metrics[k] = v

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Unserializable: {type(obj)}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2, default=_convert)

        log.info("[METRICS] JSON report written: %s", json_path)

    except Exception as exc:
        log.warning("[METRICS] Could not write metrics JSON: %s", exc)


# ---------------------------------------------------------------------------
# Directory cleanup and standalone runner
# ---------------------------------------------------------------------------

def clean_evaluation_results_dir():
    """Ensure evaluation_results/ only contains the requested confusion matrix PNGs (static and cyclic)."""
    eval_dir = abs_path("evaluation_results")
    if not eval_dir.exists():
        return
    allowed = {"confusion_matrix_static.png", "confusion_matrix_cyclic.png"}
    for p in eval_dir.glob("*"):
        if p.is_file() and p.name not in allowed:
            try:
                p.unlink()
            except Exception:
                pass


def evaluate_both_models(config_path=None):
    """
    Load saved models and datasets for both Static and Cyclic equipment,
    evaluate on hold-out test data, print the requested metrics to the logger,
    and generate visual confusion matrix PNGs in evaluation_results/.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    else:
        logging.getLogger().setLevel(logging.INFO)
    cfg = load_config(config_path)
    from . import ingest, model

    eval_results = {}
    for name, et in cfg["equipment_types"].items():
        if not et.get("enabled", True):
            continue
        log.info(">>> Evaluating %s (%s) <<<", name, et["mode"])
        try:
            art = model.load(name)
            df = ingest.load_table(et)
            m = log_training_metrics(name, art, df)
            eval_results[name] = m
        except Exception as e:
            log.error("[%s] Evaluation failed: %s", name, e)
            log.debug(traceback.format_exc())

    clean_evaluation_results_dir()
    return eval_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_both_models()
