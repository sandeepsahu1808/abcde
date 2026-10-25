"""Step 4 - TRAIN on healthy data only, then save the model artefact."""
import json
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import sklearn
from . import features, quality
from .baseline import Baseline
from .config import abs_path
from . import metrics_logger

log = logging.getLogger(__name__)


def assign_split(df, eligible, split_cfg, seed=42):
    """Episode-aware split per asset.

    Guarantees:
    1. Fault episodes / cycles are NEVER sliced in two.
    2. Zero fault episodes cut across splits.
    3. For cyclic machinery (Cycle ID present): Stratified cycle-level splitting across all fault types
       so that every fault class is represented in training, validation, and test splits without leakage.
    4. Supports 3-way (train, calib, test) or 4-way (train, val, calib, test) splitting.
    """
    cycle_col = "Cycle ID" if "Cycle ID" in df.columns else ("cycle_id" if "cycle_id" in df.columns else None)
    if cycle_col is not None and "_fault" in df.columns:
        split = pd.Series("test", index=df.index)
        obj_col = "object_id" if "object_id" in df.columns else ("Crane ID" if "Crane ID" in df.columns else None)
        if obj_col is not None:
            cycle_key = df[obj_col].astype(str) + "__" + df[cycle_col].astype(str)
        else:
            cycle_key = df[cycle_col].astype(str)
        cycle_faults = df.groupby(cycle_key)["_fault"].first()
        rng = np.random.RandomState(seed)
        
        train_cycles, val_cycles, calib_cycles, test_cycles = [], [], [], []
        r_tr_raw = split_cfg.get("train", 0.70)
        r_val = split_cfg.get("val_from_train", split_cfg.get("val", 0.10))
        r_tr = max(0.0, r_tr_raw - r_val) if "val_from_train" in split_cfg else r_tr_raw
        r_ca = split_cfg.get("calib", 0.15)
        
        for fault, cycles in cycle_faults.groupby(cycle_faults):
            c_list = list(cycles.index)
            rng.shuffle(c_list)
            n = len(c_list)
            if fault in ["Normal", "Healthy", "normal", "healthy"]:
                n_tr = int(round(n * r_tr))
                n_val = int(round(n * r_val))
                n_ca = int(round(n * r_ca))
                train_cycles.extend(c_list[:n_tr])
                val_cycles.extend(c_list[n_tr:n_tr + n_val])
                calib_cycles.extend(c_list[n_tr + n_val:n_tr + n_val + n_ca])
                test_cycles.extend(c_list[n_tr + n_val + n_ca:])
            else:
                # Fault cycles: guarantee at least 1 in test (and at least 1 in val if n >= 4)
                if n <= 2:
                    train_cycles.append(c_list[0])
                    test_cycles.extend(c_list[1:])
                elif n == 3:
                    train_cycles.extend(c_list[:2])
                    test_cycles.append(c_list[2])
                else:
                    n_te = max(1, int(round(n * 0.20)))
                    n_val = 1
                    n_tr = n - n_te - n_val
                    train_cycles.extend(c_list[:n_tr])
                    val_cycles.extend(c_list[n_tr:n_tr + n_val])
                    test_cycles.extend(c_list[n_tr + n_val:])

        split.loc[cycle_key.isin(train_cycles)] = "train"
        split.loc[cycle_key.isin(val_cycles)] = "val"
        split.loc[cycle_key.isin(calib_cycles)] = "calib"
        split.loc[cycle_key.isin(test_cycles)] = "test"
        return split

    bs = split_cfg.get("block", 24)
    split = pd.Series(np.where(df["_healthy"], "excluded", "unhealthy"), index=df.index)
    sub = df[eligible]

    has_val = "val" in split_cfg
    r_tr = split_cfg.get("train", 0.70)
    r_val = split_cfg.get("val", 0.0) if has_val else 0.0
    r_ca = split_cfg.get("calib", 0.15)

    for _, idx in sub.groupby("object_id").indices.items():
        ids = sub.index[idx]
        sub_df = sub.loc[ids].sort_values("timestamp")
        sub_ids = sub_df.index.values
        n = len(sub_ids)

        is_fault = (~sub_df["_healthy"]).values

        runs = []
        if is_fault.any():
            pos = np.flatnonzero(is_fault)
            for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
                runs.append((s[0], s[-1]))

        if has_val:
            c1 = int(round(n * r_tr))
            c2 = int(round(n * (r_tr + r_val)))
            c3 = int(round(n * (r_tr + r_val + r_ca)))
            cuts = [c1, c2, c3]
        else:
            c1 = int(round(n * r_tr))
            c2 = int(round(n * (r_tr + r_ca)))
            cuts = [c1, c2]

        snapped_cuts = []
        for c in cuts:
            snapped = c
            for s_start, s_end in runs:
                if s_start <= c <= s_end:
                    # Snap to start of episode so the entire episode belongs to the later split
                    snapped = s_start
                    break
            snapped_cuts.append(snapped)

        if has_val:
            sc1, sc2, sc3 = snapped_cuts
            split.loc[sub_ids[:sc1]] = "train"
            split.loc[sub_ids[sc1:sc2]] = "val"
            split.loc[sub_ids[sc2:sc3]] = "calib"
            split.loc[sub_ids[sc3:]] = "test"
        else:
            sc1, sc2 = snapped_cuts
            split.loc[sub_ids[:sc1]] = "train"
            split.loc[sub_ids[sc1:sc2]] = "calib"
            split.loc[sub_ids[sc2:]] = "test"

    return split


def create_validation_split(df, tr_mask, val_ratio=0.20):
    """Carve out an asset/time- and episode-aware validation split inside the training data.

    Takes the last val_ratio fraction of training data per asset, snapped to episode
    boundaries so that zero fault episodes are cut in half.
    """
    val_mask = np.zeros(len(df), dtype=bool)
    sub = df[tr_mask]
    for _, idx in sub.groupby("object_id").indices.items():
        ids = sub.index[idx]
        sub_df = sub.loc[ids].sort_values("timestamp")
        sub_ids = sub_df.index.values
        n_tr = len(sub_ids)
        is_fault = (~sub_df["_healthy"]).values

        runs = []
        if is_fault.any():
            pos = np.flatnonzero(is_fault)
            for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
                runs.append((s[0], s[-1]))

        c_val = int(round(n_tr * (1.0 - val_ratio)))
        for s_start, s_end in runs:
            if s_start <= c_val <= s_end:
                c_val = s_start
                break
        val_mask[sub_ids[c_val:]] = True
    return val_mask


def _compute_val_metrics(iso, Z_combined, le, df, val_mask, thr, name):
    """Compute validation metrics on val_mask to guide XGBoost parameter selection."""
    if not val_mask.any():
        return {}
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
    X_val = Z_combined.values[val_mask]
    y_str = df["_fault"].values[val_mask]
    class_map = {c: i for i, c in enumerate(le.classes_)}
    y_val_idx = np.array([class_map.get(c, -1) for c in y_str])
    valid_mask = y_val_idx >= 0
    if not valid_mask.any():
        return {}

    proba = iso.predict_proba(X_val[valid_mask])
    pred_idx = np.argmax(proba, axis=1)
    y_true = y_val_idx[valid_mask]

    acc = float(accuracy_score(y_true, pred_idx))
    macro_f1 = float(f1_score(y_true, pred_idx, average="macro", zero_division=0))
    try:
        ll = float(log_loss(y_true, proba, labels=list(range(len(le.classes_))))) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        ll = 0.0


    normal_idx = class_map.get("Normal", 0)
    is_fault = (y_true != normal_idx)
    anomaly_score = 1.0 - proba[:, normal_idx]
    pred_fault = (anomaly_score >= thr)

    bin_prec = float(precision_score(is_fault, pred_fault, zero_division=0))
    bin_rec = float(recall_score(is_fault, pred_fault, zero_division=0))
    bin_f1 = float(f1_score(is_fault, pred_fault, zero_division=0))
    try:
        bin_auc = float(roc_auc_score(is_fault, anomaly_score)) if len(np.unique(is_fault)) > 1 else None
    except Exception:
        bin_auc = None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "binary_f1": bin_f1,
        "binary_auc": bin_auc,
        "val_rows": int(val_mask.sum()),
    }


def anomaly_score(method, iso, Z, params, X=None, return_proba=False):
    """xgboost: supervised learning model (best with labeled data).
    max_z: largest load-adjusted |z| across parameters (robust & explainable)."""
    if method == "max_z":
        if return_proba:
            return Z[params].abs().max(axis=1).values, None, None, []
        return Z[params].abs().max(axis=1).values, None
    elif method == "xgboost":
        import pandas as _pd
        # Combine raw features + residuals (same as training)
        if X is not None:
            Z_combined = _pd.concat([X.reset_index(drop=True), Z.reset_index(drop=True).add_prefix("z_")], axis=1)
        else:
            Z_combined = Z
        if hasattr(iso, "_feature_names") and iso._feature_names:
            Z_combined = Z_combined[iso._feature_names]
        proba = iso.predict_proba(Z_combined.values)
        le = iso._label_encoder
        class_names = list(le.classes_)
        # Find the index of the "Normal" class
        normal_idx = class_names.index("Normal") if "Normal" in class_names else 0
        # Anomaly score = 1 - P(Normal), i.e. total probability of being ANY fault
        anomaly_prob = 1.0 - proba[:, normal_idx]
        # Predicted fault = class with highest probability, decoded back to string
        predicted_idx = proba.argmax(axis=1)
        predicted_class = le.inverse_transform(predicted_idx)
        if return_proba:
            return anomaly_prob, predicted_class, proba, class_names
        return anomaly_prob, predicted_class
    raise ValueError(f"Unknown score_method: {method}")



def train(name, et, df):
    params = list(et["parameters"])
    null_policy = et.get("null_policy", {"transient_limit": 3, "trailing_window": 24})
    df, _ = quality.impute_nulls(df, params, null_policy)
    healthy = df["_healthy"].values
    mc = et["model"]
    method = mc.get("score_method", "xgboost")

    # ── Step 1: Assign splits first to prevent any data leakage ──────────
    if method == "xgboost":
        eligible = df["_labelled"].values
    else:
        eligible = healthy & ~df.get("_any_spike", pd.Series(False, index=df.index)).values

    split = assign_split(df, eligible, et["split"])
    tr, ca, te = (split == "train").values, (split == "calib").values, (split == "test").values

    # ── Step 2: Create validation split inside the training data ─────────
    # Carve out an episode-aware validation partition inside training data
    # to evaluate and tune hyperparameters without touching the final test set.
    if (split == "val").any():
        val_mask = (split == "val").values
        tr_fit = tr
    else:
        val_mask = create_validation_split(df, tr, val_ratio=et["split"].get("val_ratio", 0.20))
        tr_fit = tr & ~val_mask

    if (healthy & tr_fit).sum() < 20 or (healthy & ca).sum() < 5:
        raise ValueError(f"[{name}] not enough healthy rows to train (train={(healthy & tr_fit).sum()}, calib={(healthy & ca).sum()})")

    # ── Step 3: Fit sensor scaling strictly on training data only ────────
    scale = df.loc[healthy & tr_fit, params].std(ddof=0).clip(lower=1e-9).to_dict()

    # ── Step 4: Causal spike detection with train-only scale ──────────────
    dev_ref, any_spike = {}, pd.Series(False, index=df.index)
    df_clean = df
    if et.get("spike"):
        sp, dev, med = quality.spike_scan(
            df, params, scale, et["spike"]["k"], et["spike"]["max_len"], causal=True
        )
        df_clean = quality.apply_clean(df, params, sp, med)
        any_spike = sp.any(axis=1)
        dev_ref = {p: np.sort(dev.loc[healthy & tr_fit, p].values) for p in params}

    # ── Step 5: Feature engineering ──────────────────────────────────────
    feat_df = df if method == "xgboost" else df_clean
    X, C, window_ok = features.build(feat_df, et)

    # ── Step 6: Fit baseline residuals strictly on healthy training data ──
    base = Baseline().fit(X[healthy & tr_fit], C[healthy & tr_fit])
    Z = base.residuals(X, C)

    # ── Step 7: Mahalanobis distance strictly on healthy training data ───
    healthy_mu = None
    healthy_cov_inv = None
    if method != "xgboost":
        healthy_X = X.loc[healthy & tr_fit, params].values
        healthy_mu = healthy_X.mean(axis=0)
        healthy_cov = np.cov(healthy_X, rowvar=False) + np.eye(len(params)) * 1e-6
        healthy_cov_inv = np.linalg.pinv(healthy_cov)
        X = features.add_mahalanobis(X, healthy_mu, healthy_cov_inv, params)

    iso = None
    val_metrics = {}
    if method == "xgboost":
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        import pandas as _pd

        le_global = LabelEncoder()
        le_global.fit(df["_fault"].values)

        fit_fault_labels = df["_fault"].values
        Z_combined = _pd.concat(
            [X.reset_index(drop=True), Z.reset_index(drop=True).add_prefix("z_")],
            axis=1,
        )
        fit_idx = tr_fit if len(np.unique(fit_fault_labels[tr_fit])) >= 2 else np.arange(len(df))

        le_fit = LabelEncoder()
        y_fit = le_fit.fit_transform(fit_fault_labels[fit_idx])

        n_classes = len(le_fit.classes_)
        log.info("[%s] XGBoost classes in fit_idx (%d): %s", name, n_classes, list(le_fit.classes_))

        xgb_kwargs = {
            "n_estimators": mc.get("n_estimators", 300),
            "max_depth": mc.get("max_depth", 8),
            "learning_rate": mc.get("learning_rate", 0.05),
            "random_state": mc.get("random_state", 42),
            "subsample": mc.get("subsample", 0.8),
            "colsample_bytree": mc.get("colsample_bytree", 0.8),
            "min_child_weight": mc.get("min_child_weight", 1),
            "reg_alpha": mc.get("reg_alpha", 0.0),
            "reg_lambda": mc.get("reg_lambda", 1.0),
            "gamma": mc.get("gamma", 0.0),
            "n_jobs": -1,
        }
        if n_classes > 2:
            xgb_kwargs["objective"] = "multi:softprob"
            xgb_kwargs["num_class"] = n_classes
            xgb_kwargs["eval_metric"] = "mlogloss"
        else:
            xgb_kwargs["objective"] = "binary:logistic"
            xgb_kwargs["eval_metric"] = "logloss"

        iso = XGBClassifier(**xgb_kwargs).fit(Z_combined.values[fit_idx], y_fit)

        iso._label_encoder = le_fit
        iso._global_encoder = le_global
        iso._feature_names = list(Z_combined.columns)

    score_result = anomaly_score(method, iso, Z, params, X=X)
    scores = score_result[0] if isinstance(score_result, tuple) else score_result

    # ── Step 8: Calibrate threshold strictly on healthy calibration rows ──
    calib_scores = np.sort(scores[ca & healthy])
    thr = float(np.percentile(calib_scores, et["detection"]["threshold_percentile"]))

    # ── Step 9: Evaluate validation metrics on val_mask (test set untouched) ──
    if method == "xgboost" and val_mask.any():
        val_metrics = _compute_val_metrics(iso, Z_combined, le_fit, df, val_mask, thr, name)
        log.info(
            "[%s] Validation metrics (N=%d) | Macro F1: %.4f | Accuracy: %.4f | Log Loss: %.4f | Bin F1: %.4f",
            name, val_metrics.get("val_rows", 0), val_metrics.get("macro_f1", 0),
            val_metrics.get("accuracy", 0), val_metrics.get("log_loss", 0),
            val_metrics.get("binary_f1", 0),
        )

    # ── Step 10: Store split_keys with train, val, calib, test ───────────
    split_col = split.copy()
    split_col.loc[val_mask] = "val"
    split_keys = pd.concat([df[["object_id", "timestamp"]], split_col.rename("split")], axis=1)

    art = {
        "method": method,
        "params": params,
        "name": name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "baseline": base,
        "iso": iso,
        "scale": scale,
        "dev_ref": dev_ref,
        "threshold": thr,
        "calib_scores": calib_scores,
        "healthy_mu": healthy_mu,
        "healthy_cov_inv": healthy_cov_inv,
        "val_metrics": val_metrics,
        "et": et,
        "split_keys": split_keys,
        "counts": {
            "train": int(tr_fit.sum()),
            "val": int(val_mask.sum()),
            "calib": int(ca.sum()),
            "test": int(te.sum()),
        },
    }
    log.info("[%s] trained: %s, threshold=%.4f", name, art["counts"], thr)

    try:
        metrics_logger.log_training_metrics(name, art, df)
    except Exception as _me:
        log.warning("[%s] metrics logging failed (non-fatal): %s", name, _me)

    return art


def save(art):
    d = abs_path("models") / art["name"]
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, d / "model.joblib")
    meta = {k: art[k] for k in ("name", "trained_at", "sklearn", "threshold", "counts")}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    return d


def load(name):
    import sys
    from pathlib import Path
    _src_dir = str(Path(__file__).resolve().parents[1])
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    return joblib.load(abs_path("models") / name / "model.joblib")
