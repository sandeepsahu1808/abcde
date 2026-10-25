"""Step 7 - measure how good the detector is (needs labels; PoC only)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import abs_path

def plot_evaluation(o, art, name):
    plt.figure(figsize=(10, 6))
    healthy_scores = o.loc[o["_healthy"], "score"].dropna()
    unhealthy_scores = o.loc[~o["_healthy"], "score"].dropna()
    
    if len(healthy_scores) > 0:
        plt.hist(healthy_scores, bins=50, alpha=0.5, label="Healthy", density=True)
    if len(unhealthy_scores) > 0:
        plt.hist(unhealthy_scores, bins=50, alpha=0.5, label="Un-Healthy", density=True)
    
    if "threshold" in art:
        plt.axvline(x=art["threshold"], color='r', linestyle='--', label=f'Threshold ({art["threshold"]:.2f})')
        
    plt.title(f"Anomaly Score Distribution - {name} ({art.get('trained_at', 'unknown')})")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    
    safe_time = art.get('trained_at', 'unknown').replace(':', '-')
    filename = abs_path("reports") / f"eval_dist_{name}_{safe_time}.png"
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
    plt.close()


def _runs(mask):
    pos = np.flatnonzero(mask)
    return [] if not len(pos) else np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1)


def evaluate(out, art, name, base_rate=0.05):
    if not out["_labelled"].iloc[0]:
        return None, None
    if "split_keys" not in art or art["split_keys"] is None:
        raise ValueError(f"[{name}] Artefact is missing split_keys. Evaluation must be performed strictly on hold-out test rows.")

    o = out.merge(art["split_keys"], on=["object_id", "timestamp"], how="left")
    if "split" not in o.columns or not (o["split"] == "test").any():
        raise ValueError(
            f"[{name}] No hold-out test split found in dataset/artefact. "
            "Evaluation must be performed strictly on hold-out test rows; fallback to all rows is prohibited."
        )

    test_mask = (o["split"] == "test").values
    test_healthy = test_mask & o["_healthy"].values
    test_faulty = test_mask & (~o["_healthy"]).values

    truth = (~o["_healthy"]).values
    truth_test = truth[test_mask]

    # --- Classifier-level predictions (Model alone) ---
    pred_model = (o["flag_model"]).values if "flag_model" in o.columns else (o["score"] >= art.get("threshold", 0.5)).values
    pred_model_test = pred_model[test_mask]

    rec_model = float(pred_model[test_faulty].mean()) if test_faulty.any() else np.nan
    far_model = float(pred_model[test_healthy].mean()) if test_healthy.any() else np.nan
    prec_model = float(truth_test[pred_model_test].mean()) if pred_model_test.any() else np.nan

    # --- End-to-End system predictions (Model + Rule breaches + Sensor Quality) ---
    pred_e2e = (o["flag"] | o["suspect_any"]).values
    pred_e2e_test = pred_e2e[test_mask]

    rec_e2e = float(pred_e2e[test_faulty].mean()) if test_faulty.any() else np.nan
    far_e2e = float(pred_e2e[test_healthy].mean()) if test_healthy.any() else np.nan
    prec_e2e = float(truth_test[pred_e2e_test].mean()) if pred_e2e_test.any() else np.nan

    est_prec_e2e = (
        round(rec_e2e * base_rate / (rec_e2e * base_rate + far_e2e * (1 - base_rate)), 4)
        if far_e2e > 0 or rec_e2e > 0
        else np.nan
    )

    # --- Alarm Source Attribution on hold-out test set ---
    flag_m = o["flag_model"].values[test_mask] if "flag_model" in o.columns else pred_model_test
    flag_r = o["flag_rule"].values[test_mask] if "flag_rule" in o.columns else np.zeros(len(test_mask), dtype=bool)
    flag_s = o["suspect_any"].values[test_mask] if "suspect_any" in o.columns else np.zeros(len(test_mask), dtype=bool)

    alarms_model_only = int((flag_m & ~flag_r & ~flag_s).sum())
    alarms_rule_only = int((flag_r & ~flag_m & ~flag_s).sum())
    alarms_sensor_only = int((flag_s & ~flag_m & ~flag_r).sum())
    alarms_multi_source = int(((flag_m.astype(int) + flag_r.astype(int) + flag_s.astype(int)) > 1).sum())

    overrides_applied = 0
    if "predicted_fault_model" in o.columns and "predicted_fault" in o.columns:
        overrides_applied = int((o["predicted_fault"].values[test_mask] != o["predicted_fault_model"].values[test_mask]).sum())

    summ = [
        ("Hold-out test rows (N)", int(test_mask.sum())),
        ("Healthy hold-out rows (never seen in training)", int(test_healthy.sum())),
        ("[Classifier-Only] Row recall on hold-out test rows", round(rec_model, 4)),
        ("[Classifier-Only] Precision on hold-out test rows", round(prec_model, 4) if not np.isnan(prec_model) else np.nan),
        ("[Classifier-Only] False-alarm rate on hold-out healthy rows", round(far_model, 4)),
        ("[End-to-End] Row recall on hold-out test rows", round(rec_e2e, 4)),
        ("[End-to-End] Precision on hold-out test rows", round(prec_e2e, 4) if not np.isnan(prec_e2e) else np.nan),
        ("[End-to-End] False-alarm rate on hold-out healthy rows", round(far_e2e, 4)),
        (f"[End-to-End] Estimated precision at a realistic {int(base_rate*100)}% fault rate", est_prec_e2e),
        ("[Alarms] Model-only alarms in test split", alarms_model_only),
        ("[Alarms] Rule-only breach alarms in test split", alarms_rule_only),
        ("[Alarms] Sensor-quality suspect alarms in test split", alarms_sensor_only),
        ("[Alarms] Multi-source overlapping alarms in test split", alarms_multi_source),
        ("[Overrides] Rule overrides applied to classifier predictions", overrides_applied),
    ]

    # --- Per-Fault breakdown on hold-out test set ---
    rows = []
    test_o = o[test_mask].reset_index(drop=True)
    test_truth = (~test_o["_healthy"]).values
    pred_m_o = pred_model[test_mask]
    pred_e_o = pred_e2e[test_mask]

    for f in sorted(test_o.loc[test_truth, "_fault"].unique()):
        m = (test_o["_fault"] == f).values
        n_ep = det_m = det_e = 0
        delays = []
        for obj in test_o["object_id"].unique():
            sel = (test_o["object_id"] == obj).values
            for r in _runs(m[sel]):
                n_ep += 1
                hit_m = np.flatnonzero(pred_m_o[sel][r])
                if len(hit_m):
                    det_m += 1
                hit_e = np.flatnonzero(pred_e_o[sel][r])
                if len(hit_e):
                    det_e += 1
                    delays.append(int(hit_e[0]))
        rows.append({
            "fault_type": f,
            "rows": int(m.sum()),
            "row_recall": round(float(pred_e_o[m].mean()), 3),
            "row_recall_model": round(float(pred_m_o[m].mean()), 3),
            "row_recall_e2e": round(float(pred_e_o[m].mean()), 3),
            "labelled_episodes": n_ep,
            "episodes_detected": det_e,
            "episodes_detected_model": det_m,
            "episodes_detected_e2e": det_e,
            "median_delay_rows": float(np.median(delays)) if delays else np.nan,
        })
    by_fault = pd.DataFrame(rows)

    ep_rec_model = round(by_fault["episodes_detected_model"].sum() / max(by_fault["labelled_episodes"].sum(), 1), 4) if not by_fault.empty else 0.0
    ep_rec_e2e = round(by_fault["episodes_detected_e2e"].sum() / max(by_fault["labelled_episodes"].sum(), 1), 4) if not by_fault.empty else 0.0

    summ.insert(5, ("[Classifier-Only] Episode-level recall (hit at least once)", ep_rec_model))
    summ.append(("[End-to-End] Episode-level recall (hit at least once)", ep_rec_e2e))

    summary = pd.DataFrame(summ, columns=["metric", "value"])
    summary.insert(0, "equipment_type", name)
    by_fault.insert(0, "equipment_type", name)

    try:
        plot_evaluation(o[test_mask], art, name)
    except Exception as e:
        print(f"Failed to plot evaluation: {e}")

    return summary, by_fault
