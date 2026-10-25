"""Step 6 - turn per-row alarms into ONE record per anomaly episode per parameter."""
import numpy as np
import pandas as pd

COLS = ["equipment_type", "episode_id", "object_id", "category", "parameter_column", "start", "end",
        "value", "z", "confidence", "outcome", "source", "basis", "n_flagged", "peak_score", "predicted_fault", "cycle_id", "phase"]


def ecdf_conf(value, ref_sorted, cap=99.9):
    """Confidence = share of HEALTHY reference scores that are lower than this value."""
    if len(ref_sorted) == 0:
        return cap
    return float(min(cap, round(np.searchsorted(ref_sorted, value, side="right") / len(ref_sorted) * 100, 1)))


def longest_run(pos):
    if len(pos) == 0:
        return 0
    return max(len(s) for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1))


FAULT_PRIMARY_PARAM = {
    "Overheat": "Temperature (Degree Cel)",
    "Contextual Overheat": "Temperature (Degree Cel)",
    "Bearing Degradation": "Vibration (mm/s)",
    "Incipient Drift": "Vibration (mm/s)",
    "Correlation Break": "Vibration (mm/s)",
    "Overpressure": "Pressure (bars)",
    "Efficiency Shift": "Pressure (bars)",
    "Overspeed": "Engine RPM",
    "Underspeed": "Engine RPM",
}


def model_episodes(out, et, art, name):
    params, det = list(et["parameters"]), et["detection"]
    fault_map = et.get("fault_primary_sensor", {})
    recs, n_ep = [], 0

    flagged_mask = out["flag"].values
    if not flagged_mask.any():
        return recs

    flagged_out = out[flagged_mask]
    records = flagged_out.to_dict(orient="records")

    for row in records:
        n_ep += 1
        outcome = "Anomaly"
        peak = float(row.get("score", 0.0))

        conf_val = row.get("confidence")
        if conf_val is not None and pd.notna(conf_val):
            conf = float(conf_val)
        else:
            model_conf = ecdf_conf(peak, art["calib_scores"])
            has_rule = bool(row.get("flag_rule", False))
            conf = max(model_conf, det["rule_confidence"]) if has_rule else model_conf

        has_rule = bool(row.get("flag_rule", False))
        has_model = bool(row.get("flag_model", False))
        basis = "Model + Limit" if (has_rule and has_model) else ("Limit" if has_rule else "Model")
        ep_fault = row.get("predicted_fault", "Unknown")

        desired_col = fault_map.get(ep_fault) or FAULT_PRIMARY_PARAM.get(ep_fault)
        if desired_col and desired_col in params:
            p = desired_col
        else:
            breaching = [q for q in params if bool(row.get(f"breach|{q}", False))]
            if breaching:
                p = max(breaching, key=lambda q: abs(row.get(f"z|{q}", 0.0)))
            else:
                p = max(params, key=lambda q: abs(row.get(f"z|{q}", 0.0)))

        cat = et["parameters"][p]["category"]
        cycle_val = row.get("Cycle ID") if ("Cycle ID" in row and pd.notna(row["Cycle ID"])) else row.get("cycle_id", "")
        phase_val = row.get("Move Phase") if ("Move Phase" in row and pd.notna(row["Move Phase"])) else row.get("phase", "")

        recs.append({"equipment_type": name, "episode_id": f"{name}-ROW-{n_ep:05d}", "object_id": row.get("object_id"),
                     "category": cat, "parameter_column": p, "start": row.get("timestamp"),
                     "end": row.get("timestamp"), "value": float(row.get(p, 0.0)),
                     "z": float(row.get(f"z|{p}", 0.0)), "confidence": conf, "outcome": outcome,
                     "source": "Model", "basis": basis, "n_flagged": 1, "peak_score": peak,
                     "predicted_fault": ep_fault, "cycle_id": cycle_val, "phase": phase_val})
    return recs


def spike_episodes(out, et, art, name):
    recs, n_spk = [], 0
    for p, pc in et["parameters"].items():
        for obj, g in out.groupby("object_id", sort=False):
            g = g.reset_index(drop=True)
            pos = np.flatnonzero(g[f"spike|{p}"].values)
            if not len(pos):
                continue
            for i in pos:
                n_spk += 1
                row = g.iloc[i]
                conf = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else ecdf_conf(row[f"dev|{p}"], art["dev_ref"].get(p, []))
                cycle_val = row["Cycle ID"] if "Cycle ID" in row and pd.notna(row["Cycle ID"]) else row.get("cycle_id", "")
                phase_val = row["Move Phase"] if "Move Phase" in row and pd.notna(row["Move Phase"]) else row.get("phase", "")
                recs.append({"equipment_type": name, "episode_id": f"{name}-SPK-{n_spk:05d}",
                             "object_id": obj, "category": pc["category"], "parameter_column": p,
                             "start": row["timestamp"], "end": row["timestamp"],
                             "value": float(row[p]), "z": float(row[f"dev|{p}"]),
                             "confidence": conf,
                             "outcome": "Sensor Suspect", "source": "Spike rule", "basis": "Spike rule",
                             "n_flagged": 1, "peak_score": np.nan,
                             "predicted_fault": "Sensor Spike", "cycle_id": cycle_val, "phase": phase_val})
    return recs


def flatline_episodes(out, flat_runs, et, name):
    recs, n_flt = [], 0
    if not len(flat_runs):
        return recs
    fl = et.get("flatline", {})
    if et.get("mode") == "cycle":
        col = fl.get("column", "Sensor")
        cat = fl.get("category", "Sensor")
        conf = fl.get("confidence", 99.9)
        for r in flat_runs.itertuples():
            recs.append({"equipment_type": name, "episode_id": f"{name}-FLT-{r.object_id}-{r.Index}",
                         "object_id": r.object_id, "category": getattr(r, "category", cat),
                         "parameter_column": getattr(r, "column", col), "start": r.start, "end": r.end,
                         "value": float(r.value), "z": np.nan, "confidence": conf,
                         "outcome": "Sensor Suspect", "source": "Flatline rule", "basis": "Flatline rule",
                         "n_flagged": int(r.n_points), "peak_score": np.nan,
                         "predicted_fault": "Stuck Sensor", "cycle_id": getattr(r, "cycle_id", ""), "phase": getattr(r, "phase", "")})
        return recs

    for obj, g in out.groupby("object_id", sort=False):
        g = g.reset_index(drop=True)
        pos = np.flatnonzero(g["suspect_flat"].values)
        if not len(pos):
            continue
        obj_runs = flat_runs[flat_runs["object_id"] == obj]
        col = obj_runs["column"].iloc[0] if "column" in obj_runs.columns else fl.get("column", "Vibration (mm/s)")
        cat = obj_runs["category"].iloc[0] if "category" in obj_runs.columns else et["parameters"].get(col, {}).get("category", "Vibration")
        for i in pos:
            n_flt += 1
            row = g.iloc[i]
            val = float(row[col]) if col in row else float(obj_runs["value"].iloc[0])
            cycle_val = row["Cycle ID"] if "Cycle ID" in row and pd.notna(row["Cycle ID"]) else row.get("cycle_id", "")
            phase_val = row["Move Phase"] if "Move Phase" in row and pd.notna(row["Move Phase"]) else row.get("phase", "")
            recs.append({"equipment_type": name, "episode_id": f"{name}-FLT-{n_flt:05d}", "object_id": obj,
                         "category": cat, "parameter_column": col, "start": row["timestamp"],
                         "end": row["timestamp"], "value": val, "z": np.nan, "confidence": 99.9,
                         "outcome": "Sensor Suspect", "source": "Flatline rule", "basis": "Flatline rule",
                         "n_flagged": 1, "peak_score": np.nan,
                         "predicted_fault": "Stuck Sensor", "cycle_id": cycle_val, "phase": phase_val})
    return recs


def build(out, flat_runs, et, art, name):
    recs = model_episodes(out, et, art, name) + spike_episodes(out, et, art, name) + flatline_episodes(out, flat_runs, et, name)
    return pd.DataFrame(recs, columns=COLS)

