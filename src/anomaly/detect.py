"""Step 5 - SCORE new data with the saved model + rules."""
import numpy as np
import pandas as pd
from . import features, quality, ingest
from .model import anomaly_score


def score(art, et, df, telemetry=None):
    params = list(et["parameters"])
    null_policy = et.get("null_policy", {"transient_limit": 3, "trailing_window": 24})
    df, null_flags = quality.impute_nulls(df, params, null_policy)
    out = df.copy()
    out.attrs["null_audit"] = df.attrs.get("null_audit", [])
    for col in null_flags.columns:
        out[col] = null_flags[col]

    # --- data-quality gate -------------------------------------------------
    if et.get("spike"):
        sp, dev, med = quality.spike_scan(
            df, params, art["scale"], et["spike"]["k"], et["spike"]["max_len"], causal=True
        )
        # Sensor Spike definition: short 1-3 hour extreme transient on a SINGLE channel.
        # It must breach that channel's limit, and NO OTHER channel breaches.
        for p, pc in et["parameters"].items():
            b_p = pd.Series(False, index=df.index)
            if pc.get("max") is not None:
                b_p |= df[p] > pc["max"]
            if pc.get("min") is not None:
                b_p |= df[p] < pc["min"]
            other_b = pd.Series(False, index=df.index)
            for q, qc in et["parameters"].items():
                if q != p:
                    if qc.get("max") is not None:
                        other_b |= df[q] > qc["max"]
                    if qc.get("min") is not None:
                        other_b |= df[q] < qc["min"]
            sp[p] = sp[p] & b_p & ~other_b
        clean = quality.apply_clean(df, params, sp, med)
    else:
        sp = pd.DataFrame(False, index=df.index, columns=params)
        dev = pd.DataFrame(0.0, index=df.index, columns=params)
        clean = df

    # --- model score -------------------------------------------------------
    feat_df = df if art.get("method") == "xgboost" else clean
    X, C, _ = features.build(feat_df, et)
    if art.get("method") != "xgboost" and "healthy_mu" in art and "healthy_cov_inv" in art and art["healthy_mu"] is not None:
        X = features.add_mahalanobis(X, art["healthy_mu"], art["healthy_cov_inv"], art["params"])
    Z = art["baseline"].residuals(X, C)
    score_result = anomaly_score(art["method"], art["iso"], Z, art["params"], X=X, return_proba=True)
    if len(score_result) == 4:
        out["score"] = score_result[0]
        out["predicted_fault"] = score_result[1] if score_result[1] is not None else "Unknown"
        model_proba = score_result[2]
        class_names = score_result[3]
    else:
        out["score"] = score_result[0]
        out["predicted_fault"] = score_result[1] if score_result[1] is not None else "Unknown"
        model_proba = None
        class_names = []
    out["predicted_fault_model"] = out["predicted_fault"].copy()
    is_normal = out["predicted_fault"] == "Normal"
    out["flag_model"] = (out["score"] >= art["threshold"]) & ~is_normal


    # --- hard-limit rules --------------------------------------------------
    breach_any = pd.Series(False, index=df.index)
    for p, pc in et["parameters"].items():
        out[f"z|{p}"] = Z[p]
        out[f"spike|{p}"] = sp[p]
        out[f"dev|{p}"] = dev[p]
        b = pd.Series(False, index=df.index)
        if pc.get("max") is not None:
            b |= clean[p] > pc["max"]
        if pc.get("min") is not None:
            b |= clean[p] < pc["min"]
        out[f"breach|{p}"] = b & ~sp[p]
        breach_any |= out[f"breach|{p}"]
    out["flag_rule"] = breach_any
    out["suspect_spike"] = sp.any(axis=1)

    # --- fault label overrides for static rotating equipment (MUST follow this order — never change) ------
    if et.get("mode") == "timeseries":
        # Core equipment channels for Progressive Cascade
        core_params = [p for p in ["Temperature (Degree Cel)", "Vibration (mm/s)", "Pressure (bars)"] if p in et["parameters"]]
        core_breach_count = sum(out[f"breach|{p}"] for p in core_params) if core_params else pd.Series(0, index=df.index)

        # 1. Hard-limit overrides first (Overheat / Bearing Degradation / Overpressure / Overspeed / Underspeed)
        if "Temperature (Degree Cel)" in et["parameters"]:
            out.loc[out["breach|Temperature (Degree Cel)"], "predicted_fault"] = "Overheat"
        if "Vibration (mm/s)" in et["parameters"]:
            out.loc[out["breach|Vibration (mm/s)"], "predicted_fault"] = "Bearing Degradation"
        if "Pressure (bars)" in et["parameters"]:
            out.loc[out["breach|Pressure (bars)"], "predicted_fault"] = "Overpressure"
        if "Engine RPM" in et["parameters"]:
            pc = et["parameters"]["Engine RPM"]
            if "max" in pc:
                out.loc[out["breach|Engine RPM"] & (clean["Engine RPM"] > pc["max"]), "predicted_fault"] = "Overspeed"
            if "min" in pc:
                out.loc[out["breach|Engine RPM"] & (clean["Engine RPM"] < pc["min"]), "predicted_fault"] = "Underspeed"

        # 2. Multi-parameter Cascade (>= 2 core parameters breached: Vib, Temp, Pres)
        if isinstance(core_breach_count, pd.Series):
            out.loc[core_breach_count >= 2, "predicted_fault"] = "Multi-parameter Cascade"

        # Invalidate false Multi-parameter Cascade if < 2 core limits breached
        fake_mpc = (out["predicted_fault"] == "Multi-parameter Cascade") & (core_breach_count < 2)
        if fake_mpc.any():
            for p in core_params:
                out.loc[fake_mpc & out[f"breach|{p}"], "predicted_fault"] = (
                    "Overheat" if "Temperature" in p else ("Bearing Degradation" if "Vibration" in p else "Overpressure")
                )

        # Non-breach subtle faults (all stay within thresholds)
        no_breach = (core_breach_count == 0) if isinstance(core_breach_count, pd.Series) else True
        if "breach|Engine RPM" in out.columns:
            no_breach &= ~out["breach|Engine RPM"]

        # Incipient Drift trend scan: Vibration and temperature creep upward over an episode (early bearing wear).
        # Never reach 5 / 95. The TREND is the fault.
        if "Vibration (mm/s)" in clean.columns and "Temperature (Degree Cel)" in clean.columns:
            for obj, g in clean.groupby("object_id"):
                dt_s = g["timestamp"].diff().median().total_seconds() if len(g) > 1 and "timestamp" in g.columns else 900.0
                dt_h = max(dt_s / 3600.0, 0.25) if pd.notna(dt_s) and dt_s > 0 else 0.25
                v = g["Vibration (mm/s)"]
                t = g["Temperature (Degree Cel)"]
                w = 5
                v_slope = (v - v.shift(w)) / (w * dt_h)
                t_slope = (t - t.shift(w)) / (w * dt_h)
                rpm_cond = (g["Engine RPM"].diff(w).abs() / (w * dt_h) < 30) if "Engine RPM" in g.columns else True
                p_cond = (g["Pressure (bars)"] < 2.4) if "Pressure (bars)" in g.columns else True
                cand = (v_slope >= 0.12) & (t_slope >= 0.45) & (v < 5.0) & (t < 95.0) & rpm_cond & p_cond & (no_breach.loc[g.index])
                pos = np.flatnonzero(cand.values)
                if len(pos):
                    for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
                        if len(s) >= 2:
                            first = max(0, s[0] - w)
                            drift_idx = [idx for idx in g.index[first:s[-1] + 1] if no_breach.loc[idx]]
                            if len(drift_idx):
                                out.loc[drift_idx, "flag_model"] = True
                                out.loc[drift_idx, "predicted_fault"] = "Incipient Drift"

        # 3. Under-threshold domain signatures (model-flagged, no breach)
        flagged_no_breach = (out["flag_model"] | fake_mpc) & no_breach

        # Efficiency Shift: Pressure persistently high (>=2.5) & Temp high (>=85.0) with no breach
        if "Pressure (bars)" in clean.columns and "Temperature (Degree Cel)" in clean.columns:
            is_eff = flagged_no_breach & (clean["Pressure (bars)"] >= 2.5) & (clean["Temperature (Degree Cel)"] >= 85.0)
            out.loc[is_eff, "predicted_fault"] = "Efficiency Shift"
            flagged_no_breach &= ~is_eff

        # Contextual Overheat: Temp ~90-94.5°C at low RPM/idle (<985) with no breach
        if "Temperature (Degree Cel)" in clean.columns and "Engine RPM" in clean.columns:
            is_co = flagged_no_breach & (clean["Temperature (Degree Cel)"] >= 90.0) & (clean["Temperature (Degree Cel)"] <= 94.5) & (clean["Engine RPM"] < 985)
            out.loc[is_co, "predicted_fault"] = "Contextual Overheat"
            flagged_no_breach &= ~is_co

        # Correlation Break: Vibration elevated (>=4.2 and <5.0) with no breach
        if "Vibration (mm/s)" in clean.columns:
            is_cb = flagged_no_breach & (clean["Vibration (mm/s)"] >= 4.2) & (clean["Vibration (mm/s)"] < 5.0)
            out.loc[is_cb, "predicted_fault"] = "Correlation Break"
            flagged_no_breach &= ~is_cb


    # --- frozen sensor rule -------------------------------------------------
    out["suspect_flat"] = False
    flat_runs = pd.DataFrame(columns=["object_id", "start", "end", "value", "n_points", "column", "category"])
    if et.get("flatline"):
        fl = et["flatline"]
        if telemetry is not None and "timestamp_col" in fl:
            tel = telemetry.copy()
            tel[fl["timestamp_col"]] = pd.to_datetime(tel[fl["timestamp_col"]])
            fr = quality.flatline_scan(tel, fl)
            fr["column"] = fl.get("column", "Sensor")
            fr["category"] = fl.get("category", "Temperature")
            flat_runs = fr
            nxt = out.groupby("object_id")["timestamp"].shift(-1).fillna(pd.Timestamp.max)
            for r in flat_runs.itertuples():
                hit = (out["object_id"] == r.object_id) & (out["timestamp"] <= r.end) & (nxt > r.start)
                out.loc[hit, "suspect_flat"] = True
        elif et.get("mode") == "timeseries":
            win = fl.get("window", 6)
            fr_list = []
            for p in params:
                for obj, g in clean.groupby("object_id"):
                    diff = (g[p].rolling(win).max() - g[p].rolling(win).min()) == 0
                    pos = np.flatnonzero(diff.values)
                    if not len(pos):
                        continue
                    for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
                        first = max(s[0] - (win - 1), 0)
                        start_ts = g.iloc[first]["timestamp"]
                        end_ts = g.iloc[s[-1]]["timestamp"]
                        hit_idx = g.index[first:s[-1] + 1]
                        out.loc[hit_idx, "suspect_flat"] = True
                        fr_list.append({
                            "object_id": obj,
                            "start": start_ts,
                            "end": end_ts,
                            "value": float(g.iloc[first][p]),
                            "n_points": int(s[-1] - first + 1),
                            "column": p,
                            "category": et["parameters"][p]["category"]
                        })
            if fr_list:
                flat_runs = pd.DataFrame(fr_list)

    out = out.copy()
    dropout_cols = [f"dropout|{p}" for p in params if f"dropout|{p}" in out.columns]
    out["suspect_dropout"] = out[dropout_cols].any(axis=1) if dropout_cols else pd.Series(False, index=df.index)
    if out["suspect_dropout"].any():
        for p in params:
            if f"dropout|{p}" in out.columns:
                drop_mask = out[f"dropout|{p}"] & ~breach_any
                out.loc[drop_mask, "predicted_fault"] = "Sensor Suspect: Telemetry Dropout"

    # Sustained breach is NOT a spike: if spike lasts >= 4 rows, treat as sustained machine breach
    for p in params:
        is_sp = out[f"spike|{p}"]
        if is_sp.any():
            blocks = (~is_sp).cumsum()
            sp_len = is_sp.groupby(blocks).transform("sum") * is_sp
            sustained = sp_len >= 4
            out.loc[sustained, f"spike|{p}"] = False
    out["suspect_spike"] = out[[f"spike|{p}" for p in params]].any(axis=1) if params else pd.Series(False, index=df.index)

    # Label sensor suspects
    if out["suspect_flat"].any():
        out.loc[out["suspect_flat"] & ~out["suspect_dropout"], "predicted_fault"] = "Stuck Sensor"
    if out["suspect_spike"].any():
        out.loc[out["suspect_spike"], "predicted_fault"] = "Sensor Spike"

    out["suspect_any"] = out["suspect_spike"] | out["suspect_flat"] | out["suspect_dropout"]

    # Rolling features warm-up only suppresses model score, never physical limit breaches
    enc_jump = pd.Series(False, index=df.index)
    if et["mode"] == "timeseries":
        windows = et.get("rolling_windows", [et.get("rolling_window", 6)])
        warm = out.groupby("object_id").cumcount() < max(windows) - 1
        model_flag = out["flag_model"] & ~warm
        out["flag_model"] &= ~warm
        out["flag"] = (model_flag & ~out["suspect_any"]) | out["flag_rule"]
    else:
        # Cyclic machine: cycle-level and physics-based domain signatures for reliable fault identification
        cycle_key = ["object_id", "cycle_id"] if ("cycle_id" in clean.columns and clean["cycle_id"].notna().any()) else ["object_id"]

        def _cyc_transform(s, agg_func):
            return s.groupby([clean[c] for c in cycle_key]).transform(agg_func)

        # Initialize cycle predictions
        cyc_fault = pd.Series("Normal", index=clean.index)
        cyc_flag = pd.Series(False, index=clean.index)

        # 1. Stuck Sensor: zero variance in temperature across cycle (< 1e-4)
        if "Hoist Motor Temp (C)" in clean.columns:
            temp_std = _cyc_transform(clean["Hoist Motor Temp (C)"], "std").fillna(0.0)
            stuck_mask = temp_std < 1e-4
            cyc_fault[stuck_mask] = "Stuck Sensor"
            cyc_flag[stuck_mask] = True
        elif "temp_is_flat" in X.columns and (X["temp_is_flat"] == 1.0).any():
            is_flat = X["temp_is_flat"] == 1.0
            cyc_fault[is_flat] = "Stuck Sensor"
            cyc_flag[is_flat] = True

        # 2. Hydraulic Leak: Spreader hydraulic pressure drops during Lock / Unlock phases (< 148 bar)
        phase_col = "Move Phase" if "Move Phase" in clean.columns else ("phase" if "phase" in clean.columns else None)
        if "Spreader Hyd Pressure (bar)" in clean.columns and phase_col:
            is_lock = clean[phase_col].isin(["Lock", "Unlock"])
            lock_p = clean["Spreader Hyd Pressure (bar)"].where(is_lock)
            min_lock_p = _cyc_transform(lock_p, "min").fillna(999.0)
            leak_mask = (min_lock_p < 148.0) & ~cyc_flag
            cyc_fault[leak_mask] = "Hydraulic Leak"
            cyc_flag[leak_mask] = True
        elif "spreader_pressure_drop" in X.columns and (X["spreader_pressure_drop"] > 0.23).any():
            leak_mask = (X["spreader_pressure_drop"] > 0.23) & ~cyc_flag
            cyc_fault[leak_mask] = "Hydraulic Leak"
            cyc_flag[leak_mask] = True

        # 3. Cooling Fault: motor temperature elevated above normal operating threshold (>68°C)
        if "Hoist Motor Temp (C)" in clean.columns:
            max_t = _cyc_transform(clean["Hoist Motor Temp (C)"], "max")
            cool_mask = (max_t > 68.0) & ~cyc_flag
            cyc_fault[cool_mask] = "Cooling Fault"
            cyc_flag[cool_mask] = True
        elif "temp_rise_from_crane_median" in X.columns and (X["temp_rise_from_crane_median"] > 9.0).any():
            cool_mask = (X["temp_rise_from_crane_median"] > 9.0) & ~cyc_flag
            cyc_fault[cool_mask] = "Cooling Fault"
            cyc_flag[cool_mask] = True

        # 4. Encoder Fault: step discontinuity or jump in hoist height (height change discrepancy > 0.3m/s beyond hoist speed)
        if "Hoist Height (m)" in clean.columns:
            h_step = clean.groupby(cycle_key)["Hoist Height (m)"].diff().abs().fillna(0.0)
            if "Hoist Speed (m/min)" in clean.columns:
                expected_step = (clean["Hoist Speed (m/min)"].abs() / 60.0) + 0.25
                discrepancy = (h_step - expected_step).clip(lower=0.0)
            else:
                discrepancy = (h_step - 0.4).clip(lower=0.0)
            max_disc = _cyc_transform(discrepancy, "max")
            enc_mask = (max_disc > 0.3) & ~cyc_flag
            cyc_fault[enc_mask] = "Encoder Fault"
            cyc_flag[enc_mask] = True
        elif "height_diff_jump" in X.columns:
            has_jump = (_cyc_transform(X["height_diff_jump"], "max") == 1.0) & ~cyc_flag
            cyc_fault[has_jump] = "Encoder Fault"
            cyc_flag[has_jump] = True

        # 5. Trolley Drive Fault: excessive trolley current surge (>350 A peak in cycle)
        if "Trolley Current (A)" in clean.columns:
            max_troll = _cyc_transform(clean["Trolley Current (A)"].abs(), "max")
            troll_mask = (max_troll > 350.0) & ~cyc_flag
            cyc_fault[troll_mask] = "Trolley Drive Fault"
            cyc_flag[troll_mask] = True
        elif "trolley_current_ratio" in X.columns and (X["trolley_current_ratio"] > 1.25).any():
            troll_mask = (X["trolley_current_ratio"] > 1.25) & ~cyc_flag
            cyc_fault[troll_mask] = "Trolley Drive Fault"
            cyc_flag[troll_mask] = True

        # 6. Brake Drag: elevated hoist motor current during empty moves (>180 A)
        if "Hoist Motor Current (A)" in clean.columns and phase_col:
            is_empty = clean[phase_col].isin(["HoistEmpty", "LowerEmpty"])
            empty_curr_series = clean["Hoist Motor Current (A)"].where(is_empty)
            empty_mean = _cyc_transform(empty_curr_series, "mean").fillna(0.0)
            brake_mask = (empty_mean > 180.0) & ~cyc_flag
            cyc_fault[brake_mask] = "Brake Drag"
            cyc_flag[brake_mask] = True
        elif "empty_residual_current" in X.columns and (X["empty_residual_current"] > 1.25).any():
            brake_mask = (X["empty_residual_current"] > 1.25) & ~cyc_flag
            cyc_fault[brake_mask] = "Brake Drag"
            cyc_flag[brake_mask] = True

        # 7. Mechanical Vibration: Bearing Wear vs Rope/Sheave Fault
        if "Hoist Gearbox Vibration (mm/s)" in clean.columns:
            max_vib = _cyc_transform(clean["Hoist Gearbox Vibration (mm/s)"], "max")
            vib_active = (max_vib > 3.5) & ~cyc_flag
            if vib_active.any():
                is_hl = clean[phase_col] == "HoistLoaded" if phase_col else pd.Series(True, index=clean.index)
                hl_s = clean["Hoist Gearbox Vibration (mm/s)"].where(is_hl)
                non_hl_s = clean["Hoist Gearbox Vibration (mm/s)"].where(~is_hl)
                hl_mean = _cyc_transform(hl_s, "mean").fillna(0.0)
                non_hl_mean = _cyc_transform(non_hl_s, "mean").fillna(0.0)
                contrast = hl_mean - non_hl_mean
                bear_mask = vib_active & ((contrast > 1.8) | (hl_mean > 5.7))
                rope_mask = vib_active & ~bear_mask
                cyc_fault[bear_mask] = "Bearing Wear"
                cyc_flag[bear_mask] = True
                cyc_fault[rope_mask] = "Rope/Sheave Fault"
                cyc_flag[rope_mask] = True

        # Apply cycle predictions
        out.loc[cyc_flag, "predicted_fault"] = cyc_fault[cyc_flag]
        out.loc[cyc_flag, "flag_model"] = True
        # Retain underlying classifier predictions for non-rule hits
        model_hit = (out["score"] >= art["threshold"]) & (out["predicted_fault"] != "Normal")
        out.loc[~cyc_flag, "flag_model"] = model_hit[~cyc_flag]

        model_flag = out["flag_model"]
        out["flag"] = model_flag | out["flag_rule"]

    # --- Data Confidence (% sure that the specific detected anomaly fault is present) ---
    confs_arr = np.full(len(df), 95.0, dtype=float)
    if model_proba is not None and len(class_names) > 0:
        norm_idx = class_names.index("Normal") if "Normal" in class_names else -1
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        pred_fault_series = out["predicted_fault"].values
        for i, fault in enumerate(pred_fault_series):
            if fault in class_to_idx:
                f_idx = class_to_idx[fault]
                p_raw = float(model_proba[i, f_idx])
                p_norm = float(model_proba[i, norm_idx]) if norm_idx >= 0 else 0.0
                p_cond = p_raw / max(1.0 - p_norm, 1e-4)
                c_val = max(p_raw, p_cond) * 100.0
                confs_arr[i] = min(max(c_val, 50.0), 99.9)
    confs = pd.Series(confs_arr, index=df.index, dtype=float)

    # Physical rule overrides and sensor suspects have deterministic high certainty
    if "flag_rule" in out.columns and out["flag_rule"].any():
        rule_conf = float(et.get("detection", {}).get("rule_confidence", 99.0))
        confs.loc[out["flag_rule"]] = np.maximum(confs.loc[out["flag_rule"]], rule_conf)
    if "suspect_flat" in out.columns and out["suspect_flat"].any():
        confs.loc[out["suspect_flat"]] = 99.9
    if "suspect_spike" in out.columns and out["suspect_spike"].any():
        confs.loc[out["suspect_spike"]] = 99.5
    if "suspect_dropout" in out.columns and out["suspect_dropout"].any():
        confs.loc[out["suspect_dropout"]] = 99.0
    if (out["predicted_fault"] == "Multi-parameter Cascade").any():
        confs.loc[out["predicted_fault"] == "Multi-parameter Cascade"] = 99.9
    if (out["predicted_fault"] == "Stuck Sensor").any():
        confs.loc[out["predicted_fault"] == "Stuck Sensor"] = 99.9
    if (out["predicted_fault"] == "Hydraulic Leak").any():
        confs.loc[out["predicted_fault"] == "Hydraulic Leak"] = 99.0
    if enc_jump.any():
        confs.loc[enc_jump] = 99.9

    out["confidence"] = confs.round(1)
    return out, flat_runs

