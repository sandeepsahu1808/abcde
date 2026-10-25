"""Step 3 - feature engineering (the only step that differs by data shape)."""
import pandas as pd
import numpy as np


def add_mahalanobis(df, healthy_mu, healthy_cov_inv, sensors):
    X = df[sensors].values
    diff = X - healthy_mu
    scores = np.array([
        np.sqrt(max(float(d @ healthy_cov_inv @ d), 0))
        for d in diff
    ])
    df['mahal_distance'] = scores
    return df


def build(df, et):
    """Returns X (features), C (context columns, may be empty), window_ok (row's history is all healthy)."""
    params = list(et["parameters"])
    ctx = et.get("context", [])
    C = df[ctx].copy() if ctx else pd.DataFrame(index=df.index)
    X = df[params].copy()

    # Fix 2: Interaction Features
    if "Temperature (Degree Cel)" in X.columns and "Engine RPM" in X.columns:
        X['temp_per_rpm'] = X['Temperature (Degree Cel)'] / X['Engine RPM'].clip(lower=1)
    if "Engine RPM" in X.columns and "Pressure (bars)" in X.columns:
        X['rpm_x_pressure'] = X['Engine RPM'] * X['Pressure (bars)']
    if "Vibration (mm/s)" in X.columns and "Pressure (bars)" in X.columns:
        X['vib_per_pressure'] = X['Vibration (mm/s)'] / X['Pressure (bars)'].clip(lower=0.1)

    if et["mode"] == "timeseries":
        # Static equipment: rolling windows for continuous machinery
        windows = et.get("rolling_windows", [et.get("rolling_window", 6)])
        g = df.groupby("object_id")
        for p in params:
            for w in windows:
                r = g[p].rolling(w, min_periods=1)
                mean_val = r.mean().reset_index(level=0, drop=True)
                X[f"{p} | mean{w}"] = mean_val
                X[f"{p} | std{w}"] = r.std().reset_index(level=0, drop=True).fillna(0.0)
                X[f"{p} | diff{w}"] = df[p] - mean_val
        max_w = max(windows)
        window_ok = (g["_healthy"].rolling(max_w, min_periods=1).min().reset_index(level=0, drop=True) == 1)
    else:
        # Cyclic machine: 1Hz SCADA telemetry with 9 move phases and physical interactions
        phase_col = "Move Phase" if "Move Phase" in df.columns else ("phase" if "phase" in df.columns else None)
        cycle_col = "Cycle ID" if "Cycle ID" in df.columns else ("cycle_id" if "cycle_id" in df.columns else None)
        phases_list = et.get("phases", [
            "HoistLoaded", "TrolleyLand", "LowerEmpty", "LowerLoaded", "TrolleyWater",
            "HoistEmpty", "Idle", "Lock", "Unlock"
        ])
        phase_df = pd.DataFrame(index=df.index)
        if phase_col is not None:
            for p in phases_list:
                phase_df[f"phase_{p}"] = (df[phase_col] == p).astype(float)
        
        # Domain physics & phase-conditioned interactions
        g = df.groupby("object_id")
        if cycle_col is not None:
            cycle_key = df["object_id"].astype(str) + "__" + df[cycle_col].astype(str)
            g_cycle = df.groupby(cycle_key)
        else:
            g_cycle = g

        # Asset-relative normalization for raw parameters:
        # Prevents model from memorizing machine-specific absolute numbers (e.g. 17A vs 207A)
        for p in params:
            med_p = g[p].transform("median")
            iqr_p = g[p].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
            iqr_p = iqr_p.fillna(1.0).replace(0.0, 1.0)
            X[p] = (df[p] - med_p) / iqr_p
        
        # 1. Drive & Load Physics (Bearing Wear, Motor health)
        loaded_moves = (phase_df.get("phase_HoistLoaded", 0.0) + phase_df.get("phase_LowerLoaded", 0.0))
        if "Hoist Motor Current (A)" in df.columns and "Hoist Load (t)" in df.columns:
            X["current_per_ton"] = (df["Hoist Motor Current (A)"] / df["Hoist Load (t)"].clip(lower=5.0)) * loaded_moves
        if "Total Power (kW)" in df.columns and "Hoist Load (t)" in df.columns:
            X["power_per_ton"] = (df["Total Power (kW)"] / df["Hoist Load (t)"].clip(lower=5.0)) * loaded_moves
            
        # 2. Kinematics & Encoder Integrity (Encoder Fault)
        # Note: Wire rope reeving ratio (2:1 or 4:1) varies across cranes, so hook speed != drum speed.
        # Encoder Fault is detected via within-cycle height jumps, jitter, and direction contradictions.
        if "Hoist Height (m)" in df.columns:
            h_diff = g_cycle["Hoist Height (m)"].diff().fillna(0.0)
            h_accel = g_cycle["Hoist Height (m)"].diff().diff().abs().fillna(0.0)
            X["height_diff_jump"] = (h_diff.abs() > 0.4).astype(float)
            X["height_jitter"] = h_accel
            if "Hoist Speed (m/min)" in df.columns:
                speed = df["Hoist Speed (m/min)"]
                speed_sign = np.sign(speed)
                diff_sign = np.sign(h_diff)
                X["encoder_glitch"] = ((speed.abs() > 3.0) & (h_diff.abs() > 0.02) & (speed_sign != diff_sign)).astype(float)
        if "Trolley Position (m)" in df.columns:
            t_diff = g_cycle["Trolley Position (m)"].diff().fillna(0.0)
            X["trolley_diff_jump"] = (t_diff.abs() > 0.5).astype(float)
            X["trolley_jitter"] = g_cycle["Trolley Position (m)"].diff().diff().abs().fillna(0.0)
        if "Hoist Height (m)" in df.columns and phase_col is not None:
            X["trolleyland_height"] = X["Hoist Height (m)"] * phase_df.get("phase_TrolleyLand", 0.0)
            
        # 3. Thermal & Cooling Correlation (Cooling Fault, Stuck Sensor)
        if "Hoist Motor Temp (C)" in df.columns:
            med_temp = g["Hoist Motor Temp (C)"].transform("median")
            X["temp_rise_from_crane_median"] = df["Hoist Motor Temp (C)"] - med_temp
            if "Ambient Temp (C)" in df.columns:
                X["temp_vs_ambient"] = df["Hoist Motor Temp (C)"] - df["Ambient Temp (C)"]
            cycle_std = g_cycle["Hoist Motor Temp (C)"].transform("std").fillna(0.0)
            X["temp_cycle_std"] = cycle_std
            X["temp_is_flat"] = (cycle_std < 1e-4).astype(float)
        if "Hoist Motor Temp (C)" in df.columns and phase_col is not None:
            X["idle_motor_temp"] = X["Hoist Motor Temp (C)"] * phase_df.get("phase_Idle", 0.0)
            
        # 4. Vibration & Oscillations (Rope/Sheave Fault, Bearing Wear)
        loaded_moves = (phase_df.get("phase_HoistLoaded", 0.0) + phase_df.get("phase_LowerLoaded", 0.0))
        if "Hoist Gearbox Vibration (mm/s)" in df.columns:
            med_vib = g["Hoist Gearbox Vibration (mm/s)"].transform("median").clip(lower=0.1)
            X["vibration_ratio"] = df["Hoist Gearbox Vibration (mm/s)"] / med_vib
            vib_diff = g_cycle["Hoist Gearbox Vibration (mm/s)"].diff().abs().fillna(0.0)
            X["hoist_vib_oscillation"] = (vib_diff / med_vib) * phase_df.get("phase_HoistLoaded", 0.0)
        if "Hoist Load (t)" in df.columns and "Hoist Motor Current (A)" in df.columns:
            med_hoist = g["Hoist Motor Current (A)"].transform(lambda s: s.abs().median()).clip(lower=1.0)
            load_ratio = df["Hoist Load (t)"] / (g["Hoist Load (t)"].transform("median").clip(lower=1.0))
            current_ratio = df["Hoist Motor Current (A)"].abs() / med_hoist
            X["loaded_slip_drop"] = (1.0 - load_ratio).clip(lower=0.0) * (1.0 - current_ratio).clip(lower=0.0) * loaded_moves
            if "Total Power (kW)" in df.columns:
                power_ratio = df["Total Power (kW)"].abs() / (g["Total Power (kW)"].transform(lambda s: s.abs().median()).clip(lower=1.0))
                X["rope_sheave_triple_drop"] = (1.0 - load_ratio).clip(lower=0.0) * (1.0 - current_ratio).clip(lower=0.0) * (1.0 - power_ratio).clip(lower=0.0) * loaded_moves
                
        # 5. Trolley Dynamics (Trolley Drive Fault)
        if "Trolley Current (A)" in df.columns:
            crane_trolley_peak = g["Trolley Current (A)"].transform(lambda s: s.abs().quantile(0.90)).clip(lower=10.0)
            trolley_active = (phase_df.get("phase_TrolleyWater", 0.0) + phase_df.get("phase_TrolleyLand", 0.0))
            X["trolley_current_ratio"] = (df["Trolley Current (A)"].abs() / crane_trolley_peak) * trolley_active
            X["trolley_current_diff"] = g_cycle["Trolley Current (A)"].diff().abs().fillna(0.0) / crane_trolley_peak
            
        # 6. Spreader Hydraulics (Hydraulic Leak)
        if "Spreader Hyd Pressure (bar)" in df.columns:
            crane_lock_pres = g["Spreader Hyd Pressure (bar)"].transform(lambda s: s.quantile(0.95)).clip(lower=10.0)
            lock_mask = (phase_df.get("phase_Lock", 0.0) + phase_df.get("phase_Unlock", 0.0))
            X["lock_pressure_deficit_ratio"] = ((crane_lock_pres - df["Spreader Hyd Pressure (bar)"]) / crane_lock_pres).clip(lower=0.0) * lock_mask
            X["hydraulic_pressure_ratio"] = df["Spreader Hyd Pressure (bar)"] / crane_lock_pres
            X["spreader_pressure_drop"] = ((crane_lock_pres - df["Spreader Hyd Pressure (bar)"]) / crane_lock_pres).clip(lower=0.0)
            
        # 7. Brake & Residual Torque (Brake Drag)
        if "Hoist Motor Current (A)" in df.columns and phase_col is not None:
            empty_moves = (phase_df.get("phase_HoistEmpty", 0.0) + phase_df.get("phase_LowerEmpty", 0.0))
            med_empty = g["Hoist Motor Current (A)"].transform(lambda s: s.abs().median()).clip(lower=1.0)
            X["empty_residual_current"] = (df["Hoist Motor Current (A)"].abs() / med_empty) * empty_moves
            if "Total Power (kW)" in df.columns:
                med_power = g["Total Power (kW)"].transform(lambda s: s.abs().median()).clip(lower=1.0)
                X["empty_power_surge"] = (df["Total Power (kW)"].abs() / med_power) * empty_moves

        if not phase_df.empty:
            X = pd.concat([X, phase_df], axis=1)

        X = X.fillna(0.0)
        window_ok = pd.Series(True, index=df.index)

    return X, C, window_ok

