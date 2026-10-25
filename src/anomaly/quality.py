"""Data-quality gate: runs BEFORE the model so sensor faults are not reported as machine faults."""
import numpy as np
import pandas as pd


def short_runs(mask, max_len):
    """Keep only runs of True that are at most max_len long."""
    out = np.zeros(len(mask), dtype=bool)
    pos = np.flatnonzero(mask)
    if not len(pos):
        return out
    for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
        if len(s) <= max_len:
            out[s] = True
    return out


def spike_scan(df, params, scale, k, max_len, causal=True):
    """Detect 1..max_len-row extreme transients on a single channel.

    deviation = |value - rolling median| / healthy std. A slow ramp (real fault)
    follows its own median so it is NOT a spike; a brief jump is.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with sensor columns and 'object_id'.
    params : list of str
        Sensor parameter columns to scan.
    scale : dict
        Standard deviations per parameter, fitted strictly on healthy training data.
    k : float
        Deviation threshold multiplier (e.g. 4.0 standard deviations).
    max_len : int
        Maximum consecutive points for a transient spike (e.g. 3 rows).
    causal : bool, default=True
        If True, use a causal trailing rolling median (s.shift(1).rolling(...)) so
        real-time streaming evaluation never looks ahead into future readings.
        Note on confirmation delay: In real-time streaming, confirming that an
        observed transient returns to baseline within <= max_len periods carries an
        intended detection confirmation delay of up to max_len periods (e.g. 3 time steps = 45 min).
        If False, uses centered rolling median (historical offline PoC behavior).

    Returns: spike flags, deviation ratios, rolling medians (all DataFrames).
    """
    win = 2 * max_len + 3
    spikes, devs, meds = {}, {}, {}
    for p in params:
        if causal:
            # Strictly causal: uses past win observations up to t-1 so the spike
            # at t does not pull the baseline median toward itself.
            med = df.groupby("object_id")[p].transform(
                lambda s: s.shift(1).rolling(win, min_periods=1).median().fillna(s)
            )
        else:
            med = df.groupby("object_id")[p].transform(
                lambda s: s.rolling(win, center=True, min_periods=1).median()
            )
        dev = (df[p] - med).abs() / scale[p]
        flag = pd.Series(False, index=df.index)
        for _, idx in df.groupby("object_id").indices.items():
            flag.iloc[idx] = short_runs((dev.iloc[idx] > k).values, max_len)
        spikes[p], devs[p], meds[p] = flag, dev, med
    return pd.DataFrame(spikes), pd.DataFrame(devs), pd.DataFrame(meds)


def apply_clean(df, params, spikes, meds):
    out = df.copy()
    for p in params:
        out[p] = np.where(spikes[p], meds[p], df[p])
    return out


def flatline_scan(tel, cfg):
    """Frozen-sensor rule: `window` identical consecutive readings (max-min == 0)."""
    window = cfg["window"]
    col, oc, tc = cfg["column"], cfg["object_col"], cfg["timestamp_col"]
    tel = tel.sort_values([oc, tc]).reset_index(drop=True)
    runs = []
    for obj, g in tel.groupby(oc):
        g = g.reset_index(drop=True)
        flat = ((g[col].rolling(window).max() - g[col].rolling(window).min()) == 0).values
        pos = np.flatnonzero(flat)
        if not len(pos):
            continue
        for s in np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1):
            first = max(s[0] - (window - 1), 0)
            runs.append({"object_id": obj, "start": g.loc[first, tc], "end": g.loc[s[-1], tc],
                         "value": float(g.loc[first, col]), "n_points": int(s[-1] - first + 1)})
    return pd.DataFrame(runs, columns=["object_id", "start", "end", "value", "n_points"])


def impute_nulls(df, params, null_policy=None, return_audit=False):
    """Phase 1: Two-tier causal null value imputation.

    Rules:
    1. Transient gaps (<= transient_limit consecutive NaNs, default 3):
       Forward-fill previous sensor reading (physical continuity / thermal inertia).
    2. Prolonged gaps (>= 4 consecutive NaNs):
       Impute using the asset's trailing rolling mean of previous data (shift(1).rolling(w).mean()),
       falling back to asset/parameter median if at cold start.
       Sets an explicit dropout flag so downstream rules identify sustained telemetry loss.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'object_id' and sensor columns.
    params : list of str
        Sensor parameter columns to scan and impute.
    null_policy : dict, optional
        Configuration with keys:
            transient_limit: int, default 3 (1-3 NaNs -> forward fill)
            trailing_window: int, default 24 (rolling window for trailing historical mean)
    return_audit : bool, default False
        If True, also returns the list of audit dictionaries.

    Returns
    -------
    clean_df : pd.DataFrame
        Clean DataFrame with nulls imputed (also has clean_df.attrs["null_audit"]).
    null_flags : pd.DataFrame
        DataFrame with boolean columns:
            f"imputed_short|{p}": True where 1-3 NaNs were forward filled
            f"imputed_long|{p}": True where >= 4 NaNs were filled with trailing mean
            f"dropout|{p}": True where >= 4 NaNs (sustained telemetry loss)
    audit_records : list of dict (only if return_audit=True)
        Detailed audit trail of all filled null values.
    """
    if null_policy is None:
        null_policy = {}
    transient_limit = null_policy.get("transient_limit", 3)
    trailing_window = null_policy.get("trailing_window", 24)

    out = df.copy()
    flags = {}
    for p in params:
        flags[f"imputed_short|{p}"] = pd.Series(False, index=df.index)
        flags[f"imputed_long|{p}"] = pd.Series(False, index=df.index)
        flags[f"dropout|{p}"] = pd.Series(False, index=df.index)

    # Check if any NaN exists in any param
    has_nans = False
    for p in params:
        if p in out.columns and out[p].isna().any():
            has_nans = True
            break

    audit_records = []
    if not has_nans:
        null_flags = pd.DataFrame(flags, index=df.index)
        out.attrs["null_audit"] = audit_records
        null_flags.attrs["audit_records"] = audit_records
        if return_audit:
            return out, null_flags, audit_records
        return out, null_flags

    # Group by object_id to ensure asset-isolated imputation
    for obj_id, idx in out.groupby("object_id").indices.items():
        sub_df = out.loc[idx]
        if "timestamp" in sub_df.columns:
            sub_df = sub_df.sort_values("timestamp")
        sub_indices = sub_df.index.values

        for p in params:
            if p not in sub_df.columns:
                continue
            s = sub_df[p]
            is_nan = s.isna()
            if not is_nan.any():
                continue

            blocks = (~is_nan).cumsum()
            nan_counts = is_nan.groupby(blocks).transform("sum") * is_nan

            short_gap = is_nan & (nan_counts <= transient_limit)
            long_gap = is_nan & (nan_counts > transient_limit)

            # Rule 1: Forward-fill previous sensor reading for transient gaps
            s_ffill = s.ffill()

            # Rule 2: Trailing rolling mean of previous data for prolonged gaps
            # Causal shift(1) ensures current gap does not use future or missing data
            trailing_mean = s.shift(1).rolling(trailing_window, min_periods=1).mean().ffill()

            # Cold-start fallback if series starts with NaNs
            valid_vals = s.dropna()
            if not valid_vals.empty:
                fallback_val = float(valid_vals.median())
            elif not out[p].dropna().empty:
                fallback_val = float(out[p].dropna().median())
            else:
                fallback_val = 0.0

            trailing_mean = trailing_mean.fillna(fallback_val)
            s_ffill = s_ffill.fillna(fallback_val)

            # Apply imputation
            imputed = s.copy()
            imputed[short_gap] = s_ffill[short_gap]
            imputed[long_gap] = trailing_mean[long_gap]
            imputed = imputed.fillna(fallback_val)

            # Record detailed audit for every imputed row
            for r_idx in sub_df.index[is_nan]:
                gap_len = int(nan_counts.loc[r_idx])
                imputed_val = float(imputed.loc[r_idx])
                is_short = bool(short_gap.loc[r_idx])
                is_long = bool(long_gap.loc[r_idx])

                if is_short:
                    if pd.notna(s.ffill().loc[r_idx]):
                        rule_desc = "Forward-Fill (Transient <= 3 NaNs)"
                    else:
                        rule_desc = "Cold-Start Asset Median (No Prior Value)"
                else:
                    if pd.notna(s.shift(1).rolling(trailing_window, min_periods=1).mean().ffill().loc[r_idx]):
                        rule_desc = f"Trailing Historical Mean (Prolonged >= 4 NaNs, W={trailing_window})"
                    else:
                        rule_desc = "Cold-Start Asset Median (No Prior History)"

                status = "Sensor Suspect: Telemetry Dropout" if is_long else "Normal Continuity"
                ts = sub_df.loc[r_idx, "timestamp"] if "timestamp" in sub_df.columns else None

                audit_records.append({
                    "object_id": obj_id,
                    "timestamp": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else str(ts),
                    "sensor_column": p,
                    "imputed_value": round(imputed_val, 3),
                    "imputation_rule": rule_desc,
                    "gap_length": gap_len,
                    "telemetry_status": status,
                })

            out.loc[sub_indices, p] = imputed.values
            flags[f"imputed_short|{p}"].loc[sub_indices] = short_gap.values
            flags[f"imputed_long|{p}"].loc[sub_indices] = long_gap.values
            flags[f"dropout|{p}"].loc[sub_indices] = long_gap.values

    null_flags = pd.DataFrame(flags, index=df.index)
    out.attrs["null_audit"] = audit_records
    null_flags.attrs["audit_records"] = audit_records

    if return_audit:
        return out, null_flags, audit_records
    return out, null_flags

