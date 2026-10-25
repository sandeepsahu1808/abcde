import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from anomaly import config, evaluate, features, ingest, metrics_logger, model, quality


def test_split_completeness_and_disjoint():
    """Verify that every row is assigned to exactly one of train, calib, or test with no overlap."""
    n_rows = 1000
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * (n_rows // 2) + ["EQP-002"] * (n_rows // 2),
        "timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15min"),
        "_healthy": [True] * 600 + [False] * 400,
        "_labelled": [True] * n_rows,
        "_fault": ["Normal"] * 600 + ["Overheat"] * 400,
    })
    split_cfg = {"train": 0.7, "calib": 0.15, "block": 24}
    eligible = df["_labelled"].values

    split = model.assign_split(df, eligible, split_cfg)

    # 1. Completeness: exactly 100% of rows assigned to train, calib, or test
    assert len(split) == n_rows
    assert set(split.unique()).issubset({"train", "calib", "test"})
    assert (split == "excluded").sum() == 0
    assert (split == "unhealthy").sum() == 0

    # 2. Disjoint sets: no index overlaps
    idx_tr = set(df.index[split == "train"])
    idx_ca = set(df.index[split == "calib"])
    idx_te = set(df.index[split == "test"])

    assert len(idx_tr & idx_ca) == 0, "Train and Calib must not overlap"
    assert len(idx_tr & idx_te) == 0, "Train and Test must not overlap"
    assert len(idx_ca & idx_te) == 0, "Calib and Test must not overlap"
    assert len(idx_tr | idx_ca | idx_te) == n_rows


def test_no_fault_episodes_cut_across_splits():
    """Verify that fault episodes are never sliced across split boundaries."""
    n_rows = 600
    # Create alternating fault runs of varying lengths (e.g. 5 to 15 rows)
    fault_flags = []
    fault_names = []
    while len(fault_flags) < n_rows:
        fault_flags.extend([True] * 20)  # 20 healthy
        fault_names.extend(["Normal"] * 20)
        fault_flags.extend([False] * 10) # 10 faulty
        fault_names.extend(["Bearing Degradation"] * 10)
    fault_flags = fault_flags[:n_rows]
    fault_names = fault_names[:n_rows]

    df = pd.DataFrame({
        "object_id": ["EQP-001"] * n_rows,
        "timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15min"),
        "_healthy": fault_flags,
        "_labelled": [True] * n_rows,
        "_fault": fault_names,
    })
    split_cfg = {"train": 0.70, "calib": 0.15, "block": 24}
    split = model.assign_split(df, df["_labelled"].values, split_cfg)

    # Contiguous runs
    df["split"] = split
    df["run_id"] = ((df["_fault"] != df["_fault"].shift(1)) | (df["object_id"] != df["object_id"].shift(1))).cumsum()
    fault_runs = df[~df["_healthy"]].groupby("run_id")["split"].nunique()
    cut_runs = (fault_runs > 1).sum()

    assert cut_runs == 0, f"Expected 0 fault episodes cut across splits, found {cut_runs}"


def test_validation_split_inside_training_data():
    """Verify that create_validation_split carves an episode-aware validation partition strictly inside train."""
    n_rows = 800
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * n_rows,
        "timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15min"),
        "_healthy": [True] * 500 + [False] * 300,
        "_labelled": [True] * n_rows,
        "_fault": ["Normal"] * 500 + ["Overheat"] * 300,
    })
    split = model.assign_split(df, df["_labelled"].values, {"train": 0.70, "calib": 0.15})
    tr_mask = (split == "train").values

    val_mask = model.create_validation_split(df, tr_mask, val_ratio=0.20)
    tr_fit = tr_mask & ~val_mask

    # 1. Validation split is strictly inside training data
    assert val_mask.sum() > 0
    assert (val_mask & ~tr_mask).sum() == 0, "Validation rows must be a subset of training data"
    assert (tr_fit & val_mask).sum() == 0, "Train fit and Validation must be disjoint"
    assert (tr_fit.sum() + val_mask.sum()) == tr_mask.sum()

    # 2. Episode integrity: no fault run cut between train_fit and val
    df["subsplit"] = "other"
    df.loc[tr_fit, "subsplit"] = "train_fit"
    df.loc[val_mask, "subsplit"] = "val"
    df["run_id"] = ((df["_fault"] != df["_fault"].shift(1)) | (df["object_id"] != df["object_id"].shift(1))).cumsum()
    fault_runs = df[tr_mask & ~df["_healthy"]].groupby("run_id")["subsplit"].nunique()
    assert (fault_runs > 1).sum() == 0, "No fault episode may be sliced between train_fit and val"


def test_train_only_statistics_and_preprocessing():
    """Verify that sensor scaling and baseline are fitted strictly on healthy training data only."""
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 100,
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="15min"),
        "Temperature (Degree Cel)": [70.0] * 70 + [90.0] * 30,  # Train temp is 70, Test temp is 90
        "Vibration (mm/s)": [1.5] * 100,
        "Pressure (bars)": [2.0] * 100,
        "Engine RPM": [1000] * 100,
        "_healthy": [True] * 100,
        "_labelled": [True] * 100,
        "_fault": ["Normal"] * 100,
    })
    tr_fit = np.array([True] * 70 + [False] * 30)

    # Scale fitted strictly on tr_fit
    scale_train_only = df.loc[tr_fit, ["Temperature (Degree Cel)"]].std(ddof=0).clip(lower=1e-9).to_dict()
    scale_full_leaked = df[["Temperature (Degree Cel)"]].std(ddof=0).clip(lower=1e-9).to_dict()

    # Train-only standard deviation should be near 0 (flat 70.0), whereas leaked std is ~9.16
    assert scale_train_only["Temperature (Degree Cel)"] < 1e-6
    assert scale_full_leaked["Temperature (Degree Cel)"] > 5.0


def test_causal_spike_detection_no_future_leakage():
    """Verify that causal spike detection never looks ahead into future readings."""
    params = ["Temperature (Degree Cel)"]
    scale = {"Temperature (Degree Cel)": 5.0}

    # Baseline series
    temps_a = [75.0, 75.2, 75.1, 75.3, 115.5, 75.0, 75.2, 75.1, 75.0, 75.2]
    # Series with modified future points at index 8 and 9
    temps_b = [75.0, 75.2, 75.1, 75.3, 115.5, 75.0, 75.2, 75.1, 150.0, 160.0]

    df_a = pd.DataFrame({
        "object_id": ["EQP-001"] * 10,
        "Temperature (Degree Cel)": temps_a,
    })
    df_b = pd.DataFrame({
        "object_id": ["EQP-001"] * 10,
        "Temperature (Degree Cel)": temps_b,
    })

    # Causal scan on both
    sp_a, dev_a, med_a = quality.spike_scan(df_a, params, scale, k=4.0, max_len=3, causal=True)
    sp_b, dev_b, med_b = quality.spike_scan(df_b, params, scale, k=4.0, max_len=3, causal=True)

    # In causal mode, index 0..7 must have IDENTICAL medians and spike flags regardless of future values at 8..9
    assert np.allclose(med_a.iloc[:8, 0], med_b.iloc[:8, 0]), "Causal median at t <= 7 must not depend on future points at t >= 8"
    assert (sp_a.iloc[:8, 0] == sp_b.iloc[:8, 0]).all(), "Causal spike flags at t <= 7 must not depend on future points at t >= 8"


def test_prohibit_evaluation_fallback():
    """Verify that evaluation fails clearly with ValueError when no holdout test split is found."""
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 50,
        "timestamp": pd.date_range("2025-01-01", periods=50, freq="15min"),
        "_healthy": [True] * 50,
        "_labelled": [True] * 50,
        "_fault": ["Normal"] * 50,
        "score": [0.1] * 50,
        "flag": [False] * 50,
        "flag_model": [False] * 50,
        "flag_rule": [False] * 50,
        "suspect_any": [False] * 50,
    })
    # Artefact without split_keys
    art_no_split = {
        "method": "max_z",
        "params": ["Temperature (Degree Cel)"],
        "threshold": 0.5,
    }
    with pytest.raises(ValueError, match="No hold-out test split found|missing split_keys"):
        evaluate.evaluate(df, art_no_split, "test_eqp")

    # Artefact with split_keys containing only train and calib (no test)
    split_keys_no_test = pd.DataFrame({
        "object_id": ["EQP-001"] * 50,
        "timestamp": df["timestamp"],
        "split": ["train"] * 40 + ["calib"] * 10,
    })
    art_with_no_test_split = {
        "method": "max_z",
        "params": ["Temperature (Degree Cel)"],
        "threshold": 0.5,
        "split_keys": split_keys_no_test,
    }
    with pytest.raises(ValueError, match="No hold-out test split found"):
        evaluate.evaluate(df, art_with_no_test_split, "test_eqp")

    with pytest.raises(ValueError, match="No hold-out test split found"):
        metrics_logger._compute_performance("test_eqp", art_with_no_test_split, df, split_keys_no_test)


def test_classifier_vs_end_to_end_metrics_separation():
    """Verify that evaluate.evaluate reports classifier-level metrics separately from end-to-end system metrics."""
    n = 100
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * n,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "_healthy": [True] * 70 + [False] * 30,
        "_labelled": [True] * n,
        "_fault": ["Normal"] * 70 + ["Overheat"] * 30,
        "score": [0.1] * 70 + [0.8] * 20 + [0.2] * 10, # Model catches 20/30 faults
        "flag_model": [False] * 70 + [True] * 20 + [False] * 10,
        "flag_rule": [False] * 90 + [True] * 10,       # Rule catches remaining 10 faults
        "flag": [False] * 70 + [True] * 30,
        "suspect_any": [False] * n,
        "predicted_fault_model": ["Normal"] * 70 + ["Overheat"] * 20 + ["Incipient Drift"] * 10,
        "predicted_fault": ["Normal"] * 70 + ["Overheat"] * 30, # Overridden to Overheat
    })
    split_keys = pd.DataFrame({
        "object_id": ["EQP-001"] * n,
        "timestamp": df["timestamp"],
        "split": ["train"] * 50 + ["test"] * 50,  # 50 test rows: 20 healthy, 30 faulty
    })
    art = {
        "method": "xgboost",
        "params": ["Temperature (Degree Cel)"],
        "threshold": 0.5,
        "split_keys": split_keys,
    }

    summary, by_fault = evaluate.evaluate(df, art, "test_eqp")
    summ_dict = dict(zip(summary["metric"], summary["value"]))

    # Classifier alone caught 20/30 faults -> recall = 0.6667
    assert "[Classifier-Only] Row recall on hold-out test rows" in summ_dict
    assert abs(summ_dict["[Classifier-Only] Row recall on hold-out test rows"] - 0.6667) < 0.01

    # End-to-end system caught 30/30 faults -> recall = 1.0000
    assert "[End-to-End] Row recall on hold-out test rows" in summ_dict
    assert abs(summ_dict["[End-to-End] Row recall on hold-out test rows"] - 1.0000) < 0.01

    # Alarm attribution and override counts tracked
    assert summ_dict["[Alarms] Model-only alarms in test split"] == 20
    assert summ_dict["[Alarms] Rule-only breach alarms in test split"] == 10
    assert summ_dict["[Overrides] Rule overrides applied to classifier predictions"] == 10


def test_spike_signal_preservation_in_features():
    """Verify that features.build on df preserves raw extreme spike values."""
    et = {
        "mode": "timeseries",
        "parameters": {
            "Temperature (Degree Cel)": {"category": "Temperature", "max": 95},
            "Vibration (mm/s)": {"category": "Vibration", "max": 5},
            "Pressure (bars)": {"category": "Pressure", "max": 3},
            "Engine RPM": {"category": "RPM", "min": 900, "max": 1200},
        },
        "rolling_windows": [6],
    }

    raw_df = pd.DataFrame({
        "object_id": ["EQP-001"] * 10,
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
        "Temperature (Degree Cel)": [75.0, 75.2, 75.1, 75.3, 115.5, 75.0, 75.2, 75.1, 75.0, 75.2],  # Index 4 is a spike
        "Vibration (mm/s)": [1.8] * 10,
        "Pressure (bars)": [2.0] * 10,
        "Engine RPM": [1000] * 10,
        "_healthy": [True] * 4 + [False] + [True] * 5,
    })

    # When features.build is called on raw_df, the spike value 115.5 must NOT be flattened
    X, _, _ = features.build(raw_df, et)
    assert X.loc[4, "Temperature (Degree Cel)"] == 115.5, "Spike value must be preserved in feature matrix"
    assert X.loc[4, "Temperature (Degree Cel) | diff6"] > 30.0, "Spike residual difference must be prominent"


def test_dataset_split_proportions_on_combined_data():
    """Verify that on the combined dataset, exactly 234,000 rows are partitioned with 0 sliced fault runs."""
    cfg = config.load_config()
    et = cfg["equipment_types"]["static_rotating"]
    if isinstance(et["source"].get("files"), list) and len(et["source"]["files"]) == 2:
        df = ingest.load_table(et)
        assert len(df) == 234000, f"Expected 234,000 combined rows, got {len(df)}"

        eligible = df["_labelled"].values
        split = model.assign_split(df, eligible, et["split"])

        counts = split.value_counts().to_dict()
        total_assigned = counts.get("train", 0) + counts.get("calib", 0) + counts.get("test", 0)
        assert total_assigned == 234000, f"Expected exactly 234,000 rows assigned, got {total_assigned}"

        # Train ~70% (163.8k), Calib ~15% (35.1k), Test ~15% (35.1k)
        assert 163000 <= counts.get("train", 0) <= 164500
        assert 34800 <= counts.get("calib", 0) <= 35300
        assert 34800 <= counts.get("test", 0) <= 35300

        # Verify episode integrity: zero fault runs sliced across splits
        df["split"] = split
        df["run_id"] = ((df["_fault"] != df["_fault"].shift(1)) | (df["object_id"] != df["object_id"].shift(1))).cumsum()
        fault_runs = df[~df["_healthy"]].groupby("run_id")["split"].nunique()
        cut_runs = (fault_runs > 1).sum()
        assert cut_runs == 0, f"Expected 0 fault runs cut across splits, found {cut_runs}"
