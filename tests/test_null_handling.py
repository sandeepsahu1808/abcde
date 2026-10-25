import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from anomaly import config, detect, model, quality


def test_transient_nan_forward_fill():
    """Verify that 1-3 consecutive NaNs are forward-filled from the previous sensor reading."""
    params = ["Temperature (Degree Cel)"]
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 10,
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
        "Temperature (Degree Cel)": [80.0, 81.0, np.nan, np.nan, 82.0, 83.0, np.nan, 84.0, 85.0, 86.0],
    })

    policy = {"transient_limit": 3, "trailing_window": 24}
    clean_df, flags = quality.impute_nulls(df, params, policy)

    # Indices 2 and 3 should be forward-filled from index 1 (81.0)
    assert clean_df.loc[2, "Temperature (Degree Cel)"] == 81.0
    assert clean_df.loc[3, "Temperature (Degree Cel)"] == 81.0
    # Index 6 should be forward-filled from index 5 (83.0)
    assert clean_df.loc[6, "Temperature (Degree Cel)"] == 83.0

    # Zero NaNs remaining
    assert not clean_df["Temperature (Degree Cel)"].isna().any()

    # Flags
    p = "Temperature (Degree Cel)"
    assert flags.loc[2, f"imputed_short|{p}"]
    assert flags.loc[3, f"imputed_short|{p}"]
    assert flags.loc[6, f"imputed_short|{p}"]
    assert not flags.loc[4, f"imputed_short|{p}"]
    assert not flags[f"imputed_long|{p}"].any()
    assert not flags[f"dropout|{p}"].any()


def test_prolonged_nan_trailing_mean_and_dropout_flag():
    """Verify that >= 4 consecutive NaNs are imputed with trailing historical mean and flagged as dropout."""
    params = ["Pressure (bars)"]
    # 5 valid values (average 2.0), followed by 5 consecutive NaNs (outage)
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 10,
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
        "Pressure (bars)": [2.0, 2.0, 2.0, 2.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    })

    policy = {"transient_limit": 3, "trailing_window": 24}
    clean_df, flags = quality.impute_nulls(df, params, policy)

    p = "Pressure (bars)"
    # Gaps at indices 5..9 should be filled with trailing mean (2.0)
    for idx in range(5, 10):
        assert abs(clean_df.loc[idx, p] - 2.0) < 1e-6
        assert flags.loc[idx, f"imputed_long|{p}"]
        assert flags.loc[idx, f"dropout|{p}"]
        assert not flags.loc[idx, f"imputed_short|{p}"]

    assert not clean_df[p].isna().any()


def test_leading_nans_cold_start_fallback():
    """Verify that series starting with NaNs are imputed without errors or future leakage."""
    params = ["Vibration (mm/s)"]
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 6,
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="15min"),
        "Vibration (mm/s)": [np.nan, np.nan, 1.8, 1.9, 2.0, 2.1],
    })

    policy = {"transient_limit": 3, "trailing_window": 24}
    clean_df, flags = quality.impute_nulls(df, params, policy)

    p = "Vibration (mm/s)"
    assert not clean_df[p].isna().any()
    # Leading NaNs fallback to median of available valid readings (1.95)
    assert clean_df.loc[0, p] > 1.0
    assert clean_df.loc[1, p] > 1.0


def test_multiple_assets_independent_null_imputation():
    """Verify that trailing means and forward fills are strictly isolated per asset."""
    params = ["Engine RPM"]
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * 6 + ["EQP-002"] * 6,
        "timestamp": list(pd.date_range("2025-01-01", periods=6, freq="15min")) * 2,
        # EQP-001 runs at 1000 RPM; EQP-002 runs at 500 RPM
        "Engine RPM": [1000.0, 1000.0, np.nan, np.nan, np.nan, np.nan,
                       500.0, 500.0, np.nan, np.nan, np.nan, np.nan],
    })

    policy = {"transient_limit": 3, "trailing_window": 24}
    clean_df, _ = quality.impute_nulls(df, params, policy)

    p = "Engine RPM"
    # EQP-001 prolonged gap (indices 2..5) must impute around 1000 RPM
    for idx in range(2, 6):
        assert abs(clean_df.loc[idx, p] - 1000.0) < 1e-6

    # EQP-002 prolonged gap (indices 8..11) must impute around 500 RPM (never 1000 RPM)
    for idx in range(8, 12):
        assert abs(clean_df.loc[idx, p] - 500.0) < 1e-6


def test_end_to_end_scoring_with_injected_nulls():
    """Verify that detect.score processes raw telemetry with NaNs and tags sustained dropouts as Sensor Suspect."""
    cfg = config.load_config()
    et = cfg["equipment_types"]["static_rotating"]
    art = model.load("static_rotating")

    # Construct test sample with healthy baseline
    n = 100
    df = pd.DataFrame({
        "object_id": ["EQP-001"] * n,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "Temperature (Degree Cel)": [75.0] * n,
        "Vibration (mm/s)": [1.5] * n,
        "Pressure (bars)": [2.0] * n,
        "Engine RPM": [1000.0] * n,
        "_healthy": [True] * n,
        "_labelled": [True] * n,
        "_fault": ["Normal"] * n,
    })

    # Inject a 2-step transient drop on Pressure (rows 30..31)
    df.loc[30:31, "Pressure (bars)"] = np.nan
    # Inject a 6-step sustained outage on Temperature (rows 60..65)
    df.loc[60:65, "Temperature (Degree Cel)"] = np.nan

    out, flat = detect.score(art, et, df)

    # 1. Scoring must complete with zero NaNs in output scores or residuals
    assert not out["score"].isna().any()
    assert not out["z|Temperature (Degree Cel)"].isna().any()
    assert not out["z|Pressure (bars)"].isna().any()

    # 2. Transient drop at rows 30..31 must be forward-filled and NOT flagged as dropout
    assert not out.loc[30, "suspect_dropout"]
    assert not out.loc[31, "suspect_dropout"]
    assert out.loc[30, "imputed_short|Pressure (bars)"]
    assert out.loc[31, "imputed_short|Pressure (bars)"]

    # 3. Sustained outage at rows 60..65 must be flagged as suspect_dropout
    for r in range(60, 66):
        assert out.loc[r, "suspect_dropout"]
        assert out.loc[r, "suspect_any"]
        # Machine anomaly flag must be suppressed for sensor dropout rows
        assert not out.loc[r, "flag"]
        assert out.loc[r, "predicted_fault"] == "Sensor Suspect: Telemetry Dropout"

    # 4. Null audit trail verification
    null_audit = out.attrs.get("null_audit", [])
    assert len(null_audit) == 8  # 2 rows for Pressure, 6 rows for Temperature
    # Check Pressure transient records
    press_records = [r for r in null_audit if r["sensor_column"] == "Pressure (bars)"]
    assert len(press_records) == 2
    for rec in press_records:
        assert rec["imputed_value"] == 2.0
        assert "Forward-Fill" in rec["imputation_rule"]
        assert rec["gap_length"] == 2
        assert rec["telemetry_status"] == "Normal Continuity"

    # Check Temperature sustained records
    temp_records = [r for r in null_audit if r["sensor_column"] == "Temperature (Degree Cel)"]
    assert len(temp_records) == 6
    for rec in temp_records:
        assert abs(rec["imputed_value"] - 75.0) < 1e-3
        assert "Trailing Historical Mean" in rec["imputation_rule"]
        assert rec["gap_length"] == 6
        assert rec["telemetry_status"] == "Sensor Suspect: Telemetry Dropout"


def test_null_audit_excel_worksheet_export():
    """Verify that export_to_excel_buffer generates the 'Null Value Audit' worksheet with proper headers and data."""
    import openpyxl
    from anomaly import output

    records = [{
        "Asset Category": "Rotating Equipment",
        "Sensor Device ID": "SNS-EQP-001-TEMPERATURE",
        "Object ID": "EQP-001",
        "Site": "SITE-01",
        "Parameter Category": "Temperature",
        "Anomaly Start Date/Time": "2025-01-01 06:00:00",
        "Detected Value": 98.5,
        "Anomaly Fault": "Overheat",
        "Detection Outcome": "Anomaly",
    }]

    null_imputations = [{
        "object_id": "EQP-001",
        "timestamp": "2025-01-01 04:00:00",
        "sensor_column": "Temperature (Degree Cel)",
        "imputed_value": 75.25,
        "imputation_rule": "Forward-Fill (Transient <= 3 NaNs)",
        "gap_length": 2,
        "telemetry_status": "Normal Continuity",
    }, {
        "object_id": "EQP-001",
        "timestamp": "2025-01-01 04:15:00",
        "sensor_column": "Temperature (Degree Cel)",
        "imputed_value": 75.25,
        "imputation_rule": "Forward-Fill (Transient <= 3 NaNs)",
        "gap_length": 2,
        "telemetry_status": "Normal Continuity",
    }]

    buf = output.export_to_excel_buffer(
        records=records,
        equipment_type="static_rotating",
        null_imputations=null_imputations,
    )

    wb = openpyxl.load_workbook(buf)
    assert "All Anomalies" in wb.sheetnames
    assert "Fault Summary" in wb.sheetnames
    assert "Asset Summary" in wb.sheetnames
    assert "Null Value Audit" in wb.sheetnames

    ws_audit = wb["Null Value Audit"]
    headers = [cell.value for cell in ws_audit[1]]
    expected_headers = [
        "Asset Category", "Sensor Device ID", "Object ID", "Site", "Parameter Category",
        "Sensor Column", "Timestamp", "Imputed Value", "Imputation Rule", "Consecutive Null Count", "Telemetry Status"
    ]
    assert headers == expected_headers
    # Check data rows: header + 2 rows = 3 rows total
    assert ws_audit.max_row == 3
    row2 = [cell.value for cell in ws_audit[2]]
    assert row2[2] == "EQP-001"
    assert row2[5] == "Temperature (Degree Cel)"
    assert row2[7] == 75.25


def test_clean_telemetry_null_audit_sheet():
    """Verify that when telemetry has 0 nulls, 'Null Value Audit' sheet is still present with a clean data note."""
    import openpyxl
    from anomaly import output

    records = [{
        "Asset Category": "Rotating Equipment",
        "Sensor Device ID": "SNS-EQP-001-TEMPERATURE",
        "Object ID": "EQP-001",
        "Site": "SITE-01",
        "Parameter Category": "Temperature",
        "Anomaly Start Date/Time": "2025-01-01 06:00:00",
        "Detected Value": 98.5,
        "Anomaly Fault": "Overheat",
        "Detection Outcome": "Anomaly",
    }]

    buf = output.export_to_excel_buffer(records=records, equipment_type="static_rotating", null_imputations=[])
    wb = openpyxl.load_workbook(buf)
    assert "Null Value Audit" in wb.sheetnames
    ws_audit = wb["Null Value Audit"]
    assert ws_audit.max_row == 2
    row2 = [cell.value for cell in ws_audit[2]]
    assert "No missing or null values" in row2[8]
    assert row2[10] == "Clean Data"

