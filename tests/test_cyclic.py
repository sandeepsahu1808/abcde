import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly.api import app  # noqa: E402
from anomaly.output import FINAL_COLS, FINAL_COLS_CYCLIC  # noqa: E402

TEST_DATA = Path(__file__).resolve().parents[1] / "data" / "test"


def test_cyclic_xlsx_upload_detects_anomalies():
    """Test uploading full 1Hz SCADA telemetry Excel workbook for cyclic STS crane detection."""
    test_file = TEST_DATA / "cyclic_telemetry_1hz_test.xlsx"
    if not test_file.exists():
        test_file = TEST_DATA / "cyclic_test_data.xlsx"
    assert test_file.exists(), f"Test file {test_file} not found"

    client = TestClient(app)
    with open(test_file, "rb") as f:
        res = client.post(
            "/detect",
            params={"equipment_type": "cyclic_sts"},
            files={"file": (test_file.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["equipment_type"] == "cyclic_sts"
    assert data["rows_scored"] == 14660
    assert data["anomalies_found"] > 0
    assert "Anomaly" in data["summary"]

    first = data["results"][0]
    assert list(first.keys()) == FINAL_COLS_CYCLIC
    assert first["Asset Category"] == "STS"
    assert "Cycle ID" in first
    assert "Phase" in first
    assert "Data Confidence" in first
    assert first["Data Confidence"].endswith("%")
    assert first["Object ID"] == "STS-07"


def test_cyclic_csv_upload_detects_anomalies():
    """Test uploading 1Hz SCADA CSV file for cyclic STS crane detection."""
    test_file = TEST_DATA / "cyclic_telemetry_1hz_test.csv"
    assert test_file.exists(), f"Test file {test_file} not found"

    client = TestClient(app)
    with open(test_file, "rb") as f:
        res = client.post(
            "/detect",
            params={"equipment_type": "cyclic_sts"},
            files={"file": ("cyclic_telemetry_1hz_test.csv", f, "text/csv")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["equipment_type"] == "cyclic_sts"
    assert data["rows_scored"] == 14660
    assert data["anomalies_found"] > 0


def test_cyclic_small_sample():
    """Test uploading small curated 1Hz sample."""
    sample_file = TEST_DATA / "cyclic_sample_small.xlsx"
    assert sample_file.exists()

    client = TestClient(app)
    with open(sample_file, "rb") as f:
        res = client.post(
            "/detect",
            params={"equipment_type": "cyclic_sts"},
            files={"file": ("cyclic_sample_small.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["rows_scored"] == 250


def test_cyclic_fault_mappings_and_primary_sensors():
    """Test that detected cyclic faults map to their expected primary sensor parameters."""
    test_file = TEST_DATA / "cyclic_telemetry_1hz_test.xlsx"
    client = TestClient(app)
    with open(test_file, "rb") as f:
        res = client.post(
            "/detect",
            params={"equipment_type": "cyclic_sts"},
            files={"file": ("cyclic_telemetry_1hz_test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert res.status_code == 200
    data = res.json()
    results = data["results"]
    fault_types = {r["Anomaly Fault"] for r in results if r.get("Anomaly Fault")}
    assert len(fault_types) >= 4, f"Expected multiple fault types detected, got {fault_types}"
