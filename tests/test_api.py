import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient  # noqa: E402
from anomaly.api import app  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_static_upload_returns_expected_json():
    from anomaly.output import FINAL_COLS_STATIC
    c = TestClient(app)
    with open(RAW / "Steady_State_Equipment_Data_Claude.xlsx", "rb") as f:
        r = c.post("/detect", params={"equipment_type": "static_rotating"}, files={"file": ("d.xlsx", f)})
    assert r.status_code == 200
    j = r.json()
    assert j["anomalies_found"] > 0
    assert list(j["results"][0]) == FINAL_COLS_STATIC
    assert "Data Confidence" in j["results"][0]
    assert j["results"][0]["Data Confidence"].endswith("%")
    assert "sensor_timeseries" in j
    st = j["sensor_timeseries"]
    assert st is not None
    assert "sensors" in st
    assert "assets" in st
    expected_sensors = ["Temperature (Degree Cel)", "Vibration (mm/s)", "Pressure (bars)", "Engine RPM"]
    for s in expected_sensors:
        assert s in st["sensors"], f"Expected sensor '{s}' in st['sensors']"
        sensor_data = st["sensors"][s]
        assert "name" in sensor_data
        assert "unit" in sensor_data
        assert "points" in sensor_data
        assert len(sensor_data["points"]) > 0
        p0 = sensor_data["points"][0]
        assert "t" in p0
        assert "y" in p0
        assert "anomaly" in p0
        assert "status" in p0
        assert "fault" in p0
        assert "asset" in p0

        # Verify NO healthy point exceeds limit thresholds
        lim_max = sensor_data.get("limit_max")
        lim_min = sensor_data.get("limit_min")
        for pt in sensor_data["points"]:
            if pt["status"] == "healthy":
                if lim_max is not None:
                    assert pt["y"] <= lim_max, f"Sensor {s} healthy point {pt['y']} > limit_max {lim_max} at {pt['t']}"
                if lim_min is not None:
                    assert pt["y"] >= lim_min, f"Sensor {s} healthy point {pt['y']} < limit_min {lim_min} at {pt['t']}"
            elif pt["status"] == "suspect":
                assert pt["anomaly"] == 2
            elif pt["status"] == "anomaly":
                assert pt["anomaly"] == 1

    # 100% Bidirectional Cross-Check Verification between table below and graph
    table_keys = set((r["Object ID"], r["Anomaly Start Date/Time"], r["Parameter Category"]) for r in j["results"])

    # Direction 1: Every table row MUST be represented on the graph
    for r in j["results"]:
        cat = r["Parameter Category"]
        col = next(c for c, m in st["sensors"].items() if m.get("category") == cat or cat in c)
        pts = st["sensors"][col]["points"]
        matches = [p for p in pts if p["asset"] == r["Object ID"] and p["t"] == r["Anomaly Start Date/Time"]]
        assert len(matches) > 0, f"Table record {r['Object ID']} at {r['Anomaly Start Date/Time']} for {cat} missing from graph"
        expected_status = "suspect" if r["Detection Outcome"] == "Sensor Suspect" else "anomaly"
        assert matches[0]["status"] == expected_status, f"Status mismatch: table={r['Detection Outcome']} vs graph={matches[0]['status']}"

    # Direction 2: Every anomaly/suspect on graph MUST be a legit record in the table below or a physical limit breach
    for s, s_data in st["sensors"].items():
        cat = s_data["category"]
        lim_max = s_data.get("limit_max")
        lim_min = s_data.get("limit_min")
        for pt in s_data["points"]:
            if pt["status"] != "healthy":
                key = (pt["asset"], pt["t"], cat)
                is_breach = (lim_max is not None and pt["y"] > lim_max) or (lim_min is not None and pt["y"] < lim_min)
                assert (key in table_keys) or is_breach, (
                    f"Graph shows {pt['status']} for {s} at {pt['t']} (asset {pt['asset']}), but it is not in the table below nor is it a limit breach!"
                )


def test_bad_file_returns_422():
    c = TestClient(app)
    import io, pandas as pd
    b = io.BytesIO(); pd.DataFrame({"x": [1]}).to_excel(b, index=False); b.seek(0)
    r = c.post("/detect", params={"equipment_type": "static_rotating"}, files={"file": ("d.xlsx", b)})
    assert r.status_code == 422


def test_export_excel_endpoint():
    c = TestClient(app)
    records = [{
        "Asset Category": "Rotating Equipment",
        "Sensor Device ID": "EQP-001_VIB",
        "Object ID": "EQP-001",
        "Site": "SITE-A",
        "Parameter Category": "Vibration",
        "Anomaly Start Date/Time": "2031-05-02 06:00:00",
        "Detected Value": 1.99,
        "Anomaly Fault": "Incipient Drift",
        "Detection Outcome": "Anomaly",
    }]
    r = c.post("/export-excel", json={"records": records, "filename": "Test_Data.xlsx"})
    assert r.status_code == 200
    assert "spreadsheetml.sheet" in r.headers["content-type"]
    assert "Test_Data_Anomalies.xlsx" in r.headers["content-disposition"]
    assert len(r.content) > 0


def test_export_excel_empty_returns_400():
    c = TestClient(app)
    r = c.post("/export-excel", json={"records": []})
    assert r.status_code == 400

