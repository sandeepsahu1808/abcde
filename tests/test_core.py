import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly import detect, episodes, model, output, quality  # noqa: E402
from anomaly.config import load_config  # noqa: E402


def test_short_runs_keeps_only_short():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert quality.short_runs(m, 3).tolist() == [False, True, True, False, False, False, False, False, False]


def test_ecdf_conf_is_capped_and_monotone():
    ref = np.arange(100.0)
    assert episodes.ecdf_conf(1000, ref) == 99.9
    assert episodes.ecdf_conf(10, ref) < episodes.ecdf_conf(50, ref)


def _synthetic(n=720, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="h")
    load = 0.5 + 0.3 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    df = pd.DataFrame({"object_id": "T-1", "timestamp": t,
                       "Temperature (Degree Cel)": 80 + 10 * load + rng.normal(0, .5, n),
                       "Vibration (mm/s)": 2 + 1.5 * load + rng.normal(0, .1, n),
                       "Pressure (bars)": 1.5 + load + rng.normal(0, .05, n),
                       "Engine RPM": 950 + 200 * load + rng.normal(0, 5, n)})
    df["_fault"], df["_healthy"], df["_labelled"] = "Normal", True, True
    df.loc[600:640, "Temperature (Degree Cel)"] += np.linspace(0, 30, 41)  # injected overheat ramp
    df.loc[600:640, "_fault"], df.loc[600:640, "_healthy"] = "Overheat", False
    return df


def test_end_to_end_detects_injected_fault_and_schema():
    et = load_config()["equipment_types"]["static_rotating"]
    df = _synthetic()
    art = model.train("t", et, df)
    out, flat = detect.score(art, et, df)
    rec = episodes.build(out, flat, et, art, "t")
    anomalies = rec[(rec.outcome == "Anomaly") & (rec.category == "Temperature")]
    assert len(anomalies) >= 1
    assert pd.Timestamp("2025-01-26 00:00") <= anomalies.start.min() <= pd.Timestamp("2025-01-28 10:00")
    paths = load_config()["paths"]
    final, _ = output.finalize(rec, paths)
    assert list(final.columns) == output.FINAL_COLS


def test_spike_is_sensor_suspect_not_machine_fault():
    et = load_config()["equipment_types"]["static_rotating"]
    df = _synthetic()
    art = model.train("t", et, df)
    df2 = df.copy()
    df2.loc[300, "Vibration (mm/s)"] = 25.0
    out, flat = detect.score(art, et, df2)
    rec = episodes.build(out, flat, et, art, "t")
    assert (rec.outcome == "Sensor Suspect").any()
    assert not out.loc[300, "flag"]
