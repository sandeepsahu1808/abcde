"""Create LARGE synthetic training files with the SAME columns as the PoC files.

    python tools/generate_training_data.py                # defaults
    python tools/generate_training_data.py --static-days 300 --cyclic-cycles 2000 --seed 7

Healthy behaviour (sensor relationships, noise) is fitted from the original PoC files.
Faults are ~7% of rows, start gradually (drift -> full fault), and have different severities
(some subtle), so the data is closer to real plants than the first PoC file.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ar1(n, rho, sd, rng):
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def envelope(length, ramp):
    up = max(int(length * ramp), 1)
    env = np.ones(length)
    env[:up] = np.linspace(0.1, 1.0, up)
    return env


# ------------------------------------------------------------------ STATIC ---
STATIC_FAULTS = {  # weight, hours (min,max), ramp share, additive effect at severity 1
    "Bearing Degradation": (3, (24, 72), 0.7, {"vib": 6.1, "temp": 5.0}),
    "Overheat": (2, (12, 48), 0.3, {"temp": 19.4}),
    "Overpressure": (2, (12, 48), 0.2, {"pres": 2.8}),
    "Overspeed": (2, (12, 36), 0.2, {"rpm": 300, "temp": 2.8, "vib": 0.85}),
    "Underspeed": (2, (12, 36), 0.2, {"rpm": -345, "temp": 3.9}),
    "Multi-parameter Cascade": (2, (24, 72), 0.6, {"temp": 15.5, "vib": 5.4, "pres": 2.2, "rpm": 56}),
}


def make_static(n_assets, days, seed):
    rng = np.random.default_rng(seed)
    n = days * 24
    names = list(STATIC_FAULTS)
    weights = np.array([STATIC_FAULTS[k][0] for k in names], float)
    weights /= weights.sum()
    frames = []
    for a in range(n_assets):
        t = pd.date_range(pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 120))), periods=n, freq="h")
        hour = t.hour.values
        rpm = 1064 + rng.normal(0, 15) + 40 * np.sin(2 * np.pi * (hour - 6) / 24) + ar1(n, 0.88, 40, rng)
        rpm = np.clip(rpm, 900, 1200)
        temp = 0.0332 * rpm + 52.238 + ar1(n, 0.5, 1.42, rng)
        vib = 0.0099 * rpm - 7.8408 + rng.normal(0, 0.306, n)
        pres = 0.0055 * rpm - 4.0183 + rng.normal(0, 0.128, n)
        d = {"temp": temp, "vib": vib, "pres": pres, "rpm": rpm}
        label = np.array(["Normal"] * n, dtype=object)

        pos = 48
        while True:
            pos += int(rng.integers(200, 800))
            f = rng.choice(names, p=weights)
            _, (lo, hi), ramp, eff = STATIC_FAULTS[f]
            L = int(rng.integers(lo, hi + 1))
            if pos + L > n - 24:
                break
            env = envelope(L, ramp) * rng.uniform(0.5, 1.2)
            for k, v in eff.items():
                d[k][pos:pos + L] += v * env
            label[pos:pos + L] = np.where(env / env.max() * 1.0 >= 0.0, f, "Normal")
            # rows in the first weak part of the drift are still labelled Normal (realistic)
            weak = envelope(L, ramp) < 0.3
            label[pos:pos + L][weak] = "Normal"
            pos += L

        free = np.flatnonzero(label == "Normal")
        for s in rng.choice(free, size=rng.poisson(n * 0.0025), replace=False):
            ln = int(rng.integers(1, 4))
            ch = rng.choice(["temp", "vib", "pres"])
            jump = {"temp": rng.uniform(15, 30), "vib": rng.uniform(4, 9), "pres": rng.uniform(1.5, 3.5)}[ch]
            d[ch][s:s + ln] += jump
            label[s:s + ln] = "Sensor Spike"

        df = pd.DataFrame({
            "Object ID": f"EQP-{a + 1:03d}", "Timestamp": t,
            "Temperature (Degree Cel)": d["temp"].round(1), "Vibration (mm/s)": np.clip(d["vib"], 0.2, None).round(2),
            "Pressure (bars)": np.clip(d["pres"], 0.2, None).round(2), "Engine RPM": d["rpm"].round(0).astype(int),
            "Health Status": np.where(label == "Normal", "Healthy", "Unhealthy"), "Anomaly Type": label})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns[2:6]:   # a few missing readings, like real SCADA
        df.loc[rng.random(len(df)) < 0.0005, c] = np.nan
    return df


# ------------------------------------------------------------------ CYCLIC ---
# healthy model: feature = a + b * GrossLoad + noise(sd)   (fitted from the PoC file)
CYC = {
    "Duration (s)": (87.53, -0.0367, 5.75), "Peak Hoist Current (A)": (208.66, 17.96, 24.37),
    "Mean Hoist Current Loaded (A)": (104.86, 17.26, 5.92), "Peak Vibration (mm/s)": (2.396, 0.0002, 0.0706),
    "Motor Temp Rise (C)": (-16.09, 0.6465, 5.34), "Peak Motor Temp (C)": (33.82, 0.9132, 5.57),
    "Mean Motor Temp (C)": (35.16, 0.5204, 4.28), "Energy per Lift (kWh)": (3.288, 0.091, 0.5115),
    "Lock Time (s)": (4.281, 0.0064, 1.009), "Unlock Time (s)": (4.833, -0.0127, 1.157),
    "Peak Hyd Pressure (bar)": (180.74, 0.0157, 3.73), "Trolley Traverse (s)": (22.95, -0.0185, 2.435),
    "Peak Trolley Current (A)": (330.1, -0.506, 93.75)}
THERMAL = ["Motor Temp Rise (C)", "Peak Motor Temp (C)", "Mean Motor Temp (C)"]

CYC_FAULTS = {  # weight, cycles (min,max), ramp, multiplier on current, additive effects at severity 1
    "Bearing Wear": (3, (5, 14), 0.6, 0.175, {"Peak Vibration (mm/s)": 2.85, "Peak Hoist Current (A)": 102}),
    "Brake Drag": (2, (4, 10), 0.3, 0.69, {"Motor Temp Rise (C)": 9.6, "Peak Motor Temp (C)": 16.7, "Mean Motor Temp (C)": 11,
                                           "Energy per Lift (kWh)": 1.95, "Peak Hoist Current (A)": 256}),
    "Cooling Fault": (2, (5, 12), 0.5, 0, {"Motor Temp Rise (C)": 10, "Peak Motor Temp (C)": 12.8, "Mean Motor Temp (C)": 12.8}),
    "Hydraulic Leak": (2, (4, 10), 0.3, 0, {"Lock Time (s)": 4.3, "Unlock Time (s)": 3.8, "Peak Hyd Pressure (bar)": -53.7, "Duration (s)": 7.5}),
    "Trolley Drive Fault": (2, (4, 10), 0.3, 0, {"Peak Trolley Current (A)": 122, "Trolley Traverse (s)": 3.2}),
    "Rope/Sheave Fault": (2, (4, 10), 0.4, 0, {"Peak Vibration (mm/s)": 3.3, "Peak Hoist Current (A)": 44}),
}


def make_cyclic(n_cranes, n_cycles, seed):
    rng = np.random.default_rng(seed + 1)
    names = list(CYC_FAULTS)
    w = np.array([CYC_FAULTS[k][0] for k in names], float)
    w /= w.sum()
    frames = []
    for c in range(n_cranes):
        n = n_cycles
        payload = np.where(rng.random(n) < 0.12, 0.0, np.clip(rng.normal(18, 8, n), 3, 31.7)).round(1)
        gross = (12.0 + payload).round(1)
        shared = ar1(n, 0.9, 1.0, rng)
        f = {}
        for k, (a, b, sd) in CYC.items():
            if k in THERMAL:
                noise = sd * (0.7 * shared + np.sqrt(1 - 0.49) * rng.normal(0, 1, n))
            else:
                noise = rng.normal(0, sd, n)
            f[k] = a + b * gross + noise
        label = np.array(["Normal"] * n, dtype=object)
        cur_mul = np.ones(n)

        pos = 30
        while True:
            pos += int(rng.integers(40, 130))
            ft = rng.choice(names, p=w)
            _, (lo, hi), ramp, mul, eff = CYC_FAULTS[ft]
            L = int(rng.integers(lo, hi + 1))
            if pos + L > n - 10:
                break
            env = envelope(L, ramp) * rng.uniform(0.5, 1.2)
            sl = slice(pos, pos + L)
            cur_mul[sl] += mul * env
            for k, v in eff.items():
                f[k][sl] += v * env
            label[sl] = ft
            label[sl][envelope(L, ramp) < 0.3] = "Normal"
            pos += L

        f["Mean Hoist Current Loaded (A)"] = f["Mean Hoist Current Loaded (A)"] * cur_mul
        dur = np.clip(f["Duration (s)"], 60, None).round().astype(int)
        start = [pd.Timestamp("2025-04-01 06:00:00") + pd.Timedelta(days=c)]
        for i in range(1, n):
            gap = max(2, int(rng.normal(8, 3)))
            brk = 4 * 3600 if i % 160 == 0 else 0
            start.append(start[-1] + pd.Timedelta(seconds=int(dur[i - 1] + gap + brk)))
        mean_cur = f["Mean Hoist Current Loaded (A)"]
        df = pd.DataFrame({
            "Crane ID": f"STS-{c + 1:02d}", "Cycle ID": np.arange(n), "Cycle Start": start, "Duration (s)": dur,
            "Payload (t)": payload, "Gross Load (t)": gross,
            "Peak Hoist Current (A)": np.clip(f["Peak Hoist Current (A)"], 150, None).round(1),
            "Mean Hoist Current Loaded (A)": mean_cur.round(1),
            "Current per Ton (A/t)": (mean_cur / gross).round(2),
            "Peak Vibration (mm/s)": np.clip(f["Peak Vibration (mm/s)"], 0.5, None).round(2),
            "Motor Temp Rise (C)": f["Motor Temp Rise (C)"].round(1), "Peak Motor Temp (C)": f["Peak Motor Temp (C)"].round(1),
            "Mean Motor Temp (C)": f["Mean Motor Temp (C)"].round(1),
            "Energy per Lift (kWh)": np.clip(f["Energy per Lift (kWh)"], 0.5, None).round(2),
            "Lock Time (s)": np.clip(f["Lock Time (s)"], 1, None).round().astype(int),
            "Unlock Time (s)": np.clip(f["Unlock Time (s)"], 1, None).round().astype(int),
            "Peak Hyd Pressure (bar)": f["Peak Hyd Pressure (bar)"].round(1),
            "Trolley Traverse (s)": np.clip(f["Trolley Traverse (s)"], 8, None).round().astype(int),
            "Peak Trolley Current (A)": np.clip(f["Peak Trolley Current (A)"], 120, None).round(1),
            "Health Status": np.where(label == "Normal", "Healthy", "Faulty"), "Fault Type": label})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def update_masters(static_ids, cyclic_ids):
    am_p, sm_p = ROOT / "config/asset_master.csv", ROOT / "config/sensor_master.csv"
    am, sm = pd.read_csv(am_p), pd.read_csv(sm_p)
    s_cats = ["Temperature", "Vibration", "Pressure", "RPM"]
    c_cats = ["Hoist Current", "Vibration", "Temperature", "Hydraulic Pressure", "Lock Time", "Unlock Time",
              "Energy", "Trolley Current", "Trolley Traverse"]
    add_a, add_s = [], []
    for i, oid in enumerate(static_ids):
        add_a.append((oid, "Rotating Equipment", f"SITE-0{1 + i % 2}"))
        add_s += [(oid, c, f"SNS-{oid}-{c.upper().replace(' ', '_')}") for c in s_cats]
    for i, oid in enumerate(cyclic_ids):
        add_a.append((oid, "STS", f"SITE-0{1 + i % 2}"))
        add_s += [(oid, c, f"SNS-{oid}-{c.upper().replace(' ', '_')}") for c in c_cats]
    am = pd.concat([am, pd.DataFrame(add_a, columns=am.columns)]).drop_duplicates("object_id")
    sm = pd.concat([sm, pd.DataFrame(add_s, columns=sm.columns)]).drop_duplicates(["object_id", "parameter_category"])
    am.to_csv(am_p, index=False)
    sm.to_csv(sm_p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-assets", type=int, default=8)
    ap.add_argument("--static-days", type=int, default=150)
    ap.add_argument("--cyclic-cranes", type=int, default=4)
    ap.add_argument("--cyclic-cycles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = ROOT / "data" / "raw"
    s = make_static(a.static_assets, a.static_days, a.seed)
    c = make_cyclic(a.cyclic_cranes, a.cyclic_cycles, a.seed)
    s.to_excel(out / "Static_Training_Data.xlsx", sheet_name="Sheet1", index=False)
    c.to_excel(out / "Cyclic_Training_Data.xlsx", sheet_name="Cycle_Features", index=False)
    update_masters(sorted(s["Object ID"].unique()), sorted(c["Crane ID"].unique()))
    print("static", s.shape, s["Anomaly Type"].value_counts(normalize=True).round(3).to_dict())
    print("cyclic", c.shape, c["Fault Type"].value_counts(normalize=True).round(3).to_dict())
