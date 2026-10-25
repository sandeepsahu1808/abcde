# Anomaly Detection PoC - static + cyclic assets

One config-driven pipeline. Static equipment (pumps, motors, gensets - hourly series) and cyclic
equipment (STS/QC/RTG - one row per work cycle) share the same steps and the same final report.

## Run it
```bash
pip install -r requirements.txt

uvicorn anomaly.api:app --app-dir src --reload   # for run the API 

python run_pipeline.py all        # train + detect  -> outputs/anomaly_report.xlsx
python run_pipeline.py detect     # only score with saved models (scheduled job)
python run_pipeline.py train --types cyclic_sts
pytest tests
```
   python run_pipeline.py train --config config/pipeline_large.yaml
## Folder structure
```
anomaly_detection_poc/
├── run_pipeline.py          entry point
├── config/
│   ├── pipeline.yaml        ALL settings: columns, limits, thresholds, models (edit this, not code)
│   ├── asset_master.csv     Object ID -> Asset Category, Site          (PLACEHOLDERS - replace)
│   └── sensor_master.csv    Object ID + Parameter -> Sensor Device ID   (PLACEHOLDERS - replace)
├── data/raw/                input Excel files (later: DB)
├── data/processed/          per-row scores (audit trail)
├── models/<type>/           saved model + metadata.json
├── outputs/                 anomaly_report.xlsx / .csv  (the deliverable)
├── reports/                 evaluation csvs
├── logs/                    pipeline.log
├── src/anomaly/
│   ├── ingest.py            1 read + standardise (only file that changes for DB)
│   ├── quality.py           2 data-quality gate: spikes, frozen sensors
│   ├── features.py          3 rolling stats / per-cycle features
│   ├── baseline.py          4 "what is normal given the load" (healthy only baseline)
│   ├── model.py             5 train (XGBoost or max-z), save/load
│   ├── detect.py            6 score + hard-limit rules
│   ├── episodes.py          7 alarms -> one record per episode per parameter, confidence
│   ├── evaluate.py          8 recall / false alarms / precision (needs labels)
│   ├── output.py            9 final 9-column report
│   └── pipeline.py          orchestrates 1-9
└── tests/
```

## How it works (plain English)
1. **Learn patterns**: XGBoost learns to separate healthy and faulty rows (supervised), while Cyclic uses residual max z-score after filtering load effects.
2. **Score**: every new row gets an "how abnormal" score; high = unusual versus healthy.
3. **Hybrid**: model score OR hard limit (e.g. temp > 95) raises an alarm.
4. **Sensor gate**: 1-3 hour extreme jumps and frozen sensors become **Sensor Suspect**, not machine faults.
5. **Episodes**: consecutive alarms merge into one episode. Sustained = **Anomaly**, brief = **Watch**.
6. **Report**: one row per episode per parameter, first time that parameter went abnormal.

Same approach for both asset types; what differs per type is only the config + features:
static = rolling mean/std of each sensor, XGBoost Classifier; cyclic = every feature adjusted for crane load, then max z-score of the residual.

## Final report columns
Asset Category | Sensor Device ID | Object ID | Site | Parameter Category | Anomaly Start Date/Time | Detected Value | Data Confidence | Detection Outcome

**Data Confidence** = % of healthy reference scores that are lower than this episode's score. It is a
ranking confidence, not a true probability.

## Add a new equipment type (e.g. RTG, Genset)
1. Copy a block in `config/pipeline.yaml`, set file/sheet, column names, parameters, limits.
2. Add rows in `asset_master.csv` / `sensor_master.csv`.
3. `python run_pipeline.py all --types my_new_type`

## Known limits (be honest in the demo)
- Data is synthetic; evaluation numbers are optimistic. Judge on real, maintenance-confirmed failures.
- Only 128 healthy cycles / ~1000 healthy hours: hold-out false-alarm numbers are noisy.
- Encoder faults and slow drift that never breaches limits are not reliably detected yet.
- Sensor/Site/Category masters are placeholders.

## Moving to production (minimal-touch operation)
1. Replace `ingest.py` Excel read with a DB query (incremental, last N hours).
2. Schedule `detect` (cron/Airflow); retrain monthly or on drift; promote a new model only if it beats the old one.
3. Use time-based splits and per-asset models once months of data exist; add LSTM/autoencoder only if needed.
4. Write results to a DB table (same 9 columns) instead of Excel; add operator feedback (true/false alarm) to improve thresholds.
5. Monitor: alert volume, score drift, data-missing rate.

## Retrain with the LARGE training files (or your own)
```bash
python tools/generate_training_data.py                       # (re)create data/raw/Static_Training_Data.xlsx + Cyclic_Training_Data.xlsx
python run_pipeline.py all --config config/pipeline_large.yaml
```
Results: `outputs/large/anomaly_report.xlsx`, `reports/large/evaluation_*.csv`, models in `models/static_large`, `models/cyclic_large`.
To use YOUR file: keep the same column names, put it in `data/raw/`, and point `source.file` in `config/pipeline_large.yaml` to it.
Training needs a Health/Fault label column only to pick healthy rows (and to evaluate); without labels all rows are assumed healthy.
