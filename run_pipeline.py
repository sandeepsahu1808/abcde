"""Usage:
    python run_pipeline.py all                      # train + detect, all equipment types
    python run_pipeline.py train --types cyclic_sts # retrain one type
    python run_pipeline.py detect                   # score with saved models (daily/hourly job)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from anomaly.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Anomaly detection pipeline")
    ap.add_argument("stage", choices=["train", "detect", "all"], nargs="?", default="all")
    ap.add_argument("--types", nargs="*", help="equipment types from config (default: all enabled)")
    ap.add_argument("--config", help="path to a different pipeline.yaml")
    a = ap.parse_args()
    run(a.stage, a.types, a.config)
