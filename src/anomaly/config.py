"""Load the single YAML config and resolve project paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "pipeline.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def abs_path(rel):
    return ROOT / rel
