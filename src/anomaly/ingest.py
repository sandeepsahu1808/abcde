"""Step 1 - read the source table and standardise it.

Today the source is Excel. When data moves to a database, ONLY this file
changes (replace pd.read_excel with pd.read_sql); everything downstream
works on the same standardised DataFrame.
"""
import logging
import numpy as np
import pandas as pd
from .config import abs_path

log = logging.getLogger(__name__)


def read_any(file, sheet=None, name=""):
    """Read xlsx or csv from a path or file-like object (used by the API upload)."""
    if str(name or file).lower().endswith(".csv"):
        return pd.read_csv(file)
    if sheet is not None:
        try:
            return pd.read_excel(file, sheet_name=sheet)
        except ValueError:          # sheet name not in the uploaded workbook -> first sheet
            if hasattr(file, "seek"):
                file.seek(0)
    return pd.read_excel(file, sheet_name=0)


def load_table(et, file=None, name=""):
    """Load from the config path (batch run) or from an uploaded file (API)."""
    src = et["source"]
    if file is None:
        file_list = src.get("files") if "files" in src else src.get("file")
        if isinstance(file_list, list):
            dfs = []
            for idx, fpath in enumerate(file_list):
                log.info("Loading file %d: %s", idx + 1, fpath)
                temp_df = pd.read_excel(abs_path(fpath), sheet_name=src.get("sheet", 0))
                
                obj_col = et["columns"]["object_id"]
                if obj_col in temp_df.columns:
                    temp_df[obj_col] = temp_df[obj_col].astype(str).str.strip()
                dfs.append(temp_df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            log.info("Loading %s", file_list)
            df = pd.read_excel(abs_path(file_list), sheet_name=src.get("sheet", 0))
    else:
        df = read_any(file, src.get("sheet", 0), name)
    cols = et["columns"]
    params = list(et["parameters"])
    context = et.get("context", [])
    missing = [c for c in [cols["object_id"], cols["timestamp"], *params, *context] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in source: {missing}")

    df = df.rename(columns={cols["object_id"]: "object_id", cols["timestamp"]: "timestamp"})
    if "cycle_id" in cols and cols["cycle_id"] in df.columns and "cycle_id" not in df.columns:
        df["cycle_id"] = df[cols["cycle_id"]]
    if "phase" in cols and cols["phase"] in df.columns and "phase" not in df.columns:
        df["phase"] = df[cols["phase"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in params + context:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.drop_duplicates(["object_id", "timestamp"]).sort_values(["object_id", "timestamp"]).reset_index(drop=True)
    if len(df) < n0:
        log.warning("Dropped %d duplicate rows", n0 - len(df))

    # Fill context columns (load/rpm if any) via simple forward/backward fill
    if context:
        df[context] = df.groupby("object_id")[context].ffill().bfill()
    # Missing sensor values in params are preserved here and causally imputed
    # with a complete audit trail in quality.impute_nulls().

    lab = et.get("labels", {})
    if lab and lab.get("fault") in df.columns:
        df["_fault"] = df[lab["fault"]].astype(str)
        df["_healthy"] = df["_fault"] == lab["normal_value"]
        df["_labelled"] = True
    else:  # unlabelled production data: assume it is healthy for training
        df["_fault"] = np.nan
        df["_healthy"] = True
        df["_labelled"] = False
    return df


def load_raw(file, sheet):
    return pd.read_excel(abs_path(file), sheet_name=sheet)
