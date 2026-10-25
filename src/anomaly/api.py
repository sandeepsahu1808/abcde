"""FastAPI wrapper. Run:  uvicorn anomaly.api:app --app-dir src --reload   ->  http://127.0.0.1:8000/docs"""
import io
import sys
from pathlib import Path

# Ensure 'src' is in sys.path so joblib can unpickle 'anomaly.*' modules regardless of working directory
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from enum import Enum
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from . import detect, episodes, ingest, model, output
from .config import load_config
import os
import re
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

cfg = load_config()
TYPES = {k: v for k, v in cfg["equipment_types"].items() if v.get("enabled", True)}
# dropdown in Swagger is built from the config: add a type in YAML -> it appears here
EquipmentType = Enum("EquipmentType", {k: k for k in TYPES}, type=str)

app = FastAPI(title="Anomaly Detection API",
              description="Choose static / cyclic, upload the data file, get anomalies as JSON.", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "equipment_types": list(TYPES)}


@app.post("/detect")
async def detect_anomalies(
    equipment_type: EquipmentType,
    file: UploadFile = File(..., description="Data file (.xlsx or .csv) with the columns defined in config"),
):
    name = equipment_type.value
    et = TYPES[name]
    try:
        art = model.load(name)
    except FileNotFoundError:
        raise HTTPException(409, f"No trained model for '{name}'. Run: python run_pipeline.py train --types {name}")
    try:
        df = ingest.load_table(et, io.BytesIO(await file.read()), file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))

    out, flat = detect.score(art, et, df, None)
    rec = episodes.build(out, flat, et, art, name)
    final, _ = output.finalize(rec, cfg["paths"], equipment_type=name)
    final["Anomaly Start Date/Time"] = pd.to_datetime(final["Anomaly Start Date/Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    null_audit = out.attrs.get("null_audit", [])
    formatted_audit = output.format_null_audit(null_audit, equipment_type=name)
    audit_records = formatted_audit.to_dict(orient="records") if (formatted_audit is not None and not formatted_audit.empty) else []
    real_imputations_count = len([r for r in audit_records if r.get("Consecutive Null Count", 0) > 0])

    sensor_timeseries = output.build_sensor_timeseries(out, et, equipment_type=name, episodes_df=final)

    return {"equipment_type": name, "rows_scored": int(len(df)), "anomalies_found": int(len(final)),
            "null_imputations_count": real_imputations_count,
            "summary": final["Detection Outcome"].value_counts().to_dict(),
            "results": final.to_dict(orient="records"),
            "null_imputations": audit_records,
            "sensor_timeseries": sensor_timeseries}


class ExportRequest(BaseModel):
    records: list[dict]
    filtered_records: list[dict] | None = None
    equipment_type: str | None = None
    filename: str | None = None
    null_imputations: list[dict] | None = None


@app.post("/export-excel")
async def export_excel(payload: ExportRequest):
    if not payload.records:
        raise HTTPException(400, "No records provided to export")

    buf = output.export_to_excel_buffer(
        records=payload.records,
        filtered_records=payload.filtered_records,
        equipment_type=payload.equipment_type,
        null_imputations=payload.null_imputations,
    )

    base_name = payload.filename or "Anomaly_Detection_Results"
    if base_name.lower().endswith(".xlsx") or base_name.lower().endswith(".csv"):
        base_name = os.path.splitext(base_name)[0]
    safe_base = re.sub(r"[^\w\-.]", "_", base_name).strip("_")
    out_filename = f"{safe_base}_Anomalies.xlsx"

    headers = {
        "Content-Disposition": f'attachment; filename="{out_filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition"
    }
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )


_static_dir = Path(__file__).resolve().parents[2] / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/{full_path:path}")
async def serve_ui(full_path: str):
    return FileResponse(str(_static_dir / "index.html"))
