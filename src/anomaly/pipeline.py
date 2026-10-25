"""Orchestrator: ingest -> quality -> features -> train/score -> episodes -> report."""
import logging
import pandas as pd
from . import detect, episodes, evaluate, ingest, model, output
from . import metrics_logger
from .config import abs_path, load_config

log = logging.getLogger("anomaly")


def setup_logging(cfg):
    lf = abs_path(cfg["paths"]["log_file"])
    lf.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(lf)], force=True)


def run(stage="all", types=None, config_path=None):
    cfg = load_config(config_path)
    setup_logging(cfg)
    for k in ("processed_dir", "report_dir"):
        abs_path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    recs, sums, byf, null_audits = [], [], [], []
    for name, et in cfg["equipment_types"].items():
        if not et.get("enabled", True) or (types and name not in types):
            continue
        log.info("=== %s (%s) ===", name, et["mode"])
        df = ingest.load_table(et)
        log.info("loaded %d rows, %d assets", len(df), df["object_id"].nunique())

        if stage in ("train", "all"):
            art = model.train(name, et, df)
            model.save(art)
        if stage in ("detect", "all"):
            art = model.load(name)
            tel = ingest.load_raw(et["flatline"]["file"], et["flatline"]["sheet"]) if (et.get("flatline") and "file" in et["flatline"]) else None
            out, flat = detect.score(art, et, df, tel)
            out.to_csv(abs_path(cfg["paths"]["processed_dir"]) / f"{name}_scored.csv", index=False)
            rec = episodes.build(out, flat, et, art, name)
            log.info("%d episode-parameter records (%s)", len(rec), rec["outcome"].value_counts().to_dict())
            recs.append(rec)
            null_audits.extend(out.attrs.get("null_audit", []))
            s, b = evaluate.evaluate(out, art, name)
            if s is not None:
                sums.append(s)
                byf.append(b)

    if stage in ("detect", "all") and recs:
        final, details = output.finalize(pd.concat(recs, ignore_index=True), cfg["paths"])
        ev_s = pd.concat(sums, ignore_index=True) if sums else pd.DataFrame()
        ev_f = pd.concat(byf, ignore_index=True) if byf else pd.DataFrame()
        xlsx, csv = output.write_report(final, details, ev_s, ev_f, abs_path(cfg["paths"]["output_dir"]), null_audit=null_audits)
        ev_s.to_csv(abs_path(cfg["paths"]["report_dir"]) / "evaluation_summary.csv", index=False)
        ev_f.to_csv(abs_path(cfg["paths"]["report_dir"]) / "evaluation_by_fault.csv", index=False)
        log.info("report written: %s", xlsx)
        return final, ev_s, ev_f
    return None
