"""Collect every results table into docs/data/info.json for the "Data" page (docs/info.html).

Run from the repo root after the analysis scripts (any that haven't been run are skipped):
    python export_info.py

The page builds its tables from this file when it loads, so rerunning the pipeline and this
script updates the page; no numbers are typed into the HTML.
"""
import json
from datetime import date
from pathlib import Path

from register import OUT

DOCS_DATA = Path("docs") / "data"


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def registration_row(r):
    """The fields the page shows for one 3T T1 -> coarse-scan registration."""
    sc = r["self_consistency"]
    return {
        "subject": r["subject"],
        "contrast": r.get("contrast", "T2w"),
        "mi_before": r["mutual_information"]["before"],
        "mi_after": r["mutual_information"]["after"],
        "self_consistency_mean_mm": sc["mean_error_mm"]["after"],
        "self_consistency_max_mm": sc["max_error_mm"]["after"],
        "converged": r.get("converged", True),
    }


def main():
    info = {"generated": date.today().isoformat()}

    main_rows = load("all_subjects.json")
    if main_rows:
        info["registrations"] = [registration_row(r) for r in main_rows]

    loop = load("consistency_loop.json")
    if loop:
        info["consistency_loop"] = [{
            "subject": r["subject"],
            "loop_mean_mm": r["loop_error_mm"]["mean"],
            "loop_max_mm": r["loop_error_mm"]["max"],
            "motion_3t_mean_mm": r["within_3t_session_motion"]["mean_mm"],
            "motion_3t_rotation_deg": r["within_3t_session_motion"]["rotation_deg"],
            "converged": r.get("converged", True),
        } for r in loop]

    sessions = load("repeat_sessions.json")
    if sessions:
        info["repeat_sessions"] = [{
            "subject": r["subject"],
            "session": r["session"],
            "head_moved_mean_mm": r["head_position_change"]["mean_mm"],
            "head_rotation_deg": r["head_position_change"]["rotation_deg"],
            "loop_mean_mm": r["loop_error_mm"]["mean"],
            "loop_max_mm": r["loop_error_mm"]["max"],
            "converged": r.get("converged", True),
        } for r in sessions]

    field = load("field_strength.json")
    if field:
        info["field_strength"] = [registration_row(r) for r in field]

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    out = DOCS_DATA / "info.json"
    out.write_text(json.dumps(info, indent=1))
    print(f"wrote {out}: {', '.join(k for k in info if k != 'generated')}")


if __name__ == "__main__":
    main()
