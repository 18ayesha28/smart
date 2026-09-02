"""
Smart NeuroCare — Patient Visit History (SQLite persistence)

Stores one row per analyzed scan ("visit"), keyed by patient_id, so that
treatment_response.py can compare measurements across visits for the same
patient over time.

Deliberately minimal: no name, no contact info, no demographics — only what
is needed to support longitudinal measurement comparison. patient_id is
whatever pseudonymous identifier the clinician enters (e.g. "P-10293"); this
module does not attach any other identifying information to it.

No Streamlit imports here — this module is pure persistence + data access
and can be used/tested independently of the UI.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

DB_PATH = "patient_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    tumor_type TEXT,
    max_diameter_mm REAL,
    perpendicular_diameter_mm REAL,
    product_bidirectional_mm2 REAL,
    area_mm2 REAL,
    severity_score TEXT,
    overlay_path TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_visits_patient_date ON visits(patient_id, scan_date);
"""


@dataclass
class VisitRecord:
    visit_id: int
    patient_id: str
    scan_date: str
    tumor_type: Optional[str]
    max_diameter_mm: Optional[float]
    perpendicular_diameter_mm: Optional[float]
    product_bidirectional_mm2: Optional[float]
    area_mm2: Optional[float]
    severity_score: Optional[str]
    overlay_path: Optional[str]
    is_demo: bool
    created_at: str


@contextmanager
def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_visit(
    patient_id: str,
    scan_date: str,
    tumor_type: Optional[str] = None,
    max_diameter_mm: Optional[float] = None,
    perpendicular_diameter_mm: Optional[float] = None,
    product_bidirectional_mm2: Optional[float] = None,
    area_mm2: Optional[float] = None,
    severity_score: Optional[str] = None,
    overlay_path: Optional[str] = None,
    is_demo: bool = False,
    db_path: str = DB_PATH,
) -> int:
    """Persist one visit and return its visit_id. Raises ValueError on a blank patient_id."""
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id must be a non-empty string")
    if not scan_date or not scan_date.strip():
        raise ValueError("scan_date must be a non-empty string")

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO visits (
                patient_id, scan_date, tumor_type, max_diameter_mm,
                perpendicular_diameter_mm, product_bidirectional_mm2, area_mm2,
                severity_score, overlay_path, is_demo, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id.strip(), scan_date, tumor_type,
                max_diameter_mm, perpendicular_diameter_mm, product_bidirectional_mm2, area_mm2,
                severity_score, overlay_path, 1 if is_demo else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid


def get_visit_history(patient_id: str, db_path: str = DB_PATH) -> List[VisitRecord]:
    """All visits for a patient, ordered oldest -> newest (by scan_date, then visit_id
    as a tiebreaker for same-day visits so insertion order is preserved)."""
    if not patient_id or not patient_id.strip():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM visits WHERE patient_id = ? ORDER BY scan_date ASC, visit_id ASC",
            (patient_id.strip(),),
        ).fetchall()
    return [
        VisitRecord(
            visit_id=r["visit_id"], patient_id=r["patient_id"], scan_date=r["scan_date"],
            tumor_type=r["tumor_type"], max_diameter_mm=r["max_diameter_mm"],
            perpendicular_diameter_mm=r["perpendicular_diameter_mm"],
            product_bidirectional_mm2=r["product_bidirectional_mm2"], area_mm2=r["area_mm2"],
            severity_score=r["severity_score"], overlay_path=r["overlay_path"],
            is_demo=bool(r["is_demo"]), created_at=r["created_at"],
        )
        for r in rows
    ]


def get_all_patient_ids(db_path: str = DB_PATH) -> List[str]:
    """Distinct patient IDs with at least one recorded visit. Useful for debugging/admin views."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT patient_id FROM visits ORDER BY patient_id").fetchall()
    return [r["patient_id"] for r in rows]


def delete_patient_history(patient_id: str, db_path: str = DB_PATH) -> int:
    """Deletes all visits for a patient_id. Returns the number of rows removed."""
    if not patient_id or not patient_id.strip():
        return 0
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM visits WHERE patient_id = ?", (patient_id.strip(),))
        return cur.rowcount


if __name__ == "__main__":
    vid = record_visit(
        patient_id="P-DEMO01", scan_date="2026-01-01", tumor_type="glioma",
        max_diameter_mm=22.0, perpendicular_diameter_mm=15.0,
        product_bidirectional_mm2=330.0, area_mm2=310.0, severity_score="moderate",
    )
    print("Inserted visit_id:", vid)
    for v in get_visit_history("P-DEMO01"):
        print(v)
