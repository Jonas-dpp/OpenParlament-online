#!/usr/bin/env python3
"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------
db_patch.py – Fügt fehlende Spalten zur SQLite-Datenbank hinzu und
backfüllt fehlende Werte, ohne bestehende Daten zu löschen.

Run from the project root:
    python scripts/db_patch.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from sqlalchemy import text, select
from src.database import get_engine, get_session
from src.models import Sitzung


def _add_column_safe(conn, table: str, col_name: str, col_type: str) -> None:
    """ALTER TABLE … ADD COLUMN, silently skipping if the column already exists."""
    try:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
        logger.info("Added column %s.%s (%s)", table, col_name, col_type)
    except Exception as exc:
        if "duplicate column name" in str(exc).lower():
            logger.info("Column %s.%s already exists – skipped.", table, col_name)
        else:
            logger.error("Error adding %s.%s: %s", table, col_name, exc)
            raise


def patch_schema() -> None:
    """Add any missing columns to existing tables."""
    engine = get_engine()
    logger.info("Patching schema: %s", engine.url)

    with engine.begin() as conn:
        # sitzungen
        logger.info("--- Table: sitzungen ---")
        _add_column_safe(conn, "sitzungen", "wochentag", "VARCHAR(16)")

        # zwischenrufe
        logger.info("--- Table: zwischenrufe ---")
        _add_column_safe(conn, "zwischenrufe", "tone_scores", "JSON")

        # reden
        logger.info("--- Table: reden ---")
        for col_name, col_type in [
            ("sentiment_score", "FLOAT"),
            ("ton_label",       "VARCHAR(64)"),
            ("tone_scores",     "JSON"),
            ("adressaten",      "VARCHAR(512)"),
        ]:
            _add_column_safe(conn, "reden", col_name, col_type)


def patch_sitzung_dates() -> None:
    """Backfill datum and wochentag for Sitzungen where either value is NULL.

    Requires the corresponding XML files to be present in data/xml/.
    Only sessions whose XML file exists locally are updated.
    """
    from src.parser import BundestagXMLParser
    from src.parser import weekday_german

    xml_dir = _PROJECT_ROOT / "data" / "xml"
    if not xml_dir.exists():
        logger.warning("XML directory not found (%s) – skipping date backfill.", xml_dir)
        return

    parser = BundestagXMLParser()

    with get_session() as session:
        # Only process sessions that are still missing datum or wochentag.
        rows = session.execute(
            select(Sitzung).where(
                (Sitzung.datum.is_(None)) | (Sitzung.wochentag.is_(None))
            )
        ).scalars().all()

        if not rows:
            logger.info("All Sitzungen already have datum and wochentag – nothing to do.")
            return

        logger.info("%d Sitzung(en) with missing datum/wochentag found.", len(rows))
        updated = 0

        for sitzung in rows:
            xml_file = xml_dir / f"{sitzung.wahlperiode:02d}{sitzung.sitzungsnr:03d}.xml"
            if not xml_file.exists():
                logger.debug("XML not found locally: %s – skipping.", xml_file.name)
                continue
            try:
                result = parser.parse_file(xml_file)
                changed = False
                if sitzung.datum is None and result.sitzung.datum is not None:
                    sitzung.datum = result.sitzung.datum
                    changed = True
                if sitzung.wochentag is None and sitzung.datum is not None:
                    sitzung.wochentag = weekday_german(sitzung.datum)
                    changed = True
                if changed:
                    updated += 1
                    logger.info(
                        "Updated WP%d/%d → datum=%s wochentag=%s",
                        sitzung.wahlperiode, sitzung.sitzungsnr,
                        sitzung.datum, sitzung.wochentag,
                    )
            except Exception:
                logger.exception("Failed to re-parse %s", xml_file.name)

        session.commit()
        logger.info("Backfilled %d/%d session(s).", updated, len(rows))


def patch_database() -> None:
    patch_schema()
    patch_sitzung_dates()
    logger.info("Patch complete.")


if __name__ == "__main__":
    patch_database()
