#!/usr/bin/env python3
"""
Import all XML files from data/xml using the project's ProtocolImporter.
Also backfills datum and wochentag for any already-imported Sitzungen that
are still missing those values (e.g. imported before the parser was fixed).

Usage:
  python scripts/import_xmls.py
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import select
from src.database import init_db, get_session
from src.models import Sitzung
from src.parser import BundestagXMLParser, weekday_german
from scripts.run_scraper import ProtocolImporter


def _backfill_missing_dates(xml_dir: Path) -> None:
    """Re-parse local XML files to fill in datum / wochentag where NULL."""
    parser = BundestagXMLParser()

    with get_session() as session:
        rows = session.execute(
            select(Sitzung).where(
                (Sitzung.datum.is_(None)) | (Sitzung.wochentag.is_(None))
            )
        ).scalars().all()

        if not rows:
            logger.info("All Sitzungen already have datum and wochentag.")
            return

        logger.info(
            "Backfilling datum/wochentag for %d Sitzung(en) with missing data…",
            len(rows),
        )
        updated = 0
        for sitzung in rows:
            xml_file = xml_dir / f"{sitzung.wahlperiode:02d}{sitzung.sitzungsnr:03d}.xml"
            if not xml_file.exists():
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
                        "  WP%d/%d → datum=%s  wochentag=%s",
                        sitzung.wahlperiode, sitzung.sitzungsnr,
                        sitzung.datum, sitzung.wochentag,
                    )
            except Exception:
                logger.exception("Failed to re-parse %s", xml_file.name)

        session.commit()
        logger.info("Backfilled %d/%d session(s).", updated, len(rows))


def main() -> None:
    init_db()
    importer = ProtocolImporter()
    xml_dir = _PROJECT_ROOT / "data" / "xml"
    if not xml_dir.exists():
        logger.error("XML directory not found: %s", xml_dir)
        return

    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        logger.error("No XML files found in %s", xml_dir)
        return

    # ── Import new files ───────────────────────────────────────────────────────
    imported = 0
    for p in files:
        logger.info("Importing %s", p.name)
        try:
            if importer.import_file(p):
                imported += 1
        except Exception:
            logger.exception("Failed to import %s", p.name)

    logger.info("Imported %d/%d files.", imported, len(files))

    # ── Backfill any sessions that are still missing datum / wochentag ─────────
    _backfill_missing_dates(xml_dir)


if __name__ == "__main__":
    main()
