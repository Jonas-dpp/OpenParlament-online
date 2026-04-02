#!/usr/bin/env python3
"""
db_init.py – Initialise the OpenParlament SQLite database.

Run from the project root:
    python scripts/db_init.py

Options:
    --reset   Drops all tables before initializing (WARNING: Deletes all data).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_engine, init_db
from src.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the OpenParlament database.")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Drop all tables before initializing (WARNING: Deletes all data)."
    )
    args = parser.parse_args()

    engine = get_engine()

    if args.reset:
        logger.warning("Resetting database: Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped.")

    logger.info("Initialising database...")
    init_db()
    
    db_url = str(engine.url)
    logger.info("Database ready at: %s", db_url)
    
    with engine.connect() as conn:
        tables = list(engine.dialect.get_table_names(conn))
        logger.info("Tables in database: %s", tables)


if __name__ == "__main__":
    main()