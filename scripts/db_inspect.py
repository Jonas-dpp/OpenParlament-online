#!/usr/bin/env python3
"""
Simple DB inspection helper for OpenParlament.

Run: python scripts/db_inspect.py
It prints the SQLAlchemy session bind and row counts for main tables,
plus samples for Zwischenrufe and Reden to verify schema structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

# make sure running from repo root works
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_session
from src.models import Zwischenruf, Rede, Sitzung, Redner
from sqlalchemy import select, func

def main() -> None:
    print(f"Project root: {_PROJECT_ROOT}")
    with get_session() as session:
        # Print session bind / engine info
        try:
            bind = getattr(session, "bind", None)
            print("Session.bind:", bind)
        except Exception as exc:
            print("Session.bind: <unavailable> -", exc)

        print("-" * 50)
        # Counts
        try:
            for cls in (Zwischenruf, Rede, Sitzung, Redner):
                cnt = session.execute(select(func.count()).select_from(cls)).scalar()
                print(f"{cls.__name__} count: {cnt}")
        except Exception as exc:
            print("Failed to query counts:", exc)
            return

        print("-" * 50)
        # Show sample Zwischenruf
        try:
            z_row = session.execute(select(Zwischenruf).limit(1)).scalar_one_or_none()
            if not z_row:
                print("No Zwischenruf rows found.")
            else:
                z_dict = {k: v for k, v in vars(z_row).items() if not k.startswith('_')}
                print("Sample Zwischenruf (DB Structure):")
                print(z_dict)
                print(f"-> Has 'tone_scores'? {'tone_scores' in z_dict}")
        except Exception as exc:
            print("Failed to fetch sample Zwischenruf:", exc)

        print("-" * 50)
        # Show sample Rede
        try:
            r_row = session.execute(select(Rede).limit(1)).scalar_one_or_none()
            if not r_row:
                print("No Rede rows found.")
            else:
                r_dict = {k: v for k, v in vars(r_row).items() if not k.startswith('_')}
                # Wir kürzen den Text, damit die Konsole nicht überflutet wird
                if 'text' in r_dict and r_dict['text']:
                    r_dict['text'] = r_dict['text'][:100] + "..."
                print("Sample Rede (DB Structure):")
                print(r_dict)
                print(f"-> Has NLP fields? {'sentiment_score' in r_dict}")
        except Exception as exc:
            print("Failed to fetch sample Rede:", exc)


if __name__ == "__main__":
    main()