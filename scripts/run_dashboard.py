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
run_dashboard.py – Launch the OpenParlament Streamlit dashboard.

Registered as the `openparlament-dashboard` console-script entry-point in
pyproject.toml so that after ``pip install -e .`` you can simply run:

    openparlament-dashboard

Equivalent manual command:

    streamlit run src/app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _default_db_url() -> str:
    """Return a stable, user-writable default SQLite URL for the dashboard.

    Resolves a platform-appropriate application-data directory so that
    ``openparlament-dashboard`` behaves consistently regardless of the
    working directory from which it is invoked.
    """
    if sys.platform == "win32":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        data_dir = base_dir / "OpenParlament"
    elif sys.platform == "darwin":
        data_dir = Path.home() / "Library" / "Application Support" / "OpenParlament"
    else:
        base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        data_dir = base_dir / "openparlament"

    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = (data_dir / "openparlament.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def main() -> None:
    """Start the Streamlit dashboard for OpenParlament."""
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "src" / "app.py"

    env = os.environ.copy()
    env.setdefault("OPENPARLAMENT_DB_URL", _default_db_url())

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
