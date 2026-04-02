"""
Frontend tests for the OpenParlament Streamlit dashboard.

Uses ``streamlit.testing.v1.AppTest`` to exercise the app's render paths
with an empty, per-test SQLite database in a temporary directory so that
runtime errors (e.g. a missing import, a NameError, or a broken render
function) are caught early in CI.
"""

from __future__ import annotations

import pytest
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")

# Every page key that exists in the dashboard's PAGES map (see src/app.py).
ALL_PAGE_KEYS = [
    "startseite",
    "aggressions-radar",
    "themen-trend",
    "interaktions-netzwerk",
    "ton-analyse",
    "adressaten-analyse",
    "scraping-monitor",
    "wahlperioden-vergleich",
    "top-analyse",
    "reaktions-analyse",
    "redezeit-gerechtigkeit",
    "debattenklima",
    "redner-profil",
]


def _reset_db_cache() -> None:
    """Reset the cached SQLAlchemy engine and session factory via the public API."""
    from src.database import reset_db_state
    reset_db_state()


def _make_app(page_key: str):
    """Return a configured ``AppTest`` instance for *page_key*."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = page_key
    return at


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Each test gets its own isolated SQLite database."""
    db_url = f"sqlite:///{tmp_path}/test_app.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", db_url)

    _reset_db_cache()

    # Also clear Streamlit's cache so helpers re-query the fresh DB.
    try:
        import streamlit as st
        st.cache_data.clear()
    except (ImportError, AttributeError):
        # If Streamlit is missing or the cache API is unavailable, skip cache clear.
        pass

    yield

    _reset_db_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAppLoadsWithoutCrash:
    """The app must not raise an exception on any page with an empty DB."""

    @pytest.mark.parametrize("page_key", ALL_PAGE_KEYS)
    def test_page_renders_without_exception(self, page_key):
        """Each dashboard page should load without raising a Python exception."""
        at = _make_app(page_key)
        at.run()
        assert not at.exception, (
            f"Page '{page_key}' raised an exception: {at.exception}"
        )


class TestStartseiteContent:
    """Basic smoke-test for the landing page."""

    def test_startseite_has_header(self):
        at = _make_app("startseite")
        at.run()
        assert not at.exception, f"Startseite raised: {at.exception}"
        # The page should contain at least one visible text element.
        all_text_parts = []
        for collection in (at.title, at.header, at.subheader, at.markdown):
            for el in collection:
                if hasattr(el, "value") and el.value:
                    all_text_parts.append(el.value)
        assert len(all_text_parts) > 0, "Startseite rendered no visible text elements"

    def test_startseite_navigation_buttons_present(self):
        at = _make_app("startseite")
        at.run()
        assert not at.exception
        # The landing page should show at least one navigation button.
        assert len(at.button) >= 1, "Expected at least one navigation button on Startseite"


class TestImportsAreComplete:
    """Regression test: ensure Rede (and other models) are imported in app source."""

    def test_rede_import_in_app_source(self):
        """src/app.py must explicitly import Rede from src.models."""
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "Rede" in app_source, (
            "src/app.py does not import 'Rede' from src.models. "
            "This would cause a NameError at runtime."
        )
        # Check the models import line contains Rede.
        import re
        matches = re.findall(r"from src\.models import[^\n]+", app_source)
        assert any("Rede" in m for m in matches), (
            "The 'from src.models import ...' line does not include Rede."
        )

    def test_no_use_container_width_in_app(self):
        """src/app.py must not use the deprecated use_container_width parameter."""
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "use_container_width" not in app_source, (
            "Deprecated 'use_container_width' parameter found in src/app.py. "
            "Replace it with width='stretch' or width='content'."
        )
