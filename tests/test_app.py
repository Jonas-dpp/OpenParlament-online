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


class TestTOPAnalysePage:
    """Tests for the top-analyse dashboard page, including the NLP-score fallback path."""

    def _seed_top_data_no_nlp(self) -> None:
        """Seed TOPs with un-scored Zwischenrufe (sentiment_score=None).

        The autouse ``_isolated_db`` fixture has already set ``OPENPARLAMENT_DB_URL``
        and reset the engine cache, so we just init tables and insert rows.
        At least 3 reden per TOP are inserted so the default ``min_reden=3``
        slider does not filter them out.
        """
        from src.database import init_db, get_session
        from src.models import Sitzung, Redner, Rede, Zwischenruf
        from datetime import date

        init_db()
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=1,
                              datum=date(2022, 1, 1), gesamtwortzahl=500)
            redner = Redner(vorname="Anna", nachname="Test", fraktion="SPD")
            reden = [
                Rede(
                    sitzung=sitzung, redner=redner, text=f"Rede {i}",
                    tagesordnungspunkt="TOP 1 – Klimaschutz", wortanzahl=5,
                )
                for i in range(4)
            ]
            # Un-scored interjections on each rede (NLP pipeline not yet run)
            zwischenrufe = [
                Zwischenruf(rede=r, text="Widerspruch", fraktion="AfD",
                            sentiment_score=None, kategorie="Widerspruch")
                for r in reden
            ]
            s.add_all([sitzung, redner] + reden + zwischenrufe)

    def _make_render_fn(self):
        """Return a minimal Streamlit function that exercises render_top_analyse's critical paths."""

        def _render():
            import streamlit as st
            import plotly.express as px
            from src.analytics import TOPAnalyse
            from src.database import get_session

            top_n = st.slider("Top N Tagesordnungspunkte", 5, 30, 15, key="top_n_top")
            min_reden = st.slider("Mindestanzahl Reden", 1, 20, 3, key="min_reden_top")

            with get_session() as session:
                ta = TOPAnalyse(session)
                df = ta.aggression_by_top(n=top_n, wahlperiode=20, min_reden=min_reden)
                df_kat = ta.kategorie_by_top(n=top_n, wahlperiode=20, min_reden=min_reden)

            if df.empty:
                st.info(
                    "Keine Tagesordnungspunkt-Daten vorhanden. "
                    "Bitte zuerst Protokolle mit dem Scraper importieren."
                )
            else:
                has_nlp_scores = df["avg_aggression"].notna().any()
                if has_nlp_scores:
                    fig = px.bar(
                        df, x="avg_aggression", y="tagesordnungspunkt", orientation="h",
                        color="anteil_negativ_pct", color_continuous_scale="Reds",
                        title=f"Top {top_n} Reizthemen nach Aggressions-Score",
                    )
                else:
                    st.info(
                        "ℹ️ Keine Sentiment-Scores für Zwischenrufe vorhanden. "
                        "Bitte die NLP-Pipeline ausführen."
                    )
                    fig = px.bar(
                        df.sort_values("gesamt_zwischenrufe", ascending=False),
                        x="gesamt_zwischenrufe", y="tagesordnungspunkt", orientation="h",
                        title=f"Top {top_n} Tagesordnungspunkte nach Anzahl Zwischenrufe",
                    )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig)

            if not df_kat.empty:
                st.subheader("Kategorie-Verteilung")
                st.plotly_chart(px.bar(
                    df_kat, x="tagesordnungspunkt", y="anzahl", color="kategorie",
                    barmode="stack", title="Reaktions-Kategorien nach TOP",
                ))

        return _render

    def test_top_analyse_fallback_chart_when_no_nlp_scores(self):
        """Fallback chart path must render without exception when TOPs exist but are un-scored."""
        from streamlit.testing.v1 import AppTest

        self._seed_top_data_no_nlp()

        at = AppTest.from_function(self._make_render_fn(), default_timeout=30)
        at.run()
        assert not at.exception, (
            f"render_top_analyse raised an exception with un-scored data: {at.exception}"
        )
        info_texts = [el.value for el in at.info]
        assert any("NLP" in t for t in info_texts), (
            "Expected an info banner mentioning NLP when sentiment scores are absent; "
            f"got info messages: {info_texts}"
        )

    def test_top_analyse_with_nlp_scores_renders_aggression_chart(self):
        """Aggression-score bar chart path must render without exception."""
        from streamlit.testing.v1 import AppTest
        from src.database import init_db, get_session
        from src.models import Sitzung, Redner, Rede, Zwischenruf
        from datetime import date

        init_db()
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=2,
                              datum=date(2022, 2, 1), gesamtwortzahl=200)
            redner = Redner(vorname="Hans", nachname="Scored", fraktion="CDU/CSU")
            reden = [
                Rede(
                    sitzung=sitzung, redner=redner, text=f"Rede {i}",
                    tagesordnungspunkt="TOP 2 – Wirtschaft", wortanzahl=4,
                )
                for i in range(4)
            ]
            zwischenrufe = [
                Zwischenruf(rede=r, text="Widerspruch", fraktion="AfD",
                            sentiment_score=-0.8, kategorie="Widerspruch")
                for r in reden
            ]
            s.add_all([sitzung, redner] + reden + zwischenrufe)

        at = AppTest.from_function(self._make_render_fn(), default_timeout=30)
        at.run()
        assert not at.exception, (
            f"render_top_analyse raised an exception with scored data: {at.exception}"
        )
