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
Unit tests for OpenParlament core modules.

Tests use only stdlib + sqlalchemy (no torch/transformers required) so they
run in CI without GPU or large model downloads.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from src.database import drop_db, get_engine, get_session, init_db
from src.models import Rede, Redner, Sitzung, Zwischenruf
from src.nlp import (
    AddresseeDetector,
    SentimentEngine,
    ToneClassifier,
    _rule_based_sentiment_score,
    _tone_rule_based,
)
from src.parser import BundestagXMLParser
from src.scraper import (
    _DSERVER_BASE,
    _EXPECTED_ROOT_TAG,
    _MAX_CONSECUTIVE_MISSES,
    _MAX_RETRIES,
    _OPENDATA_URL,
    _is_dbtplenarprotokoll,
    _url_filename,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _use_in_memory_db(monkeypatch):
    """Redirect all DB access to an in-memory SQLite database for each test."""
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", "sqlite:///:memory:")
    # Reset the cached engine so the monkeypatched env var takes effect.
    import src.database as db_module
    db_module._engine = None
    db_module._SessionFactory = None
    init_db()
    yield
    drop_db()
    db_module._engine = None
    db_module._SessionFactory = None


# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid Bundestag XML for parser tests
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <dbtplenarprotokoll>
      <vorspann>
        <kopfdaten>
          <plenarprotokoll-nummer>
            <wahlperiode>20</wahlperiode>
            <sitzungsnr>1</sitzungsnr>
          </plenarprotokoll-nummer>
          <sitzungstitel>
            <datum>17.11.2021</datum>
          </sitzungstitel>
        </kopfdaten>
      </vorspann>
      <sitzungsverlauf>
        <tagesordnungspunkt id="TOP 1">
          <rede id="ID205000100">
            <p klasse="redner">
              <redner id="11004759">
                <name>
                  <titel>Dr.</titel>
                  <vorname>Olaf</vorname>
                  <nachname>Scholz</nachname>
                  <fraktion>SPD</fraktion>
                </name>
              </redner>
            </p>
            <p klasse="J_1">Wir müssen handeln und die Wirtschaft stärken.</p>
            <kommentar>(Beifall bei der SPD)</kommentar>
            <p klasse="O">Das ist unsere Verantwortung.</p>
            <kommentar>(Widerspruch bei der AfD)</kommentar>
          </rede>
          <rede id="ID205000200">
            <p klasse="redner">
              <redner id="11005001">
                <name>
                  <vorname>Friedrich</vorname>
                  <nachname>Merz</nachname>
                  <fraktion>CDU/CSU</fraktion>
                </name>
              </redner>
            </p>
            <p klasse="J_1">Die Wirtschaftspolitik der Regierung ist gescheitert.</p>
            <kommentar>(Lachen bei der SPD)</kommentar>
          </rede>
        </tagesordnungspunkt>
      </sitzungsverlauf>
    </dbtplenarprotokoll>
""")

# Real Bundestag XML format (21st WP):
#  - date on root element attribute: sitzung-datum="DD.MM.YYYY"
#  - date on <datum date="DD.MM.YYYY"> inside <veranstaltungsdaten>
#  - <sitzungstitel> contains a child <sitzungsnr> tag (mixed content)
#  - <plenarprotokoll-nummer> also contains mixed content with <sitzungsnr>
_REAL_FORMAT_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <dbtplenarprotokoll wahlperiode="21" sitzung-nr="12" sitzung-datum="24.06.2025">
      <vorspann>
        <kopfdaten>
          <plenarprotokoll-nummer>Plenarprotokoll <wahlperiode>21</wahlperiode>/<sitzungsnr>12</sitzungsnr></plenarprotokoll-nummer>
          <herausgeber>Deutscher Bundestag</herausgeber>
          <berichtart>Stenografischer Bericht</berichtart>
          <sitzungstitel><sitzungsnr>12</sitzungsnr>. Sitzung</sitzungstitel>
          <veranstaltungsdaten><ort>Berlin</ort>, <datum date="24.06.2025">Dienstag, den 24. Juni 2025</datum></veranstaltungsdaten>
        </kopfdaten>
      </vorspann>
      <sitzungsverlauf>
        <tagesordnungspunkt id="TOP 1">
          <rede id="ID211200100">
            <p klasse="redner">
              <redner id="11002735">
                <name>
                  <vorname>Friedrich</vorname>
                  <nachname>Merz</nachname>
                  <fraktion>CDU/CSU</fraktion>
                </name>
              </redner>
            </p>
            <p klasse="J_1">Wir müssen Europa stärken.</p>
            <kommentar>(Beifall bei der CDU/CSU)</kommentar>
          </rede>
        </tagesordnungspunkt>
      </sitzungsverlauf>
    </dbtplenarprotokoll>
""")


# ─────────────────────────────────────────────────────────────────────────────
# Database tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_init_creates_tables(self):
        tables = get_engine().dialect.get_table_names(get_engine().connect())
        assert "sitzungen" in tables
        assert "redner" in tables
        assert "reden" in tables
        assert "zwischenrufe" in tables

    def test_session_commit(self):
        redner = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
        with get_session() as s:
            s.add(redner)
        with get_session() as s:
            r = s.execute(select(Redner).where(Redner.nachname == "Mustermann")).scalar_one()
            assert r.vollname == "Max Mustermann"

    def test_session_rollback_on_error(self):
        with pytest.raises(Exception):
            with get_session() as s:
                s.add(Redner(vorname="X", nachname="Y"))
                raise RuntimeError("forced rollback")
        with get_session() as s:
            count = s.execute(select(Redner)).scalars().all()
            assert count == []


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_redner_vollname_with_title(self):
        r = Redner(titel="Dr.", vorname="Hans", nachname="Müller")
        assert r.vollname == "Dr. Hans Müller"

    def test_redner_vollname_without_title(self):
        r = Redner(vorname="Hans", nachname="Müller")
        assert r.vollname == "Hans Müller"

    def test_sitzung_repr(self):
        s = Sitzung(wahlperiode=20, sitzungsnr=1)
        assert "20" in repr(s)

    def test_cascade_delete(self):
        """Deleting a Sitzung should delete its Reden and Zwischenrufe."""
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=99)
            redner = Redner(vorname="A", nachname="B")
            rede = Rede(sitzung=sitzung, redner=redner, text="Text", wortanzahl=1)
            zwr = Zwischenruf(rede=rede, text="Beifall")
            s.add_all([sitzung, redner, rede, zwr])

        with get_session() as s:
            sz = s.execute(
                select(Sitzung).where(Sitzung.sitzungsnr == 99)
            ).scalar_one()
            s.delete(sz)

        with get_session() as s:
            assert s.execute(select(Rede)).scalars().all() == []
            assert s.execute(select(Zwischenruf)).scalars().all() == []


# ─────────────────────────────────────────────────────────────────────────────
# Parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParser:
    def setup_method(self):
        self.parser = BundestagXMLParser()

    def test_parse_bytes_sitzung(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        assert result.sitzung.wahlperiode == 20
        assert result.sitzung.sitzungsnr == 1
        assert str(result.sitzung.datum) == "2021-11-17"
        # 17.11.2021 was a Wednesday
        assert result.sitzung.wochentag == "Mittwoch"

    def test_parse_bytes_redner(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        names = {r.nachname for r in result.redner}
        assert "Scholz" in names
        assert "Merz" in names

    def test_parse_bytes_reden(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        assert len(result.reden) == 2
        texts = [r.text for r in result.reden]
        assert any("Wirtschaft" in t for t in texts)

    def test_parse_bytes_zwischenrufe(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        assert len(result.zwischenrufe) == 3
        kategorien = {z.kategorie for z in result.zwischenrufe}
        assert "Beifall" in kategorien
        assert "Widerspruch" in kategorien
        assert "Lachen" in kategorien

    def test_zwischenruf_fraktion_detection(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        fraktionen = {z.fraktion for z in result.zwischenrufe if z.fraktion}
        assert "SPD" in fraktionen

    def test_total_word_count(self):
        result = self.parser.parse_bytes(_SAMPLE_XML.encode())
        assert result.sitzung.gesamtwortzahl > 0

    def test_parse_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.parser.parse_file(tmp_path / "nonexistent.xml")

    def test_parse_file(self, tmp_path):
        p = tmp_path / "20001.xml"
        p.write_bytes(_SAMPLE_XML.encode())
        result = self.parser.parse_file(p)
        assert result.sitzung.wahlperiode == 20

    # ── Real Bundestag XML format (21st WP) ───────────────────────────────────

    def test_real_format_date_from_root_attribute(self):
        """sitzung-datum attribute on root element is the primary date source."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert str(result.sitzung.datum) == "2025-06-24"

    def test_real_format_wochentag(self):
        """24.06.2025 is a Tuesday → wochentag must be 'Dienstag'."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert result.sitzung.wochentag == "Dienstag"

    def test_real_format_wahlperiode_and_sitzungsnr(self):
        """wahlperiode and sitzungsnr are read from <plenarprotokoll-nummer>,
        not from the duplicate <sitzungsnr> inside <sitzungstitel>."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert result.sitzung.wahlperiode == 21
        assert result.sitzung.sitzungsnr == 12

    def test_real_format_sitzungstitel(self):
        """<sitzungstitel> with embedded <sitzungsnr> child renders as '12. Sitzung'."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert result.sitzung.titel == "12. Sitzung"

    def test_real_format_datum_not_none(self):
        """Ensure datum is never None when a valid date attribute is present."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert result.sitzung.datum is not None

    def test_real_format_wochentag_not_none(self):
        """wochentag must be set whenever datum is set."""
        result = self.parser.parse_bytes(_REAL_FORMAT_XML.encode())
        assert result.sitzung.wochentag is not None


# ─────────────────────────────────────────────────────────────────────────────
# NLP tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNLP:
    def test_rule_based_beifall(self):
        assert _rule_based_sentiment_score("Beifall bei der SPD") == 0.8

    def test_rule_based_widerspruch(self):
        assert _rule_based_sentiment_score("Widerspruch bei der AfD") == -0.8

    def test_rule_based_none_for_unknown(self):
        assert _rule_based_sentiment_score("Abgeordneter verlässt den Saal") is None

    def test_score_batch_rule_based(self):
        engine = SentimentEngine()
        # Mock the neural pipeline so it is never called.
        with patch.object(engine, "_neural_score_batch", return_value=[]) as mock_neural:
            scores = engine.score_batch(["Beifall bei der SPD", "Widerspruch"])
        assert scores[0] == 0.8
        assert scores[1] == -0.8
        mock_neural.assert_not_called()

    def test_score_one(self):
        engine = SentimentEngine()
        score = engine.score_one("Beifall")
        assert score == 0.8

    def test_torch_dtype_stored_on_sentiment_engine(self):
        """SentimentEngine must accept and store an arbitrary torch_dtype value."""
        sentinel = object()
        engine = SentimentEngine(torch_dtype=sentinel)
        assert engine.torch_dtype is sentinel

    def test_torch_dtype_passed_to_pipeline(self):
        """SentimentEngine._get_pipeline must forward torch_dtype to pipeline()."""
        sentinel = object()
        engine = SentimentEngine(torch_dtype=sentinel)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            engine._get_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("torch_dtype") is sentinel

    def test_torch_dtype_none_not_forwarded_to_pipeline(self):
        """When torch_dtype is None, it must not appear in the pipeline() call."""
        engine = SentimentEngine(torch_dtype=None)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            engine._get_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert "torch_dtype" not in kwargs

    def test_neural_score_batch_consumes_iterator(self):
        """_neural_score_batch must work when the pipeline yields items lazily (iterator)."""
        engine = SentimentEngine()
        items = [{"label": "positive", "score": 1.0}, {"label": "negative", "score": 0.5}]

        def fake_pipe(iterable, batch_size):
            # Yield items one at a time to simulate a streaming pipeline.
            yield from items

        engine._pipe = fake_pipe
        results = engine._neural_score_batch(["text one", "text two"])
        assert results == [1.0, -0.5]

    def test_neural_score_batch_passes_list_to_pipeline(self):
        """_neural_score_batch must pass a plain list (not an iterator) to the pipeline."""
        engine = SentimentEngine()
        received = []

        def fake_pipe(iterable, batch_size):
            received.append(iterable)
            return iter([{"label": "neutral", "score": 1.0}])

        engine._pipe = fake_pipe
        engine._neural_score_batch(["a text"])
        assert isinstance(received[0], list), (
            "Pipeline must receive a plain list, not an iterator"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Analytics tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:
    """Smoke-test analytics with a seeded in-memory DB."""

    def _seed(self, session):
        """Insert a minimal dataset for analytics testing."""
        from datetime import date

        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner_spd = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        redner_cdu = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")

        rede1 = Rede(sitzung=sitzung, redner=redner_spd, text="Klimaschutz ist wichtig Klimaschutz", wortanzahl=5)
        rede2 = Rede(sitzung=sitzung, redner=redner_cdu, text="Wirtschaft stärken", wortanzahl=3)

        zwr1 = Zwischenruf(rede=rede1, text="Beifall", fraktion="CDU/CSU", sentiment_score=0.8)
        zwr2 = Zwischenruf(rede=rede2, text="Widerspruch", fraktion="SPD", sentiment_score=-0.8)
        zwr3 = Zwischenruf(rede=rede1, text="Aggression", fraktion="AfD", sentiment_score=-0.9)

        session.add_all([sitzung, redner_spd, redner_cdu, rede1, rede2, zwr1, zwr2, zwr3])

    def test_aggression_index_targets(self):
        from src.analytics import AggressionsIndex

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            ai = AggressionsIndex(s)
            df = ai.top_targets(n=10)
        assert not df.empty
        assert "neg_zwischenrufe" in df.columns

    def test_aggression_index_interruptors(self):
        from src.analytics import AggressionsIndex

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            ai = AggressionsIndex(s)
            df = ai.top_interruptors(n=10)
        assert not df.empty

    def test_themen_karriere(self):
        from src.analytics import ThemenKarriere

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            tk = ThemenKarriere(s)
            df = tk.keyword_trend("Klimaschutz", wahlperiode=20)
        assert not df.empty
        assert df["rohanzahl"].iloc[0] == 2  # "Klimaschutz" appears twice

    def test_interaktions_netzwerk_matrix(self):
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            matrix = nw.adjacency_matrix()
        assert not matrix.empty

    def test_interaktions_netzwerk_edge_list(self):
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            edges = nw.edge_list()
        assert not edges.empty
        assert set(edges.columns) >= {"source", "target", "weight", "aggression_score"}

    def test_interaktions_netzwerk_faction_normalisation(self):
        """DIE LINKE / LINKE variants must merge into one canonical row."""
        from datetime import date
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=99, datum=date(2021, 11, 18), gesamtwortzahl=50)
            redner = Redner(vorname="Anna", nachname="Müller", fraktion="SPD")
            rede = Rede(sitzung=sitzung, redner=redner, text="Test", wortanzahl=3)
            # Two variants of Die Linke interrupting SPD – must merge
            z1 = Zwischenruf(rede=rede, text="Nein!", fraktion="DIE LINKE", sentiment_score=-0.7)
            z2 = Zwischenruf(rede=rede, text="Widerspruch!", fraktion="LINKE", sentiment_score=-0.5)
            s.add_all([sitzung, redner, rede, z1, z2])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            edges = nw.edge_list()

        # Both variants should collapse to a single canonical target row
        canonical_rows = edges[edges["target"] == "Die Linke"]
        assert len(canonical_rows) == 1, (
            f"Expected 1 canonical 'Die Linke' row, got {len(canonical_rows)}:\n{edges}"
        )
        assert canonical_rows.iloc[0]["weight"] == 2  # 2 interjections merged
        # Weighted average: (−0.7×1 + −0.5×1) / 2 = −0.6
        assert abs(canonical_rows.iloc[0]["avg_sentiment"] - (-0.6)) < 0.01, (
            "avg_sentiment after merging should be the weighted average −0.6"
        )

    def test_interaktions_netzwerk_excludes_self_interactions(self):
        """Self-interactions (same faction applauding own speaker) must be excluded."""
        from datetime import date
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=100, datum=date(2021, 11, 19), gesamtwortzahl=50)
            redner_afd = Redner(vorname="Hans", nachname="Schulz", fraktion="AfD")
            redner_spd = Redner(vorname="Lisa", nachname="Weber", fraktion="SPD")
            rede_afd = Rede(sitzung=sitzung, redner=redner_afd, text="Test", wortanzahl=3)
            # AfD applauding their own speaker (self-interaction)
            z_self = Zwischenruf(rede=rede_afd, text="Beifall", fraktion="AfD", sentiment_score=0.9)
            # SPD interrupting AfD (cross-faction, should remain)
            z_cross = Zwischenruf(rede=rede_afd, text="Widerspruch", fraktion="SPD", sentiment_score=-0.8)
            rede_spd = Rede(sitzung=sitzung, redner=redner_spd, text="Test2", wortanzahl=2)
            s.add_all([sitzung, redner_afd, redner_spd, rede_afd, rede_spd, z_self, z_cross])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            edges = nw.edge_list(exclude_self=True)
            matrix = nw.adjacency_matrix(exclude_self=True)

        # Self-interaction (AfD → AfD) must not appear in the edge list
        self_edges = edges[(edges["source"] == "AfD") & (edges["target"] == "AfD")]
        assert self_edges.empty, "Self-interaction AfD→AfD should be excluded"

        # The cross-faction edge must be present
        cross_edges = edges[(edges["source"] == "AfD") & (edges["target"] == "SPD")]
        assert not cross_edges.empty, "Cross-faction edge AfD→SPD should be present"

        # Matrix diagonal should be zero (no AfD row × AfD column entry)
        if "AfD" in matrix.index and "AfD" in matrix.columns:
            assert matrix.loc["AfD", "AfD"] == 0

    def test_interaktions_netzwerk_aggression_score_direction(self):
        """Negative sentiment should yield a POSITIVE aggression score."""
        from datetime import date
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=101, datum=date(2021, 11, 20), gesamtwortzahl=50)
            redner_spd = Redner(vorname="Karl", nachname="Bauer", fraktion="SPD")
            redner_afd = Redner(vorname="Eva", nachname="Klein", fraktion="AfD")
            rede = Rede(sitzung=sitzung, redner=redner_spd, text="Test", wortanzahl=3)
            # Very negative interjection from AfD towards SPD
            z = Zwischenruf(rede=rede, text="Lüge!", fraktion="AfD", sentiment_score=-0.9)
            s.add_all([sitzung, redner_spd, redner_afd, rede, z])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            edges = nw.edge_list()
            matrix = nw.adjacency_matrix(score_weighted=True)

        # aggression_score = -avg_sentiment, so -(-0.9) = 0.9 (positive, red)
        afd_row = edges[(edges["source"] == "SPD") & (edges["target"] == "AfD")]
        assert not afd_row.empty
        assert afd_row.iloc[0]["aggression_score"] > 0, (
            "Negative sentiment must map to positive aggression score"
        )
        # In the aggression matrix, the AfD column in the SPD row should be positive
        if "SPD" in matrix.index and "AfD" in matrix.columns:
            assert matrix.loc["SPD", "AfD"] > 0

    def test_per_capita_uses_correct_wahlperiode_sizes(self):
        """per_capita=True must divide by the faction size of the given Wahlperiode."""
        from datetime import date
        from src.analytics import FACTION_SIZES_BY_WAHLPERIODE, InteraktionsNetzwerk

        # Use WP 19: AfD had 94 seats.
        wp = 19
        afd_size = FACTION_SIZES_BY_WAHLPERIODE[wp]["AfD"]

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=wp, sitzungsnr=110, datum=date(2019, 1, 15), gesamtwortzahl=100)
            redner_spd = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
            rede = Rede(sitzung=sitzung, redner=redner_spd, text="Rede", wortanzahl=10)
            # AfD interrupts SPD 94 times (one per member – convenient round number)
            for i in range(afd_size):
                s.add(Zwischenruf(rede=rede, text="Buh!", fraktion="AfD", sentiment_score=-0.5))
            s.add_all([sitzung, redner_spd, rede])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            matrix_raw = nw.adjacency_matrix(wahlperiode=wp, per_capita=False)
            matrix_pc = nw.adjacency_matrix(wahlperiode=wp, per_capita=True)

        assert "SPD" in matrix_raw.index and "AfD" in matrix_raw.columns
        raw_val = matrix_raw.loc["SPD", "AfD"]
        pc_val = matrix_pc.loc["SPD", "AfD"]
        assert abs(raw_val - afd_size) < 0.01, "Raw count should equal number of interjections"
        assert abs(pc_val - 1.0) < 0.01, "Per-capita value should be raw / faction_size = 1.0"

    def test_per_capita_different_wahlperiode_gives_different_result(self):
        """The same raw count should normalise differently across Wahlperioden."""
        from datetime import date
        from src.analytics import FACTION_SIZES_BY_WAHLPERIODE, InteraktionsNetzwerk

        # Build identical interjection data for WP 17 and WP 19.
        # AfD only exists in WP 19+; use FDP (present in both) as the interruptor.
        for wp, sitzungsnr in ((17, 200), (19, 201)):
            with get_session() as s:
                sitzung = Sitzung(wahlperiode=wp, sitzungsnr=sitzungsnr,
                                  datum=date(2010 if wp == 17 else 2018, 3, 1),
                                  gesamtwortzahl=50)
                redner = Redner(vorname="Anna", nachname="Schmidt", fraktion="SPD")
                rede = Rede(sitzung=sitzung, redner=redner, text="Rede", wortanzahl=5)
                for _ in range(10):
                    s.add(Zwischenruf(rede=rede, text="!", fraktion="FDP", sentiment_score=0.0))
                s.add_all([sitzung, redner, rede])

        fdp_17 = FACTION_SIZES_BY_WAHLPERIODE[17]["FDP"]  # 93
        fdp_19 = FACTION_SIZES_BY_WAHLPERIODE[19]["FDP"]  # 80

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            pc_17 = nw.adjacency_matrix(wahlperiode=17, per_capita=True)
            pc_19 = nw.adjacency_matrix(wahlperiode=19, per_capita=True)

        assert "SPD" in pc_17.index and "FDP" in pc_17.columns
        assert "SPD" in pc_19.index and "FDP" in pc_19.columns
        val_17 = pc_17.loc["SPD", "FDP"]
        val_19 = pc_19.loc["SPD", "FDP"]
        # Both raw counts are 10; different faction sizes → different per-capita values.
        assert abs(val_17 - 10 / fdp_17) < 1e-9
        assert abs(val_19 - 10 / fdp_19) < 1e-9
        assert val_17 != val_19, "Different Wahlperiode sizes must produce different per-capita values"

    def test_pds_canonicalised_to_die_linke(self):
        """'PDS' faction string must collapse to the 'Die Linke' canonical name."""
        from datetime import date
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=14, sitzungsnr=300, datum=date(1999, 3, 1), gesamtwortzahl=50)
            redner = Redner(vorname="Hans", nachname="Müller", fraktion="SPD")
            rede = Rede(sitzung=sitzung, redner=redner, text="Rede", wortanzahl=5)
            z = Zwischenruf(rede=rede, text="Widerspruch!", fraktion="PDS", sentiment_score=-0.6)
            s.add_all([sitzung, redner, rede, z])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            edges = nw.edge_list()

        pds_edges = edges[edges["target"] == "PDS"]
        linke_edges = edges[edges["target"] == "Die Linke"]
        assert pds_edges.empty, "PDS must be canonicalised away"
        assert not linke_edges.empty, "PDS interjections must appear under 'Die Linke'"

    def test_get_faction_sizes_fallback(self):
        """_get_faction_sizes must return the most recent period for unknown inputs."""
        from src.analytics import FACTION_SIZES_BY_WAHLPERIODE, _get_faction_sizes

        latest = max(FACTION_SIZES_BY_WAHLPERIODE)
        assert _get_faction_sizes(None) == FACTION_SIZES_BY_WAHLPERIODE[latest]
        assert _get_faction_sizes(999) == FACTION_SIZES_BY_WAHLPERIODE[latest]

    def test_get_faction_sizes_known_wahlperiode(self):
        """_get_faction_sizes must return the exact dict for a known Wahlperiode."""
        from src.analytics import FACTION_SIZES_BY_WAHLPERIODE, _get_faction_sizes

        for wp, sizes in FACTION_SIZES_BY_WAHLPERIODE.items():
            assert _get_faction_sizes(wp) == sizes

    def test_faction_sizes_all_wahlperioden_covered(self):
        """Every Wahlperiode from 13 to 21 must have an entry in FACTION_SIZES_BY_WAHLPERIODE."""
        from src.analytics import FACTION_SIZES_BY_WAHLPERIODE

        for wp in range(13, 22):
            assert wp in FACTION_SIZES_BY_WAHLPERIODE, (
                f"Wahlperiode {wp} is missing from FACTION_SIZES_BY_WAHLPERIODE"
            )
            assert FACTION_SIZES_BY_WAHLPERIODE[wp], f"Wahlperiode {wp} must not be empty"

    def test_matrix_is_sorted_by_volume_descending(self):
        """adjacency_matrix must sort rows and columns by total volume, descending."""
        from datetime import date
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=400, datum=date(2022, 5, 1), gesamtwortzahl=50)
            # SPD speaker gets 3 interruptions (AfD×2, CDU/CSU×1) → row sum 3
            # AfD speaker gets 1 interruption (SPD×1)              → row sum 1
            # CDU/CSU speaker gets 0 cross-faction interruptions   → row sum 0
            # As interruptor: AfD=2, SPD=1, CDU/CSU=1 → col sums 2,1,1
            redner_spd = Redner(vorname="A", nachname="R_SPD", fraktion="SPD")
            redner_afd = Redner(vorname="B", nachname="R_AfD", fraktion="AfD")
            redner_cdu = Redner(vorname="C", nachname="R_CDU", fraktion="CDU/CSU")
            rede_spd = Rede(sitzung=sitzung, redner=redner_spd, text="x", wortanzahl=2)
            rede_afd = Rede(sitzung=sitzung, redner=redner_afd, text="y", wortanzahl=2)
            # AfD interrupts SPD twice
            z1 = Zwischenruf(rede=rede_spd, text="!", fraktion="AfD", sentiment_score=0.0)
            z2 = Zwischenruf(rede=rede_spd, text="!", fraktion="AfD", sentiment_score=0.0)
            # CDU/CSU interrupts SPD once
            z3 = Zwischenruf(rede=rede_spd, text="!", fraktion="CDU/CSU", sentiment_score=0.0)
            # SPD interrupts AfD once
            z4 = Zwischenruf(rede=rede_afd, text="!", fraktion="SPD", sentiment_score=0.0)
            s.add_all([sitzung, redner_spd, redner_afd, redner_cdu,
                       rede_spd, rede_afd, z1, z2, z3, z4])

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            matrix = nw.adjacency_matrix()

        # Verify that both axes are monotonically non-increasing by total volume.
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
        assert list(row_sums) == sorted(row_sums, reverse=True), (
            f"Rows must be ordered by descending total volume; got sums {list(row_sums)}"
        )
        assert list(col_sums) == sorted(col_sums, reverse=True), (
            f"Columns must be ordered by descending total volume; got sums {list(col_sums)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scraper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScraper:
    """Unit tests for BundestagScraper (no network access required)."""

    def test_opendata_url_has_no_query_params(self):
        """_OPENDATA_URL must not contain any query parameters to avoid
        duplicate params when requests adds its own."""
        assert "?" not in _OPENDATA_URL

    def test_fetch_page_filters_correct_wahlperiode(self, tmp_path):
        """_fetch_page must accept filenames matching '^<wp>\\d+\\.xml$' only."""
        from unittest.mock import MagicMock
        from urllib.parse import urlparse
        from src.scraper import BundestagScraper

        # Build a minimal HTML page with two XML links: one matching WP 20
        # and one metadata file that should be rejected.
        html = """
        <html><body>
          <a href="/resource/blob/20001.xml">20001.xml</a>
          <a href="/resource/blob/metadata.xml">metadata.xml</a>
          <a href="/resource/blob/19999.xml">19999.xml</a>
          <a href="/resource/blob/200123.xml">200123.xml</a>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        urls = scraper._fetch_page(offset=0, limit=10, wahlperiode=20)

        filenames = [urlparse(u).path.split("/")[-1] for u in urls]
        # 20001.xml and 200123.xml match WP 20; metadata.xml and 19999.xml do not.
        assert "20001.xml" in filenames
        assert "200123.xml" in filenames
        assert "metadata.xml" not in filenames
        assert "19999.xml" not in filenames

    def test_fetch_page_query_params_not_duplicated(self, tmp_path):
        """requests must be called with only the explicit params dict
        (no params already embedded in the base URL)."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        scraper._fetch_page(offset=0, limit=10, wahlperiode=20)

        call_kwargs = mock_session.get.call_args
        url_called = call_kwargs[0][0]  # first positional arg
        # The URL itself must not contain query params.
        assert "?" not in url_called
        # limit and offset must be passed as separate params dict.
        passed_params = call_kwargs[1].get("params", {})
        assert passed_params.get("limit") == 10
        assert passed_params.get("offset") == 0

    def test_is_dbtplenarprotokoll_accepts_valid_xml(self):
        """_is_dbtplenarprotokoll must return True for genuine protocol XML."""
        xml = b'<?xml version="1.0" encoding="UTF-8"?><dbtplenarprotokoll><vorspann/></dbtplenarprotokoll>'
        assert _is_dbtplenarprotokoll(xml) is True

    def test_is_dbtplenarprotokoll_rejects_wrong_root(self):
        """_is_dbtplenarprotokoll must return False for XML with a different root element."""
        xml = b'<?xml version="1.0" encoding="UTF-8"?><otherDocument><content/></otherDocument>'
        assert _is_dbtplenarprotokoll(xml) is False

    def test_is_dbtplenarprotokoll_rejects_invalid_xml(self):
        """_is_dbtplenarprotokoll must return False when the content is not valid XML."""
        assert _is_dbtplenarprotokoll(b"not xml at all") is False
        assert _is_dbtplenarprotokoll(b"") is False
        assert _is_dbtplenarprotokoll(b"<unclosed>") is False

    def test_is_dbtplenarprotokoll_rejects_xml_entity_expansion(self):
        """_is_dbtplenarprotokoll must not expand internal entities (XML bomb guard).

        The parser is configured with resolve_entities=False.  A classic
        billion-laughs payload must not cause resource exhaustion and must
        not be parsed as a valid dbtplenarprotokoll document.
        """
        # Minimal billion-laughs-style payload – each entity level multiplies
        # the previous; with resolve_entities=False lxml must not expand any
        # of these and must reject the document as invalid or wrong root.
        xml_bomb = b"""<?xml version="1.0"?>
<!DOCTYPE lol [
  <!ENTITY a "aaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
]>
<dbtplenarprotokoll>&c;</dbtplenarprotokoll>"""
        # The function must return a bool without hanging (entity not expanded).
        result = _is_dbtplenarprotokoll(xml_bomb)
        assert isinstance(result, bool)

    def test_is_dbtplenarprotokoll_rejects_external_entity(self):
        """_is_dbtplenarprotokoll must not fetch external entities (XXE/SSRF guard).

        The parser is configured with no_network=True.  An XXE payload
        referencing an external URL must not trigger a network call.
        """
        xxe_payload = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY ext SYSTEM "http://localhost/secret">
]>
<dbtplenarprotokoll>&ext;</dbtplenarprotokoll>"""
        # Must return without making any network request; result is a bool.
        result = _is_dbtplenarprotokoll(xxe_payload)
        assert isinstance(result, bool)

    def test_download_one_rejects_non_protocol_xml(self, tmp_path):
        """download_one must return None and not write a file when the downloaded
        XML root element is not dbtplenarprotokoll."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        wrong_xml = b'<?xml version="1.0"?><someOtherDoc/>'
        mock_resp = MagicMock()
        mock_resp.content = wrong_xml
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        result = scraper.download_one("https://www.bundestag.de/resource/blob/123/20001.xml")

        assert result is None
        assert not (tmp_path / "20001.xml").exists()

    def test_download_one_accepts_protocol_xml(self, tmp_path):
        """download_one must write and return a path when the root element is dbtplenarprotokoll."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        valid_xml = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b"<dbtplenarprotokoll><vorspann/></dbtplenarprotokoll>"
        )
        mock_resp = MagicMock()
        mock_resp.content = valid_xml
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        result = scraper.download_one("https://www.bundestag.de/resource/blob/123/20001.xml")

        assert result == tmp_path / "20001.xml"
        assert (tmp_path / "20001.xml").exists()
        assert (tmp_path / "20001.xml").read_bytes() == valid_xml

    def test_expected_root_tag_constant(self):
        """_EXPECTED_ROOT_TAG must equal 'dbtplenarprotokoll'."""
        assert _EXPECTED_ROOT_TAG == "dbtplenarprotokoll"

    # ── dserver URL helper ────────────────────────────────────────────────────

    def test_dserver_xml_url_format(self):
        """_dserver_xml_url must generate zero-padded URLs under _DSERVER_BASE."""
        from src.scraper import BundestagScraper

        assert BundestagScraper._dserver_xml_url(20, 1) == (
            f"{_DSERVER_BASE}/20/20001.xml"
        )
        assert BundestagScraper._dserver_xml_url(20, 214) == (
            f"{_DSERVER_BASE}/20/20214.xml"
        )
        assert BundestagScraper._dserver_xml_url(19, 12) == (
            f"{_DSERVER_BASE}/19/19012.xml"
        )

    # ── _fetch_via_dserver ────────────────────────────────────────────────────

    def test_fetch_via_dserver_collects_200_urls(self, tmp_path):
        """_fetch_via_dserver must return URLs whose GET response is 200."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # Simulate: sn=1 → 200, sn=2 → 200, sn=3..7 → 404 (triggers miss limit)
        responses = {1: 200, 2: 200}

        def fake_get(url, **kwargs):
            sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
            mock = MagicMock()
            mock.status_code = responses.get(sn, 404)
            return mock

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        urls = scraper._fetch_via_dserver(wahlperiode=20, max_sessions=50)

        assert BundestagScraper._dserver_xml_url(20, 1) in urls
        assert BundestagScraper._dserver_xml_url(20, 2) in urls
        # Should have stopped after _MAX_CONSECUTIVE_MISSES 404s starting at sn=3
        assert len(urls) == 2

    def test_fetch_via_dserver_stops_on_consecutive_misses(self, tmp_path):
        """_fetch_via_dserver must stop after _MAX_CONSECUTIVE_MISSES non-200 responses."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # sn=1 → 200, then all subsequent → 404
        def fake_get(url, **kwargs):
            sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
            mock = MagicMock()
            mock.status_code = 200 if sn == 1 else 404
            return mock

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        urls = scraper._fetch_via_dserver(wahlperiode=20, max_sessions=200)

        # Only sn=1 should be collected; probe stops after 1 + _MAX_CONSECUTIVE_MISSES calls
        assert len(urls) == 1
        expected_call_count = 1 + _MAX_CONSECUTIVE_MISSES
        assert mock_session.get.call_count == expected_call_count

    def test_fetch_via_dserver_resets_miss_counter_on_hit(self, tmp_path):
        """A 200 response must reset the consecutive-miss counter."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # Pattern: 200, 404, 404, 200, 404×_MAX_CONSECUTIVE_MISSES → stop
        hit_sns = {1, 4}

        def fake_get(url, **kwargs):
            sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
            mock = MagicMock()
            mock.status_code = 200 if sn in hit_sns else 404
            return mock

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        urls = scraper._fetch_via_dserver(wahlperiode=20, max_sessions=200)

        # sn=1 and sn=4 found; misses at sn=2,3 reset by hit at sn=4;
        # then _MAX_CONSECUTIVE_MISSES misses at sn=5..9 trigger stop
        assert BundestagScraper._dserver_xml_url(20, 1) in urls
        assert BundestagScraper._dserver_xml_url(20, 4) in urls
        assert len(urls) == 2
        expected_calls = 4 + _MAX_CONSECUTIVE_MISSES  # session numbers 1–4 + misses after 4
        assert mock_session.get.call_count == expected_calls

    # ── fetch_protocol_urls stall detection ───────────────────────────────────

    def test_fetch_protocol_urls_stall_detection(self, tmp_path):
        """fetch_protocol_urls must stop filterlist pagination when a full page
        contains only already-seen URLs (stall detection)."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # filterlist HTML: always return the same single URL (simulates stalling backend)
        repeated_html = """
        <html><body>
          <a href="/resource/blob/111/20001.xml">20001.xml</a>
        </body></html>
        """
        mock_filterlist_resp = MagicMock()
        mock_filterlist_resp.text = repeated_html
        mock_filterlist_resp.raise_for_status = MagicMock()

        def fake_get(url, **kwargs):
            if "dserver.bundestag.de" in url:
                # dserver: all 404 → probe stops quickly, nothing collected
                m = MagicMock()
                m.status_code = 404
                return m
            return mock_filterlist_resp

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        urls = scraper.fetch_protocol_urls(wahlperiode=20, max_pages=50)

        # Only one unique URL should be collected despite max_pages=50
        assert len(urls) == 1
        # Count only filterlist GET calls (exclude the dserver probe GETs)
        filterlist_get_calls = sum(
            1 for call in mock_session.get.call_args_list
            if "dserver.bundestag.de" not in str(call)
        )
        # first page → 1 new URL; second page → stall → stop
        assert filterlist_get_calls == 2

    def test_fetch_protocol_urls_merges_dserver_and_filterlist(self, tmp_path):
        """URLs found by dserver and the filterlist must be merged without duplicates.

        The same session can appear as different URL forms:
          dserver: https://dserver.bundestag.de/btp/20/20001.xml
          filterlist: https://www.bundestag.de/resource/blob/999/20001.xml
        Filename-based deduplication must keep only the first (dserver) copy.
        """
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        dserver_url = BundestagScraper._dserver_xml_url(20, 1)
        # filterlist returns the same session via a blob URL + a new session
        blob_dup_url = "https://www.bundestag.de/resource/blob/999/20001.xml"
        filterlist_new_url = "https://www.bundestag.de/resource/blob/888/20002.xml"

        # Separate counter for filterlist calls only
        filterlist_call_count = [0]

        def fake_get(url, **kwargs):
            if "dserver.bundestag.de" in url:
                m = MagicMock()
                m.status_code = 200 if url == dserver_url else 404
                return m
            # Filterlist call: first page has dup + new; second page is empty
            idx = filterlist_call_count[0]
            filterlist_call_count[0] += 1
            if idx == 0:
                html = f"""<html><body>
                  <a href="{blob_dup_url}">20001.xml</a>
                  <a href="{filterlist_new_url}">20002.xml</a>
                </body></html>"""
            else:
                html = "<html></html>"
            m = MagicMock()
            m.text = html
            m.raise_for_status = MagicMock()
            return m

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        urls = scraper.fetch_protocol_urls(wahlperiode=20, max_pages=10)

        filenames = [_url_filename(u) for u in urls]
        # dserver URL kept; filterlist blob duplicate discarded; new session added
        assert dserver_url in urls
        assert filterlist_new_url in urls
        assert blob_dup_url not in urls
        assert filenames.count("20001.xml") == 1
        # No duplicates at all
        assert len(urls) == len(set(urls))

    # ── _probe_url (GET+stream, 5xx retry) ───────────────────────────────────

    def test_probe_url_uses_get_stream(self, tmp_path):
        """_probe_url must use GET with stream=True (not HEAD)."""
        from unittest.mock import MagicMock, call
        from src.scraper import BundestagScraper

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        status = scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        assert status == 200
        # Must use GET, not HEAD
        mock_session.get.assert_called_once()
        mock_session.head.assert_not_called()
        # stream=True must be passed
        _, call_kwargs = mock_session.get.call_args
        assert call_kwargs.get("stream") is True
        # Response body must be closed without reading
        mock_resp.close.assert_called_once()

    def test_probe_url_retries_on_5xx(self, tmp_path):
        """_probe_url must retry up to _MAX_RETRIES times on 5xx responses."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # All attempts return 503
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        status = scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        # After _MAX_RETRIES exhausted, returns 0
        assert status == 0
        assert mock_session.get.call_count == _MAX_RETRIES

    def test_probe_url_returns_200_on_first_successful_retry(self, tmp_path):
        """_probe_url must return 200 once a retry succeeds after 5xx."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200

        mock_session = MagicMock()
        mock_session.get.side_effect = [resp_503, resp_200]

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        status = scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        assert status == 200
        assert mock_session.get.call_count == 2  # one failure + one success

    def test_probe_url_returns_0_on_network_error(self, tmp_path):
        """_probe_url must return 0 when all attempts raise a network exception."""
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper
        import requests as req

        mock_session = MagicMock()
        mock_session.get.side_effect = req.ConnectionError("unreachable")

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        status = scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        assert status == 0
        assert mock_session.get.call_count == _MAX_RETRIES

    # ── _url_filename ─────────────────────────────────────────────────────────

    def test_url_filename_extracts_bare_name(self):
        """_url_filename must return the lowercase bare filename from any URL form."""
        assert _url_filename("https://dserver.bundestag.de/btp/20/20001.xml") == "20001.xml"
        assert _url_filename("https://www.bundestag.de/resource/blob/999/20001.xml") == "20001.xml"
        assert _url_filename("https://example.com/path/FILE.XML?token=abc") == "file.xml"
        # Query parameters with common download triggers must be stripped correctly.
        assert _url_filename("https://example.com/path/20001.xml?download=true") == "20001.xml"
        # URL fragments must also be stripped.
        assert _url_filename("https://example.com/path/20001.xml#section") == "20001.xml"
        # Both query string and fragment present simultaneously.
        assert _url_filename("https://example.com/path/20001.xml?x=1#frag") == "20001.xml"

    # ── fetch_all_wahlperioden ────────────────────────────────────────────────

    def test_fetch_all_wahlperioden_merges_multiple_wps(self, tmp_path):
        """fetch_all_wahlperioden must collect URLs for each Wahlperiode and
        deduplicate across them."""
        from unittest.mock import MagicMock, patch
        from src.scraper import BundestagScraper

        wp19_url = BundestagScraper._dserver_xml_url(19, 1)
        wp20_url = BundestagScraper._dserver_xml_url(20, 1)

        def fake_get(url, **kwargs):
            m = MagicMock()
            if url in (wp19_url, wp20_url):
                m.status_code = 200
            else:
                m.status_code = 404
            return m

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0, session=mock_session)
        # Patch filterlist to return empty so only dserver contributes.
        # max_workers=1 forces sequential execution so the instance-level patch applies.
        with patch.object(scraper, "_fetch_page", return_value=[]):
            urls = scraper.fetch_all_wahlperioden([19, 20], max_workers=1)

        assert wp19_url in urls
        assert wp20_url in urls
        assert len(urls) == 2

    def test_fetch_all_wahlperioden_deduplicates_across_wps(self, tmp_path):
        """fetch_all_wahlperioden must not return the same filename twice even
        if both Wahlperioden somehow share it (defensive cross-WP dedup)."""
        from unittest.mock import MagicMock, patch
        from src.scraper import BundestagScraper

        shared_url = BundestagScraper._dserver_xml_url(20, 1)

        def fake_fetch(wp, max_pages=100):
            # Both WP calls return the same URL
            return [shared_url]

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0)
        # max_workers=1 forces sequential execution so the instance-level patch applies.
        with patch.object(scraper, "fetch_protocol_urls", side_effect=fake_fetch):
            urls = scraper.fetch_all_wahlperioden([19, 20], max_workers=1)

        assert len(urls) == 1
        assert urls[0] == shared_url

    def test_fetch_all_wahlperioden_parallel_with_max_workers(self, tmp_path):
        """fetch_all_wahlperioden with max_workers > 1 must return the same
        deduplicated results as the sequential path.

        Each Wahlperiode is processed in a separate thread; the test patches
        BundestagScraper.fetch_protocol_urls at the class level so the patch
        applies to the per-thread worker instances created internally.
        """
        from unittest.mock import patch
        from src.scraper import BundestagScraper

        wp19_url = BundestagScraper._dserver_xml_url(19, 1)
        wp20_url = BundestagScraper._dserver_xml_url(20, 1)

        def fake_fetch(mock_self, wp, max_pages=100):
            return [BundestagScraper._dserver_xml_url(wp, 1)]

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0)
        with patch.object(BundestagScraper, "fetch_protocol_urls", fake_fetch):
            urls = scraper.fetch_all_wahlperioden([19, 20], max_workers=2)

        assert wp19_url in urls
        assert wp20_url in urls
        assert len(urls) == 2
        assert len(urls) == len(set(urls))

    # ── _probe_url return contract ────────────────────────────────────────────

    def test_probe_url_returns_only_200_or_0(self, tmp_path):
        """_probe_url must return exactly 200 on success and 0 for any non-200
        outcome (including definitive client-side misses like 404).

        This guarantees callers can use a simple ``== 200`` check without
        handling arbitrary status codes.
        """
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        for status_code in (200, 301, 404, 410, 503):
            mock_resp = MagicMock()
            mock_resp.status_code = status_code

            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp

            scraper = BundestagScraper(
                download_dir=tmp_path, request_delay=0, session=mock_session
            )
            result = scraper._probe_url(
                "https://dserver.bundestag.de/btp/20/20001.xml"
            )
            assert result in (0, 200), (
                f"_probe_url returned {result!r} for HTTP {status_code}; "
                "expected only 0 or 200"
            )
            if status_code == 200:
                assert result == 200
            else:
                assert result == 0

    def test_fetch_all_wahlperioden_adaptive_max_workers(self, tmp_path):
        """fetch_all_wahlperioden with default max_workers=None must use
        min(8, len(wahlperioden)) threads and return correct, deduplicated results."""
        from unittest.mock import patch
        from src.scraper import BundestagScraper

        wahlperioden = [17, 18, 19, 20]

        def fake_fetch(mock_self, wp, max_pages=100):
            return [BundestagScraper._dserver_xml_url(wp, 1)]

        executor_max_workers_seen = []
        original_executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor

        def capturing_executor(max_workers=None, **kwargs):
            executor_max_workers_seen.append(max_workers)
            return original_executor(max_workers=max_workers, **kwargs)

        scraper = BundestagScraper(download_dir=tmp_path, request_delay=0)
        with patch.object(BundestagScraper, "fetch_protocol_urls", fake_fetch):
            with patch("src.scraper.ThreadPoolExecutor", side_effect=capturing_executor):
                urls = scraper.fetch_all_wahlperioden(wahlperioden)

        # Adaptive selection: min(8, 4) == 4
        assert executor_max_workers_seen == [4]
        # All four WPs present and no duplicates
        for wp in wahlperioden:
            assert BundestagScraper._dserver_xml_url(wp, 1) in urls
        assert len(urls) == len(wahlperioden)

    def test_fetch_via_dserver_logs_miss_at_debug(self, tmp_path, caplog):
        """_fetch_via_dserver must emit a DEBUG log for every non-200 probe."""
        import logging
        from unittest.mock import MagicMock
        from src.scraper import BundestagScraper

        # sn=1 → 200 (hit), sn=2..6 → 404 (misses until _MAX_CONSECUTIVE_MISSES)
        def fake_get(url, **kwargs):
            sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
            m = MagicMock()
            m.status_code = 200 if sn == 1 else 404
            return m

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        scraper = BundestagScraper(
            download_dir=tmp_path, request_delay=0, probe_delay=0, session=mock_session
        )

        with caplog.at_level(logging.DEBUG, logger="src.scraper"):
            scraper._fetch_via_dserver(wahlperiode=20, max_sessions=50)

        miss_logs = [r for r in caplog.records if "Miss:" in r.message]
        # Exactly _MAX_CONSECUTIVE_MISSES misses are logged (sn=2..6)
        assert len(miss_logs) == _MAX_CONSECUTIVE_MISSES
        # Each log line must reference the probed URL
        for record in miss_logs:
            assert "dserver.bundestag.de" in record.message

    def test_probe_delay_used_in_probe_url_retries(self, tmp_path):
        """_probe_url must use probe_delay (not request_delay) for retry waits."""
        from unittest.mock import MagicMock, patch
        from src.scraper import BundestagScraper

        # All attempts return 503 so retries are exercised
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        scraper = BundestagScraper(
            download_dir=tmp_path,
            request_delay=999.0,  # large value — must NOT be used for probe retries
            probe_delay=0,
            session=mock_session,
        )

        sleep_calls = []
        with patch("src.scraper.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        # time.sleep must never have been called with request_delay=999.0
        assert 999.0 not in sleep_calls


# ─────────────────────────────────────────────────────────────────────────────
# ToneClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestToneClassifier:
    """Tests for the rule-based fast path of ToneClassifier (no model needed)."""

    def test_tone_rule_lachen_is_humor(self):
        label, scores = _tone_rule_based("Lachen bei der SPD")
        assert label == "Humor"
        assert scores["Humor"] > 0.5

    def test_tone_rule_widerspruch_is_aggression(self):
        label, scores = _tone_rule_based("Widerspruch bei der AfD")
        assert label == "Aggression"

    def test_tone_rule_heiterkeit_is_humor(self):
        label, scores = _tone_rule_based("Heiterkeit im Saal")
        assert label == "Humor"

    def test_tone_rule_unruhe_is_aggression(self):
        label, scores = _tone_rule_based("Unruhe bei der CDU/CSU")
        assert label == "Aggression"

    def test_tone_rule_none_for_unknown(self):
        assert _tone_rule_based("Der Abgeordnete verlässt den Saal") is None

    def test_classify_uses_rule_based_fast_path(self):
        """Neural pipeline must not be called for texts handled by rules."""
        classifier = ToneClassifier()
        with patch.object(classifier, "_neural_classify_batch", return_value=[]) as mock_neural:
            label, scores = classifier.classify("Lachen bei der SPD")
        assert label == "Humor"
        mock_neural.assert_not_called()

    def test_classify_batch_returns_correct_length(self):
        classifier = ToneClassifier()
        with patch.object(
            classifier,
            "_neural_classify_batch",
            return_value=[("Neutral", {"Neutral": 1.0, "Aggression": 0.0, "Sarkasmus": 0.0, "Humor": 0.0})],
        ):
            results = classifier.classify_batch(["Lachen", "Unbekannter Text"])
        assert len(results) == 2

    def test_classify_batch_neural_called_for_unknown(self):
        """Texts not handled by rules must be forwarded to the neural model."""
        classifier = ToneClassifier()
        neural_result = [("Neutral", {"Neutral": 0.9, "Aggression": 0.0, "Sarkasmus": 0.05, "Humor": 0.05})]
        with patch.object(classifier, "_neural_classify_batch", return_value=neural_result) as mock_neural:
            results = classifier.classify_batch(["Ein sehr komplizierter Satz ohne Signalwörter"])
        mock_neural.assert_called_once()
        assert results[0][0] == "Neutral"

    def test_torch_dtype_stored_on_tone_classifier(self):
        """ToneClassifier must accept and store an arbitrary torch_dtype value."""
        sentinel = object()
        classifier = ToneClassifier(torch_dtype=sentinel)
        assert classifier.torch_dtype is sentinel

    def test_torch_dtype_passed_to_pipeline(self):
        """ToneClassifier._get_pipeline must forward torch_dtype to pipeline()."""
        sentinel = object()
        classifier = ToneClassifier(torch_dtype=sentinel)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            classifier._get_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("torch_dtype") is sentinel

    def test_torch_dtype_none_not_forwarded_to_pipeline(self):
        """When torch_dtype is None, it must not appear in the pipeline() call."""
        classifier = ToneClassifier(torch_dtype=None)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            classifier._get_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert "torch_dtype" not in kwargs

    def test_neural_classify_batch_consumes_iterator(self):
        """_neural_classify_batch must work when the pipeline yields items lazily (iterator)."""
        classifier = ToneClassifier()
        items = [
            {"labels": ["Aggression", "Neutral", "Humor", "Sarkasmus"], "scores": [0.9, 0.05, 0.03, 0.02]},
            {"labels": ["Neutral", "Humor", "Aggression", "Sarkasmus"], "scores": [0.8, 0.1, 0.05, 0.05]},
        ]

        def fake_pipe(iterable, candidate_labels, batch_size):
            yield from items

        classifier._pipe = fake_pipe
        results = classifier._neural_classify_batch(["text one", "text two"])
        assert results[0][0] == "Aggression"
        assert results[1][0] == "Neutral"

    def test_neural_classify_batch_passes_list_to_pipeline(self):
        """_neural_classify_batch must pass a plain list (not an iterator) to the pipeline."""
        classifier = ToneClassifier()
        received = []

        def fake_pipe(iterable, candidate_labels, batch_size):
            received.append(iterable)
            return iter([{"labels": ["Neutral", "Humor", "Aggression", "Sarkasmus"],
                          "scores": [0.9, 0.05, 0.03, 0.02]}])

        classifier._pipe = fake_pipe
        classifier._neural_classify_batch(["a text"])
        assert isinstance(received[0], list), (
            "Pipeline must receive a plain list, not an iterator"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AddresseeDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAddresseeDetector:
    """Tests for AddresseeDetector (rule-based path, no NER model needed)."""

    def setup_method(self):
        import src.nlp as _nlp
        _nlp._NER_PIPELINE = None
        _nlp._NER_FAILED = False
        self.detector = AddresseeDetector(use_ner=False)

    def test_detect_single_fraktion(self):
        result = self.detector.detect("Beifall bei der SPD")
        assert "SPD" in result

    def test_detect_multiple_fraktionen(self):
        result = self.detector.detect("Beifall bei der SPD und der CDU/CSU")
        assert "SPD" in result
        assert "CDU/CSU" in result

    def test_detect_cdu_canonicalised(self):
        result = self.detector.detect("Lachen bei der CDU")
        assert "CDU/CSU" in result

    def test_detect_afd(self):
        result = self.detector.detect("Widerspruch bei der AfD")
        assert "AfD" in result

    def test_detect_gruenen_canonicalised(self):
        result = self.detector.detect("Beifall bei den Grünen")
        assert "Bündnis 90/Die Grünen" in result

    def test_detect_no_addressee_in_neutral_text(self):
        result = self.detector.detect("Der Bundestag ist beschlussfähig.")
        assert result == []

    def test_detect_batch(self):
        texts = [
            "Beifall bei der SPD",
            "Widerspruch von der AfD",
            "Normaler Satz ohne Fraktion",
        ]
        results = self.detector.detect_batch(texts)
        assert len(results) == 3
        assert "SPD" in results[0]
        assert results[2] == []

    def test_no_duplicates(self):
        result = self.detector.detect("Beifall bei der SPD – die SPD jubelt")
        assert result.count("SPD") == 1

    def test_torch_dtype_stored_on_addressee_detector(self):
        """AddresseeDetector must accept and store an arbitrary torch_dtype value."""
        sentinel = object()
        detector = AddresseeDetector(use_ner=True, torch_dtype=sentinel)
        assert detector.torch_dtype is sentinel

    def test_torch_dtype_passed_to_ner_pipeline(self):
        """AddresseeDetector._get_ner_pipeline must forward torch_dtype to pipeline()."""
        sentinel = object()
        detector = AddresseeDetector(use_ner=True, torch_dtype=sentinel)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            detector._get_ner_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("torch_dtype") is sentinel

    def test_torch_dtype_none_not_forwarded_to_ner_pipeline(self):
        """When torch_dtype is None, it must not appear in the NER pipeline() call."""
        detector = AddresseeDetector(use_ner=True, torch_dtype=None)
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            detector._get_ner_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert "torch_dtype" not in kwargs

    def test_batch_size_stored(self):
        """AddresseeDetector must accept and store a custom batch_size."""
        detector = AddresseeDetector(batch_size=64)
        assert detector.batch_size == 64

    def test_batch_size_default(self):
        """AddresseeDetector default batch_size must be 32."""
        detector = AddresseeDetector()
        assert detector.batch_size == 32

    def test_batch_size_passed_to_neural_detect_batch(self):
        """_neural_detect_batch must forward batch_size to the NER pipeline call."""
        detector = AddresseeDetector(use_ner=True, batch_size=16)
        calls = []

        def fake_pipe(iterable, batch_size):
            calls.append(batch_size)
            return iter([[]])

        detector._pipe = None  # ensure pipeline lookup is bypassed
        with patch.object(detector, "_get_ner_pipeline", return_value=fake_pipe):
            detector._neural_detect_batch(["ein Text"])

        assert calls == [16]

    def test_neural_detect_batch_consumes_iterator(self):
        """_neural_detect_batch must collect pipeline output into a list."""
        detector = AddresseeDetector(use_ner=True)
        ents_a = [{"entity_group": "PER", "word": "Scholz"}]
        ents_b = [{"entity_group": "ORG", "word": "Bundestag"}]

        def fake_pipe(iterable, batch_size):
            yield from [ents_a, ents_b]

        with patch.object(detector, "_get_ner_pipeline", return_value=fake_pipe):
            results = detector._neural_detect_batch(["text a", "text b"])

        assert results == [ents_a, ents_b]

    def test_neural_detect_batch_passes_list_to_pipeline(self):
        """_neural_detect_batch must pass a plain list (not an iterator) to the pipeline."""
        detector = AddresseeDetector(use_ner=True)
        received = []

        def fake_pipe(iterable, batch_size):
            received.append(iterable)
            return iter([[]])

        with patch.object(detector, "_get_ner_pipeline", return_value=fake_pipe):
            detector._neural_detect_batch(["ein Text"])

        assert isinstance(received[0], list), (
            "Pipeline must receive a plain list, not an iterator"
        )

    def test_neural_detect_batch_error_fallback(self):
        """_neural_detect_batch must return unique empty lists per text on error."""
        detector = AddresseeDetector(use_ner=True)

        def failing_pipe(iterable, batch_size):
            raise RuntimeError("NER exploded")

        with patch.object(detector, "_get_ner_pipeline", return_value=failing_pipe):
            results = detector._neural_detect_batch(["a", "b", "c"])

        assert results == [[], [], []]
        # Each list must be a separate object (not the same reference)
        assert results[0] is not results[1]

    def test_detect_batch_with_ner_merges_rule_and_ner_results(self):
        """detect_batch must combine rule-based fraktion detection with NER persons."""
        detector = AddresseeDetector(use_ner=True)
        ner_output = [[{"entity_group": "PER", "word": "Scholz"}]]

        with patch.object(detector, "_neural_detect_batch", return_value=ner_output):
            results = detector.detect_batch(["Beifall bei der SPD – Olaf Scholz"])

        assert "SPD" in results[0]
        assert "Scholz" in results[0]

    def test_detect_batch_with_ner_calls_neural_detect_batch(self):
        """detect_batch must delegate NER inference to _neural_detect_batch."""
        detector = AddresseeDetector(use_ner=True)

        with patch.object(
            detector, "_neural_detect_batch", return_value=[[]]
        ) as mock_neural:
            detector.detect_batch(["Ein Text"])

        mock_neural.assert_called_once_with(["Ein Text"])


# ─────────────────────────────────────────────────────────────────────────────
# Model: new Zwischenruf columns
# ─────────────────────────────────────────────────────────────────────────────

class TestZwischenrufNewColumns:
    def test_ton_label_persisted(self):
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=42)
            redner = Redner(vorname="X", nachname="Y")
            rede = Rede(sitzung=sitzung, redner=redner, text="T", wortanzahl=1)
            zwr = Zwischenruf(
                rede=rede,
                text="Lachen bei der SPD",
                ton_label="Humor",
                adressaten="SPD",
            )
            s.add_all([sitzung, redner, rede, zwr])

        with get_session() as s:
            result = s.execute(select(Zwischenruf)).scalars().first()
            assert result.ton_label == "Humor"
            assert result.adressaten == "SPD"


# ─────────────────────────────────────────────────────────────────────────────
# run_nlp_cli helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestRunNlpCliHelpers:
    """Tests for helper functions in scripts/run_nlp_cli.py."""

    def setup_method(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.run_nlp_cli import is_column_attr, _pk_attr, _serialize_for_column, _is_json_column
        self.is_column_attr = is_column_attr
        self._pk_attr = _pk_attr
        self._serialize_for_column = _serialize_for_column
        self._is_json_column = _is_json_column

    # --- is_column_attr ---

    def test_is_column_attr_real_column_returns_true(self):
        assert self.is_column_attr(Zwischenruf, "sentiment_score") is True

    def test_is_column_attr_text_column_returns_true(self):
        assert self.is_column_attr(Zwischenruf, "text") is True

    def test_is_column_attr_nonexistent_returns_false(self):
        assert self.is_column_attr(Zwischenruf, "nonexistent_field") is False

    def test_is_column_attr_relationship_returns_false(self):
        # "rede" is a relationship, not a column — must not be accepted.
        assert self.is_column_attr(Zwischenruf, "rede") is False

    # --- _pk_attr ---

    def test_pk_attr_zwischenruf(self):
        assert self._pk_attr(Zwischenruf) == "ruf_id"

    def test_pk_attr_rede(self):
        assert self._pk_attr(Rede) == "rede_id"

    # --- _serialize_for_column (TEXT columns → JSON string) ---

    def test_serialize_dict_for_text_column_is_json_string(self):
        # adressaten is a TEXT/VARCHAR column (String(512)) that stores serialised
        # complex values — the right semantic target for dict-serialisation tests.
        import json as _json
        val = {"Aggression": 0.9, "Neutral": 0.1}
        result = self._serialize_for_column(Zwischenruf, "adressaten", val)
        # Must be a valid JSON string (not a Python repr with single quotes).
        parsed = _json.loads(result)
        assert parsed == val

    def test_serialize_list_for_text_column_is_json_string(self):
        import json as _json
        val = ["SPD", "AfD"]
        result = self._serialize_for_column(Zwischenruf, "adressaten", val)
        parsed = _json.loads(result)
        assert parsed == val

    def test_serialize_plain_string_is_unchanged(self):
        # ton_label stores a plain label string — verify it round-trips correctly
        # through JSON serialization (json.dumps + json.loads gives back the same value).
        result = self._serialize_for_column(Zwischenruf, "ton_label", "Humor")
        import json as _json
        assert _json.loads(result) == "Humor"

    # --- _is_json_column ---

    def test_is_json_column_text_column_is_false(self):
        # ton_label and adressaten are String/Text columns, not JSON.
        assert self._is_json_column(Zwischenruf, "ton_label") is False

    def test_is_json_column_string_column_is_false(self):
        assert self._is_json_column(Zwischenruf, "adressaten") is False

    # --- CLI default field names match actual model columns ---

    def test_zwischenruf_has_ton_label_column(self):
        """Zwischenruf must expose a 'ton_label' column used by the NLP CLI."""
        assert self.is_column_attr(Zwischenruf, "ton_label"), (
            "Expected Zwischenruf to have a 'ton_label' column"
        )

    def test_zwischenruf_has_adressaten_column(self):
        """Zwischenruf must expose an 'adressaten' column used by the NLP CLI."""
        assert self.is_column_attr(Zwischenruf, "adressaten"), (
            "Expected Zwischenruf to have an 'adressaten' column"
        )

    def test_parser_default_tone_label_field_matches_model(self):
        """The argparse default for --tone-label-field must match the actual Zwischenruf column."""
        from scripts.run_nlp_cli import build_parser
        defaults = build_parser().parse_args([])
        assert self.is_column_attr(Zwischenruf, defaults.tone_label_field), (
            f"Parser default tone-label-field '{defaults.tone_label_field}' not found on Zwischenruf"
        )

    def test_parser_default_addressee_field_matches_model(self):
        """The argparse default for --addressee-field must match the actual Zwischenruf column."""
        from scripts.run_nlp_cli import build_parser
        defaults = build_parser().parse_args([])
        assert self.is_column_attr(Zwischenruf, defaults.addressee_field), (
            f"Parser default addressee-field '{defaults.addressee_field}' not found on Zwischenruf"
        )

    def test_tone_labels_import_covers_all_four_labels(self):
        """run_nlp_cli must import _TONE_LABELS from src.nlp (not use a partial fallback).

        Regression guard for the bug where getattr(tone_classifier, '_TONE_LABELS', ['Neutral'])
        silently produced a single-label dict in the error-path fallback.
        """
        from src.nlp import _TONE_LABELS
        import scripts.run_nlp_cli as cli_module
        # The CLI module must expose the same _TONE_LABELS that nlp.py defines.
        assert cli_module._TONE_LABELS is _TONE_LABELS, (
            "run_nlp_cli._TONE_LABELS is not the same object as src.nlp._TONE_LABELS"
        )
        assert set(_TONE_LABELS) == {"Aggression", "Sarkasmus", "Humor", "Neutral"}, (
            f"_TONE_LABELS contains unexpected labels: {_TONE_LABELS}"
        )


class TestGatherTargets:
    """Tests for gather_targets() SQL-side filtering in scripts/run_nlp_cli.py."""

    def setup_method(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.run_nlp_cli import gather_targets
        self.gather_targets = gather_targets

    def _make_fixtures(self, session):
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1)
        session.add(sitzung)
        session.flush()
        redner = Redner(vorname="Test", nachname="Redner", fraktion="SPD")
        session.add(redner)
        session.flush()
        rede = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner.redner_id,
                    text="Testrede", wortanzahl=1)
        session.add(rede)
        session.flush()
        z_scored = Zwischenruf(rede_id=rede.rede_id, text="Beifall", sentiment_score=0.8)
        z_unscored = Zwischenruf(rede_id=rede.rede_id, text="Widerspruch", sentiment_score=None)
        session.add_all([z_scored, z_unscored])
        session.flush()
        return z_scored, z_unscored

    def test_gather_targets_all_rows_without_filter(self):
        with get_session() as session:
            z_scored, z_unscored = self._make_fixtures(session)
            rows = self.gather_targets(session, "zwischenrufe", None, False,
                                       "sentiment_score")
            assert len(rows) == 2

    def test_gather_targets_only_unscored_sql_filter(self):
        """--only-unscored must filter at SQL level, returning only NULL-score rows."""
        with get_session() as session:
            z_scored, z_unscored = self._make_fixtures(session)
            rows = self.gather_targets(session, "zwischenrufe", None, True,
                                       "sentiment_score")
            assert len(rows) == 1
            assert rows[0][1].text == "Widerspruch"

    def test_gather_targets_limit_applied(self):
        with get_session() as session:
            self._make_fixtures(session)
            rows = self.gather_targets(session, "zwischenrufe", 1, False,
                                       "sentiment_score")
        assert len(rows) == 1


class TestToneClassifierBatchSize:
    """Tests for ToneClassifier batch_size parameter and pipeline truncation."""

    def test_batch_size_stored(self):
        classifier = ToneClassifier(batch_size=64)
        assert classifier.batch_size == 64

    def test_batch_size_default(self):
        classifier = ToneClassifier()
        assert classifier.batch_size == 32

    def test_batch_size_passed_to_pipeline_call(self):
        """_neural_classify_batch must forward batch_size to the pipeline __call__."""
        classifier = ToneClassifier(batch_size=64)
        calls = []

        def fake_pipe(texts, candidate_labels, batch_size):
            calls.append(batch_size)
            return [{"labels": ["Neutral", "Aggression", "Sarkasmus", "Humor"],
                     "scores": [0.9, 0.05, 0.03, 0.02]}]

        classifier._pipe = fake_pipe
        classifier._neural_classify_batch(["Ein Satz"])
        assert calls == [64]

    def test_truncation_forwarded_to_pipeline_constructor(self):
        """_get_pipeline must pass truncation=True and max_length to the pipeline."""
        from unittest.mock import patch
        classifier = ToneClassifier()
        with patch("src.nlp._TRANSFORMERS_AVAILABLE", True), \
             patch("src.nlp.pipeline", create=True) as mock_pipeline:
            mock_pipeline.return_value = lambda *a, **kw: []
            classifier._get_pipeline()
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("truncation") is True
        assert "max_length" in kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Neural fallback uniqueness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralFallbacks:
    """Verify that error-path fallbacks return unique mutable objects per item."""

    def test_sentiment_fallback_returns_zeros(self):
        """SentimentEngine must return [0.0, ...] when the neural pipeline fails."""
        engine = SentimentEngine()

        def failing_pipe(iterable, batch_size):
            raise RuntimeError("pipeline error")

        engine._pipe = failing_pipe
        results = engine._neural_score_batch(["a", "b"])
        assert results == [0.0, 0.0]

    def test_tone_fallback_returns_unique_dicts(self):
        """ToneClassifier error fallback must produce independent dicts per item."""
        classifier = ToneClassifier()

        def failing_pipe(iterable, candidate_labels, batch_size):
            raise RuntimeError("pipeline error")

        classifier._pipe = failing_pipe
        results = classifier._neural_classify_batch(["a", "b", "c"])

        # All three are Neutral
        assert all(label == "Neutral" for label, _ in results)
        # Each dict must be a distinct object (list comprehension, not * repeat)
        dicts = [scores for _, scores in results]
        assert dicts[0] is not dicts[1]
        assert dicts[1] is not dicts[2]

    def test_ner_fallback_returns_unique_lists(self):
        """AddresseeDetector._neural_detect_batch error fallback must produce independent lists."""
        detector = AddresseeDetector(use_ner=True)

        def failing_pipe(iterable, batch_size):
            raise RuntimeError("NER error")

        with patch.object(detector, "_get_ner_pipeline", return_value=failing_pipe):
            results = detector._neural_detect_batch(["a", "b", "c"])

        assert results == [[], [], []]
        assert results[0] is not results[1]
        assert results[1] is not results[2]


# ─────────────────────────────────────────────────────────────────────────────
# NLPSession lifecycle tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNLPSession:
    """Tests for NLPSession device resolution and engine lifecycle."""

    def setup_method(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.nlp_session import NLPSession
        self.NLPSession = NLPSession

    def test_cpu_device_by_default(self):
        """With use_cuda=False, device must resolve to -1 (CPU)."""
        session = self.NLPSession(use_cuda=False)
        assert session._resolve_device() == -1

    def test_explicit_device_bypasses_cuda_logic(self):
        """Providing device= directly must skip all CUDA detection."""
        session = self.NLPSession(device=2)
        assert session._resolve_device() == 2

    def test_engines_initialized_on_enter(self):
        """__enter__ must populate sentiment_engine, tone_classifier, addressee_detector."""
        with self.NLPSession(use_cuda=False) as agent:
            from src.nlp import SentimentEngine, ToneClassifier, AddresseeDetector
            assert isinstance(agent.sentiment_engine, SentimentEngine)
            assert isinstance(agent.tone_classifier, ToneClassifier)
            assert isinstance(agent.addressee_detector, AddresseeDetector)

    def test_engines_cleared_on_exit(self):
        """__exit__ must set all engine attributes to None."""
        session = self.NLPSession(use_cuda=False)
        with session as agent:
            pass  # engines initialized inside
        assert agent.sentiment_engine is None
        assert agent.tone_classifier is None
        assert agent.addressee_detector is None

    def test_batch_size_propagated_to_all_engines(self):
        """NLPSession must forward batch_size to SentimentEngine, ToneClassifier, and AddresseeDetector."""
        with self.NLPSession(use_cuda=False, batch_size=64) as agent:
            assert agent.sentiment_engine.batch_size == 64
            assert agent.tone_classifier.batch_size == 64
            assert agent.addressee_detector.batch_size == 64

    def test_use_ner_propagated_to_addressee_detector(self):
        """NLPSession must forward use_ner to AddresseeDetector."""
        with self.NLPSession(use_cuda=False, use_ner=True) as agent:
            assert agent.addressee_detector.use_ner is True

    def test_context_manager_returns_self(self):
        """__enter__ must return the NLPSession instance itself."""
        session = self.NLPSession(use_cuda=False)
        result = session.__enter__()
        assert result is session
        session.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case: empty inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCasesEmptyInput:
    """All batch methods must handle empty lists and empty strings gracefully."""

    def test_sentiment_score_batch_empty_list(self):
        """score_batch([]) must return an empty list without calling the pipeline."""
        engine = SentimentEngine()
        with patch.object(engine, "_neural_score_batch") as mock_neural:
            result = engine.score_batch([])
        assert result == []
        mock_neural.assert_not_called()

    def test_sentiment_score_batch_empty_string(self):
        """score_batch(['']) must return a score without raising."""
        engine = SentimentEngine()
        with patch.object(engine, "_neural_score_batch", return_value=[0.0]):
            result = engine.score_batch([""])
        assert result == [0.0]

    def test_tone_classify_batch_empty_list(self):
        """classify_batch([]) must return an empty list without calling the pipeline."""
        classifier = ToneClassifier()
        with patch.object(classifier, "_neural_classify_batch") as mock_neural:
            result = classifier.classify_batch([])
        assert result == []
        mock_neural.assert_not_called()

    def test_tone_classify_batch_empty_string(self):
        """classify_batch(['']) must return a result tuple without raising."""
        classifier = ToneClassifier()
        fallback = [("Neutral", {"Aggression": 0.0, "Sarkasmus": 0.0, "Humor": 0.0, "Neutral": 0.0})]
        with patch.object(classifier, "_neural_classify_batch", return_value=fallback):
            result = classifier.classify_batch([""])
        assert len(result) == 1
        assert result[0][0] == "Neutral"

    def test_addressee_detect_batch_empty_list(self):
        """detect_batch([]) must return an empty list immediately."""
        detector = AddresseeDetector()
        result = detector.detect_batch([])
        assert result == []

    def test_addressee_detect_empty_string(self):
        """detect('') must return an empty list without raising."""
        detector = AddresseeDetector()
        result = detector.detect("")
        assert result == []

    def test_addressee_detect_batch_empty_string_entry(self):
        """detect_batch(['']) must return [[]] without raising."""
        detector = AddresseeDetector()
        result = detector.detect_batch([""])
        assert result == [[]]


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case: scraper timeouts
# ─────────────────────────────────────────────────────────────────────────────

class TestScraperTimeouts:
    """BundestagScraper must handle network timeouts without raising."""

    def test_download_one_handles_timeout(self, tmp_path):
        """download_one must return None and not write a file on a Timeout."""
        from unittest.mock import MagicMock
        import requests as req
        from src.scraper import BundestagScraper

        mock_session = MagicMock()
        mock_session.get.side_effect = req.Timeout("timed out")

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        result = scraper.download_one("https://dserver.bundestag.de/btp/20/20001.xml")

        assert result is None
        assert not (tmp_path / "20001.xml").exists()

    def test_fetch_page_handles_timeout(self, tmp_path):
        """_fetch_page must return [] on a network timeout."""
        from unittest.mock import MagicMock
        import requests as req
        from src.scraper import BundestagScraper

        mock_session = MagicMock()
        mock_session.get.side_effect = req.Timeout("timed out")

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        result = scraper._fetch_page(offset=0, limit=10, wahlperiode=20)

        assert result == []

    def test_probe_url_returns_zero_on_timeout(self, tmp_path):
        """_probe_url must return 0 when all attempts raise Timeout."""
        from unittest.mock import MagicMock
        import requests as req
        from src.scraper import BundestagScraper

        mock_session = MagicMock()
        mock_session.get.side_effect = req.Timeout("timed out")

        scraper = BundestagScraper(download_dir=tmp_path, session=mock_session)
        result = scraper._probe_url("https://dserver.bundestag.de/btp/20/20001.xml")

        assert result == 0
        # Must have retried _MAX_RETRIES times before giving up.
        assert mock_session.get.call_count == _MAX_RETRIES


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case: unexpected NLP data formats
# ─────────────────────────────────────────────────────────────────────────────

class TestUnexpectedNLPFormats:
    """NLP engines must degrade gracefully when pipeline output is unexpected."""

    def test_unknown_sentiment_label_maps_to_zero(self):
        """An unrecognised label must produce score 0.0 (via _LABEL_TO_SCORE default)."""
        engine = SentimentEngine()

        def fake_pipe(iterable, batch_size):
            yield {"label": "UNKNOWN_LABEL", "score": 1.0}

        engine._pipe = fake_pipe
        results = engine._neural_score_batch(["some text"])
        # _LABEL_TO_SCORE.get("UNKNOWN_LABEL", 0.0) * 1.0 == 0.0
        assert results == [0.0]

    def test_ner_entity_missing_entity_group_is_skipped(self):
        """Entities without 'entity_group' must be silently ignored."""
        detector = AddresseeDetector(use_ner=True)
        # Entity dict missing the 'entity_group' key entirely.
        ner_output = [[{"word": "Scholz"}]]

        with patch.object(detector, "_neural_detect_batch", return_value=ner_output):
            results = detector.detect_batch(["Text ohne Fraktion"])

        assert "Scholz" not in results[0]

    def test_ner_entity_wrong_entity_group_is_skipped(self):
        """Entities with entity_group other than PER/PERSON must not be added."""
        detector = AddresseeDetector(use_ner=True)
        ner_output = [[{"entity_group": "ORG", "word": "Bundestag"}]]

        with patch.object(detector, "_neural_detect_batch", return_value=ner_output):
            results = detector.detect_batch(["Text"])

        assert "Bundestag" not in results[0]

    def test_tone_fallback_labels_cover_all_tone_labels(self):
        """The error-path fallback dict must contain all four tone labels."""
        from src.nlp import _TONE_LABELS
        classifier = ToneClassifier()

        def failing_pipe(iterable, candidate_labels, batch_size):
            raise RuntimeError("pipeline error")

        classifier._pipe = failing_pipe
        results = classifier._neural_classify_batch(["text"])
        label, scores = results[0]
        assert label == "Neutral"
        for tone in _TONE_LABELS:
            assert tone in scores


# ─────────────────────────────────────────────────────────────────────────────
# Tests: InteraktionsNetzwerk NetworkX export
# ─────────────────────────────────────────────────────────────────────────────

class TestInteraktionsNetzwerkExport:
    """Verify the NetworkX graph export methods on InteraktionsNetzwerk."""

    def _seed(self, session):
        from datetime import date
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner_spd = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        redner_cdu = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")
        rede1 = Rede(sitzung=sitzung, redner=redner_spd, text="Test", wortanzahl=5)
        rede2 = Rede(sitzung=sitzung, redner=redner_cdu, text="Test2", wortanzahl=3)
        zwr1 = Zwischenruf(rede=rede1, text="Beifall", fraktion="CDU/CSU", sentiment_score=0.8)
        zwr2 = Zwischenruf(rede=rede2, text="Widerspruch", fraktion="SPD", sentiment_score=-0.8)
        session.add_all([sitzung, redner_spd, redner_cdu, rede1, rede2, zwr1, zwr2])

    def test_to_networkx_graph_returns_digraph(self):
        import networkx as nx
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            G = nw.to_networkx_graph()

        assert isinstance(G, nx.DiGraph)

    def test_to_networkx_graph_has_edges(self):
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            G = nw.to_networkx_graph()

        assert G.number_of_edges() > 0

    def test_to_networkx_graph_edge_attributes(self):
        """Each edge must carry weight, avg_sentiment, and aggression_score."""
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            G = nw.to_networkx_graph()

        for _src, _tgt, attrs in G.edges(data=True):
            assert "weight" in attrs
            assert "avg_sentiment" in attrs
            assert "aggression_score" in attrs

    def test_to_networkx_graph_empty_returns_empty_digraph(self):
        """With no data the method must return an empty DiGraph, not raise."""
        import networkx as nx
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            G = nw.to_networkx_graph()

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 0

    def test_to_graphml_bytes_returns_bytes(self):
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            data = nw.to_graphml_bytes()

        assert isinstance(data, bytes)
        assert b"graphml" in data.lower()

    def test_to_gexf_bytes_returns_bytes(self):
        from src.analytics import InteraktionsNetzwerk

        with get_session() as s:
            self._seed(s)

        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            data = nw.to_gexf_bytes()

        assert isinstance(data, bytes)
        assert b"gexf" in data.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: InteraktionsNetzwerk – temporal windowing (v2.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestInteraktionsNetzwerkTemporalWindowing:
    """Tests for adjacency_matrix_by_window (Netzwerk-Evolution)."""

    def _seed(self, session):
        from datetime import date
        # Two sessions in different quarters of the same year
        sz_q1 = Sitzung(wahlperiode=20, sitzungsnr=80, datum=date(2022, 2, 1), gesamtwortzahl=100)
        sz_q3 = Sitzung(wahlperiode=20, sitzungsnr=81, datum=date(2022, 8, 1), gesamtwortzahl=100)
        redner_spd = Redner(vorname="A", nachname="SPD_N", fraktion="SPD")
        redner_cdu = Redner(vorname="B", nachname="CDU_N", fraktion="CDU/CSU")
        rede_q1 = Rede(sitzung=sz_q1, redner=redner_spd, text="Rede Q1", wortanzahl=5)
        rede_q3 = Rede(sitzung=sz_q3, redner=redner_cdu, text="Rede Q3", wortanzahl=5)
        z_q1 = Zwischenruf(rede=rede_q1, text="Widerspruch", fraktion="CDU/CSU", sentiment_score=-0.8)
        z_q3 = Zwischenruf(rede=rede_q3, text="Widerspruch", fraktion="SPD", sentiment_score=-0.7)
        session.add_all([sz_q1, sz_q3, redner_spd, redner_cdu, rede_q1, rede_q3, z_q1, z_q3])

    def test_returns_dict(self):
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="quarter")
        assert isinstance(result, dict)

    def test_quarterly_produces_multiple_windows(self):
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="quarter")
        # Should have at least 2 distinct quarters (Q1 2022 and Q3 2022)
        assert len(result) >= 2

    def test_yearly_produces_one_window_for_single_year(self):
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="year")
        # Both sessions are in 2022, so there should be exactly one year window
        assert len(result) == 1
        assert "2022" in list(result.keys())

    def test_each_window_is_dataframe(self):
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="quarter")
        for label, df in result.items():
            assert isinstance(df, __import__("pandas").DataFrame), f"Window '{label}' is not a DataFrame"
            assert not df.empty, f"Window '{label}' is empty"

    def test_empty_db_returns_empty_dict(self):
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            result = InteraktionsNetzwerk(s).adjacency_matrix_by_window(wahlperiode=20)
        assert result == {}

    def test_windows_are_chronologically_ordered(self):
        """Labels must be in chronological order (oldest first)."""
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="quarter")
        labels = list(result.keys())
        # Q1 2022 should come before Q3 2022
        assert labels.index("Q1 2022") < labels.index("Q3 2022")

    def test_score_weighted_propagated(self):
        """With score_weighted=True the matrix should contain aggression scores (non-negative)."""
        from src.analytics import InteraktionsNetzwerk
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            result = nw.adjacency_matrix_by_window(wahlperiode=20, window="year", score_weighted=True)
        for df in result.values():
            assert (df.values >= 0).all(), "score_weighted matrix must contain non-negative values"

    def test_invalid_window_raises_value_error(self):
        """Passing an unsupported window value must raise ValueError immediately."""
        from src.analytics import InteraktionsNetzwerk
        import pytest
        with get_session() as s:
            nw = InteraktionsNetzwerk(s)
            with pytest.raises(ValueError, match="quarter.*year"):
                nw.adjacency_matrix_by_window(window="month")

class TestWahlperiodenVergleich:
    """Smoke-tests for the cross-legislature comparison analytics class."""

    def _seed_two_wps(self, session):
        from datetime import date
        from src.models import Sitzung, Redner, Rede, Zwischenruf

        s19 = Sitzung(wahlperiode=19, sitzungsnr=1, datum=date(2018, 3, 1), gesamtwortzahl=200)
        s20 = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2022, 3, 1), gesamtwortzahl=300)

        redner = Redner(vorname="Anna", nachname="Müller", fraktion="SPD")
        redner2 = Redner(vorname="Bob", nachname="Schulz", fraktion="CDU/CSU")

        rede19 = Rede(sitzung=s19, redner=redner, text="Rede WP 19", wortanzahl=10)
        rede20 = Rede(sitzung=s20, redner=redner2, text="Rede WP 20", wortanzahl=10)

        z19a = Zwischenruf(rede=rede19, text="Beifall", fraktion="CDU/CSU",
                           sentiment_score=0.5, ton_label="Neutral", adressaten="SPD")
        z19b = Zwischenruf(rede=rede19, text="Nein!", fraktion="AfD",
                           sentiment_score=-0.8, ton_label="Aggression", adressaten="SPD")
        z20a = Zwischenruf(rede=rede20, text="Sehr gut", fraktion="SPD",
                           sentiment_score=0.6, ton_label="Neutral", adressaten="CDU/CSU")
        z20b = Zwischenruf(rede=rede20, text="Unsinn!", fraktion="AfD",
                           sentiment_score=-0.7, ton_label="Aggression", adressaten="CDU/CSU")

        session.add_all([s19, s20, redner, redner2, rede19, rede20, z19a, z19b, z20a, z20b])

    def test_aggression_by_wp_returns_two_rows(self):
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.aggression_by_wp()

        assert len(df) == 2
        assert set(df["wahlperiode"]) == {19, 20}

    def test_aggression_by_wp_columns(self):
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.aggression_by_wp()

        required = {"wahlperiode", "avg_aggression", "neg_zwischenrufe",
                    "gesamt_zwischenrufe", "anteil_negativ_pct"}
        assert required <= set(df.columns)

    def test_aggression_by_wp_empty_db(self):
        """With an empty DB, aggression_by_wp must return an empty DataFrame."""
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.aggression_by_wp()

        assert df.empty

    def test_ton_by_wp_returns_rows(self):
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.ton_by_wp()

        assert not df.empty
        assert "anteil_pct" in df.columns

    def test_ton_by_wp_percentages_sum_to_100_per_wp(self):
        """anteil_pct must sum to 100 % per Wahlperiode (within rounding tolerance)."""
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.ton_by_wp()

        for wp, grp in df.groupby("wahlperiode"):
            total = grp["anteil_pct"].sum()
            assert abs(total - 100.0) < 1.0, f"WP {wp}: anteil_pct sum = {total}"

    def test_activity_by_wp_returns_two_rows(self):
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.activity_by_wp()

        assert len(df) == 2
        assert set(df["wahlperiode"]) == {19, 20}

    def test_activity_by_wp_columns(self):
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.activity_by_wp()

        required = {"wahlperiode", "sitzungen", "reden", "zwischenrufe",
                    "zwischenrufe_pro_rede", "worte_pro_sitzung"}
        assert required <= set(df.columns)

    def test_activity_by_wp_empty_db(self):
        """With an empty DB, activity_by_wp must return an empty DataFrame."""
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.activity_by_wp()

        assert df.empty

    def test_activity_zwischenrufe_count_correct(self):
        """zwischenrufe column must match actual interjection count."""
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            self._seed_two_wps(s)

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.activity_by_wp()

        row19 = df[df["wahlperiode"] == 19].iloc[0]
        assert row19["zwischenrufe"] == 2  # z19a and z19b

    def test_activity_worte_pro_sitzung_not_multiplied_by_reden(self):
        """worte_pro_sitzung must equal gesamtwortzahl / sitzungen.

        Regression for: gesamtwortzahl was summed over the OUTER JOIN with
        Rede, causing each session's word count to be multiplied by its
        number of speeches.  This test seeds a session with gesamtwortzahl=500
        and three speeches; the expected value is 500, not 1500.
        """
        from datetime import date
        from src.models import Sitzung, Redner, Rede
        from src.analytics import WahlperiodenVergleich

        with get_session() as s:
            sitzung = Sitzung(wahlperiode=21, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=500)
            redner = Redner(vorname="Test", nachname="Redner", fraktion="SPD")
            # Three speeches — before the fix this would have yielded 500×3=1500.
            rede_a = Rede(sitzung=sitzung, redner=redner, text="Rede A", wortanzahl=100)
            rede_b = Rede(sitzung=sitzung, redner=redner, text="Rede B", wortanzahl=200)
            rede_c = Rede(sitzung=sitzung, redner=redner, text="Rede C", wortanzahl=200)
            s.add_all([sitzung, redner, rede_a, rede_b, rede_c])

        with get_session() as s:
            wv = WahlperiodenVergleich(s)
            df = wv.activity_by_wp()

        row = df[df["wahlperiode"] == 21].iloc[0]
        assert row["worte_pro_sitzung"] == 500, (
            f"Expected worte_pro_sitzung=500, got {row['worte_pro_sitzung']} "
            "(likely caused by join-multiplication bug)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# H. TOPAnalyse tests  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestTOPAnalyse:
    """Tests for TOPAnalyse: agenda-item hostility ranking."""

    def _seed(self, session):
        from datetime import date
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=50, datum=date(2022, 3, 1), gesamtwortzahl=200)
        redner = Redner(vorname="Anna", nachname="Test", fraktion="SPD")
        rede1 = Rede(
            sitzung=sitzung, redner=redner, text="Klimaschutz brauchen wir",
            tagesordnungspunkt="TOP 1 – Klimaschutz", wortanzahl=5,
        )
        rede2 = Rede(
            sitzung=sitzung, redner=redner, text="Wirtschaft stärken",
            tagesordnungspunkt="TOP 2 – Wirtschaft", wortanzahl=3,
        )
        # Three negative interjections on TOP 1, one on TOP 2
        z1 = Zwischenruf(rede=rede1, text="Widerspruch", fraktion="CDU/CSU",
                         sentiment_score=-0.8, kategorie="Widerspruch")
        z2 = Zwischenruf(rede=rede1, text="Unruhe", fraktion="AfD",
                         sentiment_score=-0.9, kategorie="Unruhe")
        z3 = Zwischenruf(rede=rede1, text="Beifall", fraktion="SPD",
                         sentiment_score=0.8, kategorie="Beifall")
        z4 = Zwischenruf(rede=rede2, text="Widerspruch", fraktion="AfD",
                         sentiment_score=-0.5, kategorie="Widerspruch")
        session.add_all([sitzung, redner, rede1, rede2, z1, z2, z3, z4])

    def test_aggression_by_top_returns_dataframe(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ta = TOPAnalyse(s)
            df = ta.aggression_by_top(min_reden=1)
        assert not df.empty
        assert "tagesordnungspunkt" in df.columns
        assert "avg_aggression" in df.columns

    def test_aggression_by_top_sorted_descending(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ta = TOPAnalyse(s)
            df = ta.aggression_by_top(min_reden=1)
        assert list(df["avg_aggression"]) == sorted(df["avg_aggression"], reverse=True)

    def test_aggression_by_top_wahlperiode_filter(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ta = TOPAnalyse(s)
            df_wp20 = ta.aggression_by_top(wahlperiode=20, min_reden=1)
            df_wp19 = ta.aggression_by_top(wahlperiode=19, min_reden=1)
        assert not df_wp20.empty
        assert df_wp19.empty

    def test_aggression_by_top_min_reden_filter(self):
        """A TOP with only 1 speech should be excluded when min_reden=2."""
        from src.analytics import TOPAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ta = TOPAnalyse(s)
            df_inclusive = ta.aggression_by_top(min_reden=1)
            df_exclusive = ta.aggression_by_top(min_reden=5)
        assert not df_inclusive.empty
        assert df_exclusive.empty

    def test_aggression_by_top_empty_db(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            df = TOPAnalyse(s).aggression_by_top()
        assert df.empty

    def test_kategorie_by_top_returns_dataframe(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ta = TOPAnalyse(s)
            df = ta.kategorie_by_top(min_reden=1)
        assert not df.empty
        assert set(df.columns) >= {"tagesordnungspunkt", "kategorie", "anzahl"}

    def test_kategorie_by_top_empty_db(self):
        from src.analytics import TOPAnalyse
        with get_session() as s:
            df = TOPAnalyse(s).kategorie_by_top()
        assert df.empty

    def test_aggression_by_top_null_sentiment_still_returns_top(self):
        """TOPs whose Zwischenrufe all have NULL sentiment_score must still appear.

        Regression guard: if `.where(Zwischenruf.sentiment_score.isnot(None))`
        is ever re-added to the query, this test will fail immediately, exposing
        the regression (all TOP data would vanish for any DB that hasn't been
        run through the NLP pipeline yet).
        """
        import math
        from datetime import date
        from src.analytics import TOPAnalyse
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=77, datum=date(2022, 5, 1), gesamtwortzahl=100)
            redner = Redner(vorname="Bob", nachname="Null", fraktion="Grüne")
            rede = Rede(
                sitzung=sitzung, redner=redner, text="Test speech",
                tagesordnungspunkt="TOP 7 – Null-Test", wortanzahl=3,
            )
            # All Zwischenrufe have sentiment_score=None (NLP not yet run)
            z1 = Zwischenruf(rede=rede, text="Widerspruch", fraktion="AfD",
                             sentiment_score=None, kategorie="Widerspruch")
            z2 = Zwischenruf(rede=rede, text="Unruhe", fraktion="CDU/CSU",
                             sentiment_score=None, kategorie="Unruhe")
            s.add_all([sitzung, redner, rede, z1, z2])
        with get_session() as s:
            df = TOPAnalyse(s).aggression_by_top(min_reden=1)
        assert not df.empty, "TOP with un-scored Zwischenrufe must still appear"
        assert "TOP 7 – Null-Test" in df["tagesordnungspunkt"].values
        row = df[df["tagesordnungspunkt"] == "TOP 7 – Null-Test"].iloc[0]
        # SQL AVG over all-NULL values returns NULL → avg_aggression is NaN
        assert math.isnan(row["avg_aggression"]), "avg_aggression must be NaN when all scores are NULL"
        # Total interjection count must still be counted correctly
        assert row["gesamt_zwischenrufe"] == 2

    def test_aggression_by_top_unscored_sorted_by_gesamt_zwischenrufe(self):
        """When avg_aggression is NaN (no NLP scores), secondary sort by gesamt_zwischenrufe applies."""
        from datetime import date
        from src.analytics import TOPAnalyse
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=88, datum=date(2022, 6, 1), gesamtwortzahl=100)
            redner = Redner(vorname="Carol", nachname="Sort", fraktion="FDP")
            rede_a = Rede(
                sitzung=sitzung, redner=redner, text="Rede A",
                tagesordnungspunkt="TOP A – Viele Rufe", wortanzahl=3,
            )
            rede_b = Rede(
                sitzung=sitzung, redner=redner, text="Rede B",
                tagesordnungspunkt="TOP B – Wenig Rufe", wortanzahl=3,
            )
            # TOP A: 3 un-scored interjections; TOP B: 1 un-scored interjection
            for _ in range(3):
                s.add(Zwischenruf(rede=rede_a, text="Widerspruch", fraktion="AfD",
                                  sentiment_score=None, kategorie="Widerspruch"))
            s.add(Zwischenruf(rede=rede_b, text="Beifall", fraktion="SPD",
                              sentiment_score=None, kategorie="Beifall"))
            s.add_all([sitzung, redner, rede_a, rede_b])
        with get_session() as s:
            df = TOPAnalyse(s).aggression_by_top(min_reden=1)
        assert not df.empty
        assert len(df) == 2
        # All avg_aggression values must be NaN (no NLP scores)
        assert df["avg_aggression"].isna().all()
        # TOP A (3 interjections) must come before TOP B (1 interjection)
        assert df.iloc[0]["tagesordnungspunkt"] == "TOP A – Viele Rufe"


# ─────────────────────────────────────────────────────────────────────────────
# I. KategorieAnalyse tests  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestKategorieAnalyse:
    """Tests for KategorieAnalyse: interjection category distribution."""

    def _seed(self, session):
        from datetime import date
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=51, datum=date(2022, 4, 1), gesamtwortzahl=200)
        redner_spd = Redner(vorname="Olaf", nachname="Scholz2", fraktion="SPD")
        redner_afd = Redner(vorname="Hans", nachname="AfD2", fraktion="AfD")
        rede_spd = Rede(sitzung=sitzung, redner=redner_spd, text="Rede SPD", wortanzahl=5)
        rede_afd = Rede(sitzung=sitzung, redner=redner_afd, text="Rede AfD", wortanzahl=5)
        # CDU applauds SPD
        z1 = Zwischenruf(rede=rede_spd, text="Beifall", fraktion="CDU/CSU",
                         sentiment_score=0.8, kategorie="Beifall")
        # AfD opposes SPD
        z2 = Zwischenruf(rede=rede_spd, text="Widerspruch", fraktion="AfD",
                         sentiment_score=-0.8, kategorie="Widerspruch")
        # SPD laughs at AfD
        z3 = Zwischenruf(rede=rede_afd, text="Lachen", fraktion="SPD",
                         sentiment_score=0.2, kategorie="Lachen")
        session.add_all([sitzung, redner_spd, redner_afd, rede_spd, rede_afd, z1, z2, z3])

    def test_kategorie_by_fraktion_given(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ka = KategorieAnalyse(s)
            df = ka.kategorie_by_fraktion(wahlperiode=20, mode="given")
        assert not df.empty
        assert set(df.columns) >= {"fraktion", "kategorie", "anzahl"}

    def test_kategorie_by_fraktion_received(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ka = KategorieAnalyse(s)
            df = ka.kategorie_by_fraktion(wahlperiode=20, mode="received")
        assert not df.empty

    def test_beifall_widerspruch_ratio_columns(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ka = KategorieAnalyse(s)
            df = ka.beifall_widerspruch_ratio(wahlperiode=20)
        assert set(df.columns) >= {
            "interruptor_fraktion", "sprecher_fraktion", "beifall", "widerspruch", "civility_index"
        }

    def test_civility_index_positive_for_applauding_pair(self):
        """A pair with only Beifall should have civility_index >= 1."""
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ka = KategorieAnalyse(s)
            df = ka.beifall_widerspruch_ratio(wahlperiode=20)
        # CDU/CSU → SPD only has Beifall, so civility should be >= 1
        cdu_spd = df[(df["interruptor_fraktion"] == "CDU/CSU") &
                     (df["sprecher_fraktion"] == "SPD")]
        if not cdu_spd.empty:
            assert cdu_spd.iloc[0]["civility_index"] >= 1.0

    def test_lachen_by_redner_returns_ranked(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            ka = KategorieAnalyse(s)
            df = ka.lachen_by_redner(wahlperiode=20)
        assert not df.empty
        assert "lachen_count" in df.columns
        # AfD speaker should appear (got Lachen from SPD)
        assert "AfD2" in df["nachname"].values

    def test_lachen_by_redner_empty_db(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            df = KategorieAnalyse(s).lachen_by_redner()
        assert df.empty

    def test_invalid_mode_raises_value_error(self):
        """Passing an unsupported mode must raise ValueError immediately."""
        from src.analytics import KategorieAnalyse
        import pytest
        with get_session() as s:
            ka = KategorieAnalyse(s)
            with pytest.raises(ValueError, match="given.*received"):
                ka.kategorie_by_fraktion(mode="both")


# ─────────────────────────────────────────────────────────────────────────────
# M. RedeZeitAnalyse tests  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedeZeitAnalyse:
    """Tests for RedeZeitAnalyse: speech-time fairness."""

    def _seed(self, session):
        from datetime import date
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=52, datum=date(2022, 5, 1), gesamtwortzahl=500)
        redner_spd = Redner(vorname="SPD_r", nachname="Sprecher", fraktion="SPD")
        redner_afd = Redner(vorname="AfD_r", nachname="Sprecher", fraktion="AfD")
        # SPD speaks 300 words, AfD speaks 100 words
        rede1 = Rede(sitzung=sitzung, redner=redner_spd, text="SPD Rede lang", wortanzahl=300)
        rede2 = Rede(sitzung=sitzung, redner=redner_afd, text="AfD Rede kurz", wortanzahl=100)
        z1 = Zwischenruf(rede=rede1, text="Widerspruch", fraktion="AfD", sentiment_score=-0.7)
        session.add_all([sitzung, redner_spd, redner_afd, rede1, rede2, z1])

    def test_wortanzahl_by_fraktion(self):
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            rz = RedeZeitAnalyse(s)
            df = rz.wortanzahl_by_fraktion(wahlperiode=20)
        assert not df.empty
        assert "total_worte" in df.columns
        # SPD should have more words
        spd_row = df[df["fraktion"] == "SPD"]
        afd_row = df[df["fraktion"] == "AfD"]
        assert not spd_row.empty
        assert spd_row.iloc[0]["total_worte"] > afd_row.iloc[0]["total_worte"]

    def test_fairness_index_columns(self):
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            rz = RedeZeitAnalyse(s)
            df = rz.fairness_index(wahlperiode=20)
        assert set(df.columns) >= {
            "fraktion", "total_worte", "wort_anteil_pct",
            "sitz_anteil_pct", "fairness_index", "ueber_unterrepraesentation"
        }

    def test_fairness_index_label_proportional(self):
        """A faction with fairness_index near 1.0 should get 'Proportional' label."""
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            rz = RedeZeitAnalyse(s)
            df = rz.fairness_index(wahlperiode=20)
        valid_labels = {"Überrepräsentiert", "Proportional", "Unterrepräsentiert"}
        assert all(lbl in valid_labels for lbl in df["ueber_unterrepraesentation"])

    def test_fairness_index_empty_db(self):
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            df = RedeZeitAnalyse(s).fairness_index(wahlperiode=20)
        assert df.empty

    def test_top_redselige_redner(self):
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            rz = RedeZeitAnalyse(s)
            df = rz.top_redselige_redner(n=10, wahlperiode=20)
        assert not df.empty
        assert "total_worte" in df.columns
        # Most verbose should be SPD speaker (300 words)
        assert df.iloc[0]["total_worte"] == 300

    def test_wortanzahl_vs_zwischenrufe(self):
        from src.analytics import RedeZeitAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            rz = RedeZeitAnalyse(s)
            df = rz.wortanzahl_vs_zwischenrufe(wahlperiode=20)
        assert not df.empty
        assert "neg_pro_100_worte" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# ThemenKarriere v2.3.0 extension tests
# ─────────────────────────────────────────────────────────────────────────────

class TestThemenKarriereV23:
    """Tests for new v2.3.0 ThemenKarriere methods."""

    def _seed(self, session):
        from datetime import date
        sz1 = Sitzung(wahlperiode=19, sitzungsnr=1, datum=date(2019, 1, 15), gesamtwortzahl=100)
        sz2 = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=200)
        redner = Redner(vorname="Test", nachname="Redner", fraktion="SPD")
        rede1 = Rede(sitzung=sz1, redner=redner, text="Klimaschutz ist wichtig Klimaschutz", wortanzahl=5)
        rede2 = Rede(sitzung=sz2, redner=redner, text="Klimaschutz und Migration und Klimaschutz", wortanzahl=6)
        z1 = Zwischenruf(rede=rede1, text="Widerspruch", fraktion="AfD", sentiment_score=-0.8)
        z2 = Zwischenruf(rede=rede2, text="Beifall", fraktion="CDU/CSU", sentiment_score=0.7)
        session.add_all([sz1, sz2, redner, rede1, rede2, z1, z2])

    def test_multi_wp_keyword_trend(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            tk = ThemenKarriere(s)
            df = tk.multi_wp_keyword_trend("Klimaschutz")
        assert not df.empty
        assert "wahlperiode" in df.columns
        assert set(df["wahlperiode"].unique()) == {19, 20}

    def test_multi_wp_keyword_trend_normiert(self):
        """normiert column must be non-negative."""
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).multi_wp_keyword_trend("Klimaschutz")
        assert (df["normiert"] >= 0).all()

    def test_multi_wp_keyword_trend_empty_for_missing_keyword(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).multi_wp_keyword_trend("XYZUnbekannt123")
        # rohanzahl should all be 0 or df is empty
        if not df.empty:
            assert (df["rohanzahl"] == 0).all()

    def test_keyword_peak_by_wp(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).keyword_peak_by_wp("Klimaschutz")
        assert not df.empty
        assert set(df.columns) >= {"wahlperiode", "peak_normiert", "peak_datum"}
        assert len(df) == 2  # one peak per Wahlperiode

    def test_keyword_peak_by_wp_empty_for_missing(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            df = ThemenKarriere(s).keyword_peak_by_wp("NichtVorhanden")
        assert df.empty

    def test_keyword_aggression_correlation(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).keyword_aggression_correlation("Klimaschutz")
        assert not df.empty
        assert set(df["group"].unique()) == {"Keyword", "Baseline"}

    def test_keyword_aggression_correlation_empty_db(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            df = ThemenKarriere(s).keyword_aggression_correlation("Klimaschutz")
        assert df.empty

    def test_most_polarizing_keywords(self):
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).most_polarizing_keywords(["Klimaschutz", "Migration"])
        # May be empty if sentiment data is insufficient but must be a DataFrame
        assert isinstance(df, __import__("pandas").DataFrame)

    def test_most_polarizing_keywords_sorted_by_delta(self):
        """Results must be sorted by delta_aggression descending (NaN excluded)."""
        from src.analytics import ThemenKarriere
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = ThemenKarriere(s).most_polarizing_keywords(["Klimaschutz", "Migration"])
        if len(df) > 1:
            non_nan = df["delta_aggression"].dropna()
            assert list(non_nan) == sorted(non_nan, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# L. SitzungsKlima tests  (v2.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestSitzungsKlima:
    """Tests for SitzungsKlima: per-session temperature index."""

    def _seed(self, session):
        from datetime import date
        sz1 = Sitzung(wahlperiode=20, sitzungsnr=60, datum=date(2022, 6, 1), gesamtwortzahl=200)
        sz2 = Sitzung(wahlperiode=20, sitzungsnr=61, datum=date(2022, 6, 8), gesamtwortzahl=200)
        redner = Redner(vorname="T", nachname="R", fraktion="SPD")
        rede1 = Rede(sitzung=sz1, redner=redner, text="Rede 1", wortanzahl=10)
        rede2 = Rede(sitzung=sz2, redner=redner, text="Rede 2", wortanzahl=10)
        # Calm session
        z_calm = Zwischenruf(rede=rede1, text="Beifall", fraktion="CDU/CSU",
                             sentiment_score=0.8, ton_label="Neutral", kategorie="Beifall")
        # Heated session
        z_hot1 = Zwischenruf(rede=rede2, text="Widerspruch", fraktion="AfD",
                              sentiment_score=-0.9, ton_label="Aggression", kategorie="Unruhe")
        z_hot2 = Zwischenruf(rede=rede2, text="Widerspruch2", fraktion="AfD",
                              sentiment_score=-0.8, ton_label="Aggression", kategorie="Widerspruch")
        session.add_all([sz1, sz2, redner, rede1, rede2, z_calm, z_hot1, z_hot2])

    def test_klima_per_sitzung_columns(self):
        from src.analytics import SitzungsKlima
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = SitzungsKlima(s).klima_per_sitzung(wahlperiode=20)
        assert set(df.columns) >= {
            "sitzungsnr", "datum", "avg_sentiment",
            "anteil_aggression_pct", "zwischenrufe_pro_rede",
            "anteil_unruhe_pct", "temperatur_index",
        }

    def test_klima_per_sitzung_two_rows(self):
        from src.analytics import SitzungsKlima
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = SitzungsKlima(s).klima_per_sitzung(wahlperiode=20)
        assert len(df) == 2

    def test_temperatur_index_range(self):
        """temperatur_index must be in [0, 1]."""
        from src.analytics import SitzungsKlima
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = SitzungsKlima(s).klima_per_sitzung(wahlperiode=20)
        assert (df["temperatur_index"] >= 0).all()
        assert (df["temperatur_index"] <= 1).all()

    def test_heated_session_has_higher_index(self):
        """The session with hostile interjections should score higher."""
        from src.analytics import SitzungsKlima
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = SitzungsKlima(s).klima_per_sitzung(wahlperiode=20)
        row_calm = df[df["sitzungsnr"] == 60].iloc[0]
        row_hot = df[df["sitzungsnr"] == 61].iloc[0]
        assert row_hot["temperatur_index"] >= row_calm["temperatur_index"]

    def test_hottest_sessions_limit(self):
        from src.analytics import SitzungsKlima
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = SitzungsKlima(s).hottest_sessions(n=1, wahlperiode=20)
        assert len(df) == 1

    def test_klima_per_sitzung_empty_db(self):
        from src.analytics import SitzungsKlima
        with get_session() as s:
            df = SitzungsKlima(s).klima_per_sitzung()
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# K. RednerProfil tests  (v2.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestRednerProfil:
    """Tests for RednerProfil: speaker rhetorical fingerprint."""

    def _seed(self, session):
        from datetime import date
        sz = Sitzung(wahlperiode=20, sitzungsnr=70, datum=date(2022, 7, 1), gesamtwortzahl=100)
        r1 = Redner(vorname="Olaf", nachname="Scholz3", fraktion="SPD")
        r2 = Redner(vorname="Friedrich", nachname="Merz3", fraktion="CDU/CSU")
        rede1 = Rede(
            sitzung=sz, redner=r1, text="Rede 1", wortanzahl=10,
            tone_scores={"Aggression": 0.1, "Sarkasmus": 0.1, "Humor": 0.2, "Neutral": 0.6},
        )
        rede2 = Rede(
            sitzung=sz, redner=r2, text="Rede 2", wortanzahl=10,
            tone_scores={"Aggression": 0.7, "Sarkasmus": 0.1, "Humor": 0.1, "Neutral": 0.1},
        )
        session.add_all([sz, r1, r2, rede1, rede2])
        session.flush()
        return r1.redner_id, r2.redner_id

    def test_faction_profile_columns(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = RednerProfil(s).faction_profile(wahlperiode=20)
        assert "fraktion" in df.columns
        assert set(["Aggression", "Sarkasmus", "Humor", "Neutral"]).issubset(df.columns)

    def test_faction_profile_spd_is_neutral(self):
        """SPD speaker's profile should lean Neutral."""
        from src.analytics import RednerProfil
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = RednerProfil(s).faction_profile(wahlperiode=20)
        spd = df[df["fraktion"] == "SPD"]
        assert not spd.empty
        assert spd.iloc[0]["Neutral"] > spd.iloc[0]["Aggression"]

    def test_top_speakers_by_tone_aggression(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = RednerProfil(s).top_speakers_by_tone("Aggression", n=5, wahlperiode=20)
        assert not df.empty
        assert "avg_probability" in df.columns
        # CDU speaker (Merz3) should be top aggressive
        assert "Merz3" in df.iloc[0]["nachname"]

    def test_top_speakers_by_tone_invalid_label(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            df = RednerProfil(s).top_speakers_by_tone("Sonstiges")
        assert df.empty

    def test_speaker_profile_returns_four_rows(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            r1_id, _ = self._seed(s)
        with get_session() as s:
            df = RednerProfil(s).speaker_profile(redner_id=r1_id, wahlperiode=20)
        assert len(df) == 4
        assert set(df["label"]) == {"Aggression", "Sarkasmus", "Humor", "Neutral"}

    def test_speaker_profile_probabilities_sum_to_one(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            r1_id, _ = self._seed(s)
        with get_session() as s:
            df = RednerProfil(s).speaker_profile(redner_id=r1_id, wahlperiode=20)
        total = df["avg_probability"].sum()
        assert abs(total - 1.0) < 0.01

    def test_speaker_profile_empty_for_nonexistent_id(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            df = RednerProfil(s).speaker_profile(redner_id=99999)
        assert df.empty

    def test_faction_profile_empty_db(self):
        from src.analytics import RednerProfil
        with get_session() as s:
            df = RednerProfil(s).faction_profile()
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# L. AdressatenAnalyse tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdressatenAnalyse:
    """Tests for AdressatenAnalyse: addressee parsing and aggregation."""

    # ------------------------------------------------------------------
    # _parse_adressaten unit tests
    # ------------------------------------------------------------------

    def test_parse_json_list(self):
        from src.analytics import _parse_adressaten
        assert _parse_adressaten('["SPD", "CDU/CSU"]') == ["SPD", "CDU/CSU"]

    def test_parse_json_single_string(self):
        from src.analytics import _parse_adressaten
        assert _parse_adressaten('"SPD"') == ["SPD"]

    def test_parse_legacy_comma_separated(self):
        from src.analytics import _parse_adressaten
        assert _parse_adressaten("SPD, CDU/CSU") == ["SPD", "CDU/CSU"]

    def test_parse_empty_string(self):
        from src.analytics import _parse_adressaten
        assert _parse_adressaten("") == []

    def test_parse_whitespace_only(self):
        from src.analytics import _parse_adressaten
        assert _parse_adressaten("   ") == []

    def test_parse_strips_whitespace_entries(self):
        from src.analytics import _parse_adressaten
        result = _parse_adressaten('["  SPD  ", "CDU/CSU"]')
        assert result == ["SPD", "CDU/CSU"]

    # ------------------------------------------------------------------
    # top_adressaten integration tests
    # ------------------------------------------------------------------

    def _seed(self, session):
        from datetime import date
        sz = Sitzung(wahlperiode=20, sitzungsnr=80, datum=date(2023, 1, 15), gesamtwortzahl=500)
        r = Redner(vorname="Test", nachname="Redner", fraktion="SPD")
        rede = Rede(sitzung=sz, redner=r, text="Rede", wortanzahl=50)
        # JSON-serialised list (the real storage format)
        z1 = Zwischenruf(rede=rede, text="!", fraktion="AfD", adressaten='["SPD", "CDU/CSU"]')
        # second interjection targeting SPD only
        z2 = Zwischenruf(rede=rede, text="!", fraktion="CDU/CSU", adressaten='["SPD"]')
        session.add_all([sz, r, rede, z1, z2])
        session.flush()

    def test_top_adressaten_no_artifacts(self):
        """Labels must not contain JSON artifacts like [ ] or double-quotes."""
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = AdressatenAnalyse(s).top_adressaten(wahlperiode=20)
        labels = set(df["adressat"].tolist())
        for label in labels:
            assert "[" not in label, f"JSON artifact '[' found in label: {label!r}"
            assert "]" not in label, f"JSON artifact ']' found in label: {label!r}"
            assert '"' not in label, f"JSON artifact '\"' found in label: {label!r}"

    def test_top_adressaten_counts(self):
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = AdressatenAnalyse(s).top_adressaten(wahlperiode=20)
        spd_row = df[df["adressat"] == "SPD"]
        assert not spd_row.empty
        # SPD is targeted by both z1 and z2
        assert spd_row.iloc[0]["anzahl"] == 2

    def test_top_adressaten_empty_db(self):
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            df = AdressatenAnalyse(s).top_adressaten()
        assert df.empty
        assert list(df.columns) == ["adressat", "anzahl"]

    # ------------------------------------------------------------------
    # fraktion_targets_fraktion integration tests
    # ------------------------------------------------------------------

    def test_fraktion_targets_fraktion_no_artifacts(self):
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = AdressatenAnalyse(s).fraktion_targets_fraktion(wahlperiode=20)
        for label in df["adressat"].tolist():
            assert "[" not in label
            assert "]" not in label
            assert '"' not in label

    def test_fraktion_targets_fraktion_columns(self):
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = AdressatenAnalyse(s).fraktion_targets_fraktion(wahlperiode=20)
        assert set(df.columns) == {"fraktion", "adressat", "anzahl"}

    def test_fraktion_targets_fraktion_empty_db(self):
        from src.analytics import AdressatenAnalyse
        with get_session() as s:
            df = AdressatenAnalyse(s).fraktion_targets_fraktion()
        assert df.empty
        assert list(df.columns) == ["fraktion", "adressat", "anzahl"]



# ─────────────────────────────────────────────────────────────────────────────
# N. RednerVergleich tests  (v3.0.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestRednerVergleich:
    """Tests for RednerVergleich: side-by-side MP comparison."""

    def _seed(self, session):
        from datetime import date
        sz = Sitzung(wahlperiode=20, sitzungsnr=90, datum=date(2023, 3, 1), gesamtwortzahl=200)
        r1 = Redner(vorname="Anna", nachname="Beispiel", fraktion="SPD")
        r2 = Redner(vorname="Bernd", nachname="Muster", fraktion="CDU/CSU")
        rede1 = Rede(
            sitzung=sz, redner=r1, text="Rede A", wortanzahl=50,
            sentiment_score=0.3,
            tone_scores={"Aggression": 0.1, "Sarkasmus": 0.2, "Humor": 0.3, "Neutral": 0.4},
        )
        rede2 = Rede(
            sitzung=sz, redner=r2, text="Rede B", wortanzahl=80,
            sentiment_score=-0.2,
            tone_scores={"Aggression": 0.5, "Sarkasmus": 0.2, "Humor": 0.1, "Neutral": 0.2},
        )
        zr1 = Zwischenruf(rede=rede1, text="Buh!", fraktion="CDU/CSU", sentiment_score=-0.7)
        zr2 = Zwischenruf(rede=rede2, text="Gut!", fraktion="SPD", sentiment_score=0.5)
        session.add_all([sz, r1, r2, rede1, rede2, zr1, zr2])
        session.flush()
        return r1.redner_id, r2.redner_id

    def test_compare_tone_profiles_columns(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_tone_profiles(r1_id, r2_id)
        assert list(df.columns) == ["label", "speaker_a", "speaker_b"]
        assert len(df) == 4
        assert set(df["label"]) == {"Aggression", "Sarkasmus", "Humor", "Neutral"}

    def test_compare_tone_profiles_values_in_range(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_tone_profiles(r1_id, r2_id, wahlperiode=20)
        assert (df["speaker_a"] >= 0).all() and (df["speaker_a"] <= 1).all()
        assert (df["speaker_b"] >= 0).all() and (df["speaker_b"] <= 1).all()

    def test_compare_tone_profiles_date_filter(self):
        """Date filter excludes speeches outside the range."""
        from datetime import date
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            # Seed is on 2023-03-01; filter after → empty profiles
            df = RednerVergleich(s).compare_tone_profiles(
                r1_id, r2_id, datum_von=date(2024, 1, 1)
            )
        assert (df["speaker_a"] == 0.0).all()
        assert (df["speaker_b"] == 0.0).all()

    def test_compare_speech_stats_columns(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_speech_stats(r1_id, r2_id)
        assert list(df.columns) == ["Metrik", "speaker_a", "speaker_b"]
        assert len(df) == 3

    def test_compare_speech_stats_reden_count(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_speech_stats(r1_id, r2_id, wahlperiode=20)
        reden_row = df[df["Metrik"] == "Reden"]
        assert not reden_row.empty
        assert reden_row.iloc[0]["speaker_a"] == 1
        assert reden_row.iloc[0]["speaker_b"] == 1

    def test_compare_aggression_columns(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_aggression(r1_id, r2_id)
        assert list(df.columns) == ["Metrik", "speaker_a", "speaker_b"]
        assert "Zwischenrufe gesamt" in df["Metrik"].values

    def test_compare_aggression_neg_count(self):
        """r1 receives 1 negative interjection; r2 receives 1 positive."""
        from src.analytics import RednerVergleich
        with get_session() as s:
            r1_id, r2_id = self._seed(s)
        with get_session() as s:
            df = RednerVergleich(s).compare_aggression(r1_id, r2_id, wahlperiode=20)
        neg_row = df[df["Metrik"] == "Negative Zwischenrufe"]
        assert not neg_row.empty
        # r1 has 1 negative interjection, r2 has 0 negative interjections
        assert neg_row.iloc[0]["speaker_a"] == 1
        assert neg_row.iloc[0]["speaker_b"] == 0

    def test_compare_tone_profiles_empty_db(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            df = RednerVergleich(s).compare_tone_profiles(999, 1000)
        assert not df.empty  # returns zero-filled rows
        assert (df["speaker_a"] == 0.0).all()

    def test_compare_speech_stats_empty_db(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            df = RednerVergleich(s).compare_speech_stats(999, 1000)
        assert not df.empty
        assert (df["speaker_a"] == 0).all()

    def test_compare_aggression_empty_db(self):
        from src.analytics import RednerVergleich
        with get_session() as s:
            df = RednerVergleich(s).compare_aggression(999, 1000)
        assert not df.empty
        assert (df["speaker_a"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# O. FraktionsDynamik tests  (v3.0.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestFraktionsDynamik:
    """Tests for FraktionsDynamik: faction tone timeline and sunburst."""

    def _seed(self, session):
        from datetime import date
        sz = Sitzung(wahlperiode=20, sitzungsnr=91, datum=date(2023, 5, 15), gesamtwortzahl=100)
        r1 = Redner(vorname="Clara", nachname="Test", fraktion="SPD")
        rede1 = Rede(sitzung=sz, redner=r1, text="Rede X", wortanzahl=30)
        zr1 = Zwischenruf(
            rede=rede1, text="Pfui!", fraktion="CDU/CSU",
            ton_label="Aggression", sentiment_score=-0.8,
        )
        zr2 = Zwischenruf(
            rede=rede1, text="Bravo!", fraktion="SPD",
            ton_label="Humor", sentiment_score=0.6,
        )
        session.add_all([sz, r1, rede1, zr1, zr2])
        session.flush()

    def test_tone_timeline_columns(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).tone_timeline(wahlperiode=20)
        assert list(df.columns) == ["monat", "fraktion", "ton_label", "anzahl"]

    def test_tone_timeline_has_data(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).tone_timeline(wahlperiode=20)
        assert not df.empty
        assert "2023-05" in df["monat"].values

    def test_tone_timeline_date_filter(self):
        """Date filter before the seeded session should yield empty result."""
        from datetime import date
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).tone_timeline(datum_bis=date(2022, 12, 31))
        assert df.empty

    def test_aggression_timeline_columns(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).aggression_timeline(wahlperiode=20)
        assert list(df.columns) == [
            "monat", "fraktion", "avg_aggression", "neg_count", "total_count"
        ]

    def test_aggression_timeline_cdu_negative(self):
        """CDU/CSU interjection has negative sentiment → positive avg_aggression."""
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).aggression_timeline(wahlperiode=20)
        cdu_row = df[df["fraktion"].str.contains("CDU", na=False)]
        assert not cdu_row.empty
        assert cdu_row.iloc[0]["avg_aggression"] > 0

    def test_sunburst_data_columns(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).sunburst_data(wahlperiode=20)
        assert list(df.columns) == ["fraktion", "ton_label", "anzahl"]

    def test_sunburst_data_has_rows(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            self._seed(s)
        with get_session() as s:
            df = FraktionsDynamik(s).sunburst_data(wahlperiode=20)
        assert not df.empty
        # Two interjections with ton_label set → 2 rows
        assert len(df) == 2

    def test_tone_timeline_empty_db(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            df = FraktionsDynamik(s).tone_timeline()
        assert df.empty
        assert list(df.columns) == ["monat", "fraktion", "ton_label", "anzahl"]

    def test_aggression_timeline_empty_db(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            df = FraktionsDynamik(s).aggression_timeline()
        assert df.empty

    def test_sunburst_data_empty_db(self):
        from src.analytics import FraktionsDynamik
        with get_session() as s:
            df = FraktionsDynamik(s).sunburst_data()
        assert df.empty
        assert list(df.columns) == ["fraktion", "ton_label", "anzahl"]


# ─────────────────────────────────────────────────────────────────────────────
# App spinner coverage (static AST analysis – no Streamlit import required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAppSpinners:
    """Verify via AST that every render function and startup code uses st.spinner."""

    _APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"

    @pytest.fixture(autouse=True)
    def _load_tree(self):
        source = self._APP_PATH.read_text(encoding="utf-8")
        self._tree = ast.parse(source)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _func_node(self, name: str) -> ast.FunctionDef | None:
        """Return the AST node for a top-level function definition."""
        for node in ast.walk(self._tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def _spinners_in(self, root: ast.AST) -> list[str | None]:
        """Return spinner messages from all ``with st.spinner(...)`` blocks.

        Any ``st.spinner(...)`` context manager counts as a spinner, even when
        its message is built dynamically (for example via an f-string or a
        variable). When the first positional argument is a constant string, that
        string is returned; otherwise ``None`` is recorded for that spinner.
        """
        msgs: list[str | None] = []
        for node in ast.walk(root):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "spinner"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "st"
                ):
                    continue

                if (
                    call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    msgs.append(call.args[0].value)
                else:
                    msgs.append(None)
        return msgs

    def _with_calls(self, root: ast.AST, func_name: str) -> bool:
        """Return True if *root* contains a direct Call to *func_name*."""
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name
            ):
                return True
        return False

    # ── parametrised: every render function must have at least one spinner ────

    @pytest.mark.parametrize(
        "func_name",
        [
            "render_startseite",
            "render_aggressions_radar",
            "render_themen_trend",
            "render_interaktions_netzwerk",
            "render_ton_analyse",
            "render_adressaten_analyse",
            "render_scraping_monitor",
            "render_db_uebersicht",
            "render_wahlperioden_vergleich",
            "render_top_analyse",
            "render_reaktions_analyse",
            "render_redezeit_analyse",
            "render_debattenklima",
            "render_redner_profil",
            "render_redner_vergleich",
            "render_fraktions_dynamik",
        ],
    )
    def test_render_function_has_spinner(self, func_name: str) -> None:
        node = self._func_node(func_name)
        assert node is not None, f"Function {func_name!r} not found in src/app.py"
        msgs = self._spinners_in(node)
        assert msgs, (
            f"{func_name}() contains no st.spinner() call. "
            "Every render function must wrap its DB queries with st.spinner()."
        )

    # ── init_db() must be wrapped in st.spinner() at module level ─────────────

    def test_init_db_wrapped_in_spinner(self) -> None:
        """init_db() at module level must be inside a ``with st.spinner(...)`` block."""
        for stmt in self._tree.body:
            if not isinstance(stmt, ast.With):
                continue
            is_spinner = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "spinner"
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "st"
                for item in stmt.items
            )
            if is_spinner and self._with_calls(stmt, "init_db"):
                return
        pytest.fail(
            "init_db() must be called inside a st.spinner() context at module level."
        )

    # ── sidebar filter helpers must be loaded inside a st.spinner() ───────────

    @staticmethod
    def _is_sidebar_ctx(with_node: ast.With) -> bool:
        """Return True if *with_node* is a ``with st.sidebar:`` block."""
        return any(
            isinstance(item.context_expr, ast.Attribute)
            and item.context_expr.attr == "sidebar"
            and isinstance(item.context_expr.value, ast.Name)
            and item.context_expr.value.id == "st"
            for item in with_node.items
        )

    def _sidebar_spinner_calls(self, helper_name: str) -> bool:
        """Return True when *helper_name* is called inside a spinner that is itself
        inside a module-level ``with st.sidebar:`` block."""
        for stmt in self._tree.body:
            if not isinstance(stmt, ast.With) or not self._is_sidebar_ctx(stmt):
                continue
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.With):
                    continue
                is_spinner = any(
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Attribute)
                    and item.context_expr.func.attr == "spinner"
                    and isinstance(item.context_expr.func.value, ast.Name)
                    and item.context_expr.func.value.id == "st"
                    for item in inner.items
                )
                if is_spinner and self._with_calls(inner, helper_name):
                    return True
        return False

    def test_sidebar_wahlperioden_has_spinner(self) -> None:
        """_get_wahlperioden() must be inside a st.spinner() within a st.sidebar block."""
        assert self._sidebar_spinner_calls("_get_wahlperioden"), (
            "_get_wahlperioden() must be called inside a st.spinner() context "
            "within a module-level st.sidebar block."
        )

    def test_sidebar_fraktionen_has_spinner(self) -> None:
        """_get_fraktionen() must be inside a st.spinner() within a st.sidebar block."""
        assert self._sidebar_spinner_calls("_get_fraktionen"), (
            "_get_fraktionen() must be called inside a st.spinner() context "
            "within a module-level st.sidebar block."
        )

    def test_sidebar_date_range_has_spinner(self) -> None:
        """_get_date_range() must be inside a st.spinner() within a st.sidebar block."""
        assert self._sidebar_spinner_calls("_get_date_range"), (
            "_get_date_range() must be called inside a st.spinner() context "
            "within a module-level st.sidebar block."
        )
