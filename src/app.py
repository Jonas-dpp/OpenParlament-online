"""
Streamlit dashboard for OpenParlament.

Fourteen pages map to a landing page, the core experiments, and informational
tools:

  Willkommen:
  Startseite             – Landing page with live stats, experiments overview

  Kern-Analysen:
  Aggressions-Radar      (Experiment A)
  Themen-Trend           (Experiment B, incl. Multi-WP & Aggression-Korrelation)
  Interaktions-Netzwerk  (Experiment C, incl. NetworkX/Gephi export & Evolution)
  Tagesordnungspunkte    (Experiment H, v2.2.0)

  Sprache & Ton:
  Ton-Analyse            (Experiment D)
  Adressaten-Analyse     (Experiment E)
  Reaktions-Analyse      (Experiment I, v2.2.0)

  Parlaments-Metriken:
  Redezeit-Gerechtigkeit (Experiment M, v2.2.0)
  Debattenklima-Index    (Experiment L, v2.4.0)
  Redner-Profil          (Experiment K, v2.4.0)
  Wahlperioden-Vergleich (Experiment G)

  Werkzeuge & Daten:
  Scraping-Monitor       (Experiment F)
  DB-Übersicht           – Visual DB schema, row counts, ER diagram (v2.6.0)

Navigation uses the native ``st.navigation`` / ``st.Page`` API (Streamlit ≥ 1.36)
with five grouped categories in the sidebar.  Pages can still be reached
programmatically via ``st.switch_page()``.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as ``streamlit run src/app.py`` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sqlalchemy import func, select, inspect as sa_inspect
from typing import Optional
import datetime
import os
import requests

from src.database import get_session, init_db
from src.analytics import (
    AggressionsIndex,
    InteraktionsNetzwerk,
    ThemenKarriere,
    TonAnalyse,
    AdressatenAnalyse,
    ScrapingMonitor,
    WahlperiodenVergleich,
    TOPAnalyse,
    KategorieAnalyse,
    RedeZeitAnalyse,
    SitzungsKlima,
    RednerProfil,
)
from src.models import Sitzung, Redner, Rede, Zwischenruf

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenParlament – Bundestag Scanner",
    page_icon="🏛️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Database location / download
# ─────────────────────────────────────────────────────────────────────────────

# Easily-configurable download URL for the pre-built database release asset.
_DB_DOWNLOAD_URL = (
    "https://github.com/Jonas-dpp/OpenParlament/releases/download/v1.0/openparlament.db"
)


@st.cache_resource
def get_db_path() -> str:
    """Locate or download the combined OpenParlament database.

    Resolution order:
    1. ``data/openparlament.db``  – pre-built / developer copy
    2. ``openparlament.db``       – previously downloaded copy in the root dir
    3. Download from :data:`_DB_DOWNLOAD_URL` and save to ``openparlament.db``
    """
    db_filename = "openparlament.db"
    data_path = os.path.join("data", db_filename)
    root_path = db_filename

    if os.path.exists(data_path):
        return data_path

    if os.path.exists(root_path):
        return root_path

    tmp_path = root_path + ".part"
    try:
        os.makedirs("data", exist_ok=True)
        with st.spinner("Downloading database (this might take a moment)..."):
            response = requests.get(_DB_DOWNLOAD_URL, stream=True, timeout=120)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
        os.replace(tmp_path, data_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Could not download the OpenParlament database from {_DB_DOWNLOAD_URL}. "
            "Please check your network connection or place the file manually "
            f"in the dir '{data_path}' as '{db_filename}'. Original error: {exc}"
        ) from exc

    return data_path


# ─────────────────────────────────────────────────────────────────────────────
# Initialise DB (creates tables if absent)
# ─────────────────────────────────────────────────────────────────────────────

if not os.environ.get("OPENPARLAMENT_DB_URL"):
    os.environ["OPENPARLAMENT_DB_URL"] = "sqlite:///" + Path(get_db_path()).resolve().as_posix()

init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _get_wahlperioden() -> list[int]:
    with get_session() as session:
        rows = session.execute(
            select(Sitzung.wahlperiode).distinct().order_by(Sitzung.wahlperiode)
        ).fetchall()
    return [r[0] for r in rows] or [20]


@st.cache_data(ttl=300)
def _get_fraktionen() -> list[str]:
    with get_session() as session:
        rows = session.execute(
            select(Redner.fraktion)
            .distinct()
            .where(Redner.fraktion.isnot(None))
            .order_by(Redner.fraktion)
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=300)
def _get_date_range(wahlperiode: Optional[int] = None) -> tuple[Optional[datetime.date], Optional[datetime.date]]:
    with get_session() as session:
        stmt = select(func.min(Sitzung.datum), func.max(Sitzung.datum))
        if wahlperiode is not None:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        row = session.execute(stmt).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return (None, None)
    return row[0], row[1]


# ─────────────────────────────────────────────────────────────────────────────
# Page render functions
# ─────────────────────────────────────────────────────────────────────────────

# Registry populated after page objects are created (below).
# render_startseite reads from this dict at call-time, so forward-definition
# is safe: the registry is filled before pg.run() ever invokes the page.
_PAGE_REGISTRY: dict = {}


def _navigate_to(page_key: str) -> None:
    """Switch to a page programmatically using native st.switch_page."""
    page = _PAGE_REGISTRY.get(page_key)
    if page is not None:
        st.switch_page(page)


def render_startseite() -> None:
    # ── Global CSS for modern look ─────────────────────────────────────────────
    st.markdown(
        """
<style>
/* ── Colour tokens ── */
:root {
  --bg-card:      #111827;
  --accent-blue:  #3B82F6;
  --accent-org:   #F97316;
  --text-main:    #F3F4F6;
  --text-muted:   #9CA3AF;
  --border:       rgba(59,130,246,0.25);
}

/* ── Fade-in on load ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0);    }
}
.op-hero, .op-card, .op-stat-tile {
  animation: fadeInUp 0.55s ease both;
}
.op-card:nth-child(2) { animation-delay: 0.08s; }
.op-card:nth-child(3) { animation-delay: 0.16s; }
.op-card:nth-child(4) { animation-delay: 0.24s; }

/* ── Hero ── */
.op-hero-title {
  font-size: clamp(2.4rem, 4vw, 3.6rem);
  font-weight: 800;
  color: var(--text-main);
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.op-hero-subtitle {
  font-size: 1.15rem;
  color: #93C5FD;
  margin: 0.5rem 0 1rem;
  font-style: italic;
}
.op-hero-body {
  font-size: 1rem;
  color: var(--text-muted);
  max-width: 560px;
  line-height: 1.7;
}
.op-hero-btns { margin-top: 1.4rem; display: flex; gap: 0.75rem; flex-wrap: wrap; }
.op-btn-primary {
  background: var(--accent-blue);
  color: #fff;
  border: none;
  padding: 0.6rem 1.4rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.2s;
  text-decoration: none;
}
.op-btn-primary:hover { background: #2563EB; }
.op-btn-secondary {
  background: transparent;
  color: var(--text-main);
  border: 1.5px solid var(--border);
  padding: 0.6rem 1.4rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: border-color 0.2s;
  text-decoration: none;
}
.op-btn-secondary:hover { border-color: var(--accent-blue); }

/* ── Stat tiles ── */
.op-stat-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.op-stat-tile {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 1rem 1.5rem;
  min-width: 130px;
  flex: 1;
  text-align: center;
}
.op-stat-value {
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent-blue);
  line-height: 1.2;
}
.op-stat-label {
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-top: 0.3rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Analysis cards ── */
.op-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0; }
.op-card {
  background: var(--bg-card);
  border: 1.5px solid var(--border);
  border-radius: 0.9rem;
  padding: 1.25rem 1.4rem 1rem;
  cursor: pointer;
  transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}
.op-card:hover {
  transform: translateY(-5px);
  border-color: var(--accent-blue);
  box-shadow: 0 12px 32px rgba(0,0,0,0.45);
}
.op-card-icon { font-size: 2rem; line-height: 1; }
.op-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-main);
  margin: 0.55rem 0 0.35rem;
}
.op-card-desc { font-size: 0.84rem; color: var(--text-muted); line-height: 1.5; }
.op-card-tag {
  display: inline-block;
  background: rgba(59,130,246,0.15);
  color: var(--accent-blue);
  border-radius: 0.3rem;
  padding: 0.15rem 0.5rem;
  font-size: 0.72rem;
  font-weight: 600;
  margin-top: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # ── Live stats ─────────────────────────────────────────────────────────────
    with get_session() as session:
        monitor = ScrapingMonitor(session)
        df_overview = monitor.overview()
        df_zr = monitor.zwischenruf_stats()

    total_sitzungen  = int(df_overview["sitzungen"].sum()) if not df_overview.empty else 0
    total_reden      = int(df_overview["reden"].sum())     if not df_overview.empty else 0
    total_zr         = int(df_zr["gesamt"].sum())          if not df_zr.empty     else 0
    total_wp         = len(df_overview)

    # ── Hero ───────────────────────────────────────────────────────────────────
    hero_left, hero_right = st.columns([3, 2], gap="large")

    with hero_left:
        st.markdown(
            """
<div class="op-hero">
  <div class="op-hero-title">🏛️ OpenParlament</div>
  <div class="op-hero-subtitle">Demokratie-Mining — die digitale Röntgenaufnahme des Bundestags</div>
  <div class="op-hero-body">
    Analysiere Plenarprotokolle des Deutschen Bundestags mit NLP.
    Zwischenrufe, Themenkonjunkturen und rhetorische Muster —
    <strong>100 % lokal</strong>, ohne API-Kosten, mit quelloffenen KI-Modellen.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        _btn_col1, _btn_col2, _ = st.columns([1, 1, 2])
        with _btn_col1:
            if st.button("🔥 Analysen starten", key="hero_btn_start", width="stretch"):
                _navigate_to("aggressions-radar")
        with _btn_col2:
            if st.button("🗄️ Daten erkunden", key="hero_btn_db", width="stretch"):
                _navigate_to("db-uebersicht")

    with hero_right:
        # Mini radar chart showing tone label placeholder (or empty-state)
        _radar_labels  = ["Aggression", "Sarkasmus", "Humor", "Neutral", "Aggression"]
        if total_reden > 0:
            _radar_values = [0.32, 0.18, 0.12, 0.38, 0.32]
        else:
            _radar_values = [0.25, 0.25, 0.25, 0.25, 0.25]
        _fig_hero = go.Figure(go.Scatterpolar(
            r=_radar_values,
            theta=_radar_labels,
            fill="toself",
            line_color="#3B82F6",
            fillcolor="rgba(59,130,246,0.2)",
        ))
        _fig_hero.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 0.5], showticklabels=False, gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=260,
            title=dict(text="Ton-Profil (Beispiel)", font=dict(color="#9CA3AF", size=12)),
        )
        st.plotly_chart(_fig_hero, width="stretch")

    st.divider()

    # ── Stat tiles ─────────────────────────────────────────────────────────────
    st.subheader("📊 Aktueller Datenbestand")

    if df_overview.empty:
        st.warning(
            "⚠️ Die Datenbank ist noch leer. "
            "Bitte zuerst den Scraper ausführen: "
            "`python scripts/run_scraper.py --wahlperiode 20 --nlp`"
        )
    else:
        st.markdown(
            f"""
<div class="op-stat-grid">
  <div class="op-stat-tile">
    <div class="op-stat-value">{total_sitzungen:,}</div>
    <div class="op-stat-label">📋 Sitzungen</div>
  </div>
  <div class="op-stat-tile">
    <div class="op-stat-value">{total_reden:,}</div>
    <div class="op-stat-label">🎤 Reden</div>
  </div>
  <div class="op-stat-tile">
    <div class="op-stat-value">{total_zr:,}</div>
    <div class="op-stat-label">💬 Zwischenrufe</div>
  </div>
  <div class="op-stat-tile">
    <div class="op-stat-value">{total_wp}</div>
    <div class="op-stat-label">🗓️ Wahlperioden</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Datenbestand nach Wahlperiode anzeigen"):
            st.dataframe(df_overview, width="stretch", hide_index=True)

    st.divider()

# ── Analysis hover cards ────────────────────────────────────────────────────
    st.subheader("🚀 Schnelleinstieg — Hauptanalysen")

    _card_data = [
        ("🔥", "Aggressions-Radar",    "Wer stört am meisten? Analysiere die Schärfe der Debatten.",   "aggressions-radar",     "Kern-Analysen"),
        ("📈", "Themen-Trend",         "Verfolge die Konjunktur (Veränderung) politischer Schlagworte im Parlament über die Zeit.",  "themen-trend",          "Kern-Analysen"),
        ("🎤", "Redner-Profil",        "Das rhetorische DNA-Profil jedes Abgeordneten — Aggression, Sarkasmus, Humor.", "redner-profil",         "Parlaments-Metriken"),
        ("🕸️", "Interaktions-Netzwerk", "Welche Fraktionen interagieren am häufigsten miteinander?",      "interaktions-netzwerk", "Kern-Analysen"),
    ]

    cols = st.columns(4, gap="small")
    for col, (icon, title, desc, page_key, group) in zip(cols, _card_data):
        with col:
            st.markdown(
                f"""
<div class="op-card">
  <div class="op-card-icon">{icon}</div>
  <div class="op-card-title">{title}</div>
  <div class="op-card-desc">{desc}</div>
  <div class="op-card-tag">{group}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"→ Öffnen", key=f"card_btn_{page_key}", width="stretch"):
                _navigate_to(page_key)

    st.divider()

    # ── Alle Analysen im Überblick ─────────────────────────────────────────────
    st.subheader("🔬 Alle 12 Analysen im Überblick")

    _experiments = [
        ("🔥", "Aggressions-Radar",      "Wer kassiert / verteilt die meisten negativen Zwischenrufe?",                   "v1.0.0"),
        ("📈", "Themen-Trend",            "Wann dominieren welche Schlagworte? (inkl. Multi-WP & Reizwort-Index)",        "v1.0.0 / v2.3.0"),
        ("🕸️", "Interaktions-Netzwerk",   "Welche Fraktionen unterbrechen sich wie oft? (inkl. Gephi-Export)",            "v1.0.0"),
        ("🎭", "Ton-Analyse",             "Wie ist der rhetorische Ton: Aggression, Sarkasmus, Humor oder Neutral?",      "v1.2.0"),
        ("🎯", "Adressaten-Analyse",      "An wen richtet sich ein Zwischenruf oder eine Rede?",                          "v1.2.0"),
        ("⚖️", "Wahlperioden-Vergleich",  "Wie unterscheiden sich verschiedene Legislaturperioden in Ton & Aktivität?",   "v2.0.0"),
        ("🏛️", "Tagesordnungspunkte",     "Welche Agenda-Items provozieren die meisten negativen Reaktionen?",            "v2.2.0"),
        ("👏", "Reaktions-Analyse",       "Wer produziert Beifall, wer Widerspruch? Civility-Index pro Fraktionspaar.",   "v2.2.0"),
        ("⏱️", "Redezeit-Gerechtigkeit",  "Wird Redezeit proportional zur Fraktionsgröße verteilt?",                      "v2.2.0"),
        ("🌡️", "Debattenklima-Index",     "Wie heiß war das Parlament pro Sitzung? (Composite Temperatur-Index)",         "v2.4.0"),
        ("🎤", "Redner-Profil",           "Was ist das rhetorische DNA-Profil jedes Abgeordneten?",                       "v2.4.0"),
        ("📊", "Scraping-Monitor",        "Wie ist der Datenbestand und NLP-Abdeckungsgrad?",                             "v1.2.0"),
        ("🗄️", "DB-Übersicht",            "Übersicht über die Datenbank und ihre Inhalte.",                               "v2.5.0")
    ]
    df_exp = pd.DataFrame(_experiments, columns=["Icon", "Analyse", "Frage", "Seit"])
    st.dataframe(df_exp, width="stretch", hide_index=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏗️ Tech Stack")

    _tech = [
        ("SQLite + SQLAlchemy 2.x",          "Lokale relationale Datenbank, ORM"),
        ("Requests + BeautifulSoup (lxml)",  "Bundestag Open-Data Scraping"),
        ("HuggingFace distilbert",           "Sentiment-Analyse (lokal, offline)"),
        ("HuggingFace mDeBERTa-v3",          "Ton-Klassifikation: Aggression / Sarkasmus / Humor / Neutral (Zero-Shot)"),
        ("HuggingFace xlm-roberta NER",      "Adressaten-Erkennung (Fraktionen & Personen)"),
        ("Pandas + NetworkX",                "Aggregation, Gruppenauswertungen, Graph-Export (GraphML / GEXF)"),
        ("Streamlit ≥ 1.36 + Plotly",        "Interaktives Web-Dashboard mit nativer Navigation"),
    ]
    df_tech = pd.DataFrame(_tech, columns=["Technologie", "Zweck"])
    st.dataframe(df_tech, width="stretch", hide_index=True)

    # ── Quick Start Guide ──────────────────────────────────────────────────────
    with st.expander("💻 Quick Start — So befüllst du die Datenbank"):
        st.markdown(
            """
```bash
# 1. Datenbank initialisieren
python scripts/db_init.py

# 2. Protokolle scrapen (z. B. 20. Wahlperiode, ca. 50 Protokolle)
python scripts/run_scraper.py --wahlperiode 20 --max-pages 5

# 3. Mit NLP-Analyse (Sentiment, Ton, Adressaten) — empfohlen
python scripts/run_scraper.py --wahlperiode 20 --nlp

# 4. NLP nachträglich auf bestehenden Daten ausführen
python scripts/run_nlp_cli.py --target all --batch-size 64
```
            """
        )


def render_aggressions_radar(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
    fraktion_filter: Optional[str],
) -> None:
    st.header("🔥 Aggressions-Radar")
    st.markdown(
        "Wer kassiert die meisten **negativen Zwischenrufe**? "
        "Welche Fraktion stört am häufigsten?"
    )

    col_left, col_right = st.columns(2)
    top_n = col_left.slider("Top N Abgeordnete", 5, 50, 15)
    show_normalized = col_right.toggle("Normiert (pro 100 Wörter)", value=True)

    with get_session() as session:
        analyzer = AggressionsIndex(session)
        df_targets = analyzer.top_targets(
            n=top_n,
            fraktion_filter=fraktion_filter,
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=selected_wp,
        )
        df_interruptors = analyzer.top_interruptors(
            n=top_n,
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=selected_wp,
        )

    if df_targets.empty:
        st.info("Keine Daten für den gewählten Filter. Bitte zuerst den Scraper und die NLP-Pipeline ausführen.")
    else:
        df_targets["name"] = df_targets["vorname"] + " " + df_targets["nachname"]
        y_col = "neg_pro_100_worte" if show_normalized else "neg_zwischenrufe"
        y_label = "Neg. Zwischenrufe pro 100 Wörter" if show_normalized else "Anzahl neg. Zwischenrufe"

        fig_targets = px.bar(
            df_targets,
            x="name",
            y=y_col,
            color="fraktion",
            title="Abgeordnete mit den meisten negativen Zwischenrufen",
            labels={"name": "Abgeordnete/r", y_col: y_label},
        )
        fig_targets.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_targets, width="stretch")

    st.subheader("Störer-Rangliste nach Fraktion")
    if df_interruptors.empty:
        st.info("Keine Daten vorhanden.")
    else:
        fig_int = px.bar(
            df_interruptors,
            x="fraktion",
            y="neg_zwischenrufe",
            color="anteil_negativ",
            color_continuous_scale="RdYlGn_r",
            title="Fraktionen mit den meisten negativen Zwischenrufen",
            labels={
                "fraktion": "Fraktion",
                "neg_zwischenrufe": "Anzahl neg. Zwischenrufe",
                "anteil_negativ": "Anteil negativ (%)",
            },
        )
        st.plotly_chart(fig_int, width="stretch")


def render_themen_trend(selected_wp: int) -> None:
    st.header("📈 Themen-Trend")
    st.markdown(
        "Verfolge die Konjunktur politischer Schlagworte über die Zeit. "
        "Die Häufigkeit wird pro 1 000 Wörter normiert."
    )

    keyword_input = st.text_input(
        "Schlagwörter (kommagetrennt)",
        value="Klimaschutz, Migration, Wirtschaft",
    )
    keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

    if keywords:
        all_series: list[pd.DataFrame] = []
        with get_session() as session:
            analyzer = ThemenKarriere(session)
            for kw in keywords:
                df_kw = analyzer.keyword_trend(kw, wahlperiode=selected_wp)
                if not df_kw.empty:
                    df_kw["keyword"] = kw
                    all_series.append(df_kw)

        if all_series:
            df_all = pd.concat(all_series, ignore_index=True)
            df_all = df_all.dropna(subset=["datum"])
            fig_trend = px.line(
                df_all,
                x="datum",
                y="normiert",
                color="keyword",
                title="Keyword-Häufigkeit (normiert pro 1 000 Wörter)",
                labels={"datum": "Sitzungsdatum", "normiert": "Häufigkeit / 1 000 Wörter", "keyword": "Schlagwort"},
                markers=True,
            )
            st.plotly_chart(fig_trend, width="stretch")

            with st.expander("Rohdaten anzeigen"):
                st.dataframe(df_all, width="stretch")
        else:
            st.info("Keine Daten für die eingegebenen Schlagwörter gefunden.")
    else:
        st.warning("Bitte mindestens ein Schlagwort eingeben.")

    st.divider()

    # ── Multi-WP Themen-Karriere ───────────────────────────────────────────────
    st.subheader("📅 Multi-Wahlperioden Themen-Karriere")
    st.markdown(
        "Verfolge ein Schlagwort **über alle Wahlperioden** auf einer einzigen Zeitachse. "
        "Jede Farbe entspricht einer Legislaturperiode."
    )

    mwp_keyword = st.text_input(
        "Schlagwort (Multi-WP)",
        value="Klimaschutz",
        key="mwp_keyword",
    )

    if mwp_keyword.strip():
        with get_session() as session:
            tk_mwp = ThemenKarriere(session)
            df_mwp = tk_mwp.multi_wp_keyword_trend(mwp_keyword.strip())
            df_peaks = tk_mwp.keyword_peak_by_wp(mwp_keyword.strip())

        if not df_mwp.empty:
            df_mwp = df_mwp.dropna(subset=["datum"])
            df_mwp["wahlperiode"] = df_mwp["wahlperiode"].astype(str)
            fig_mwp = px.line(
                df_mwp,
                x="datum",
                y="normiert",
                color="wahlperiode",
                title=f'"{mwp_keyword}" – Häufigkeit über alle Wahlperioden',
                labels={
                    "datum": "Sitzungsdatum",
                    "normiert": "Häufigkeit / 1 000 Wörter",
                    "wahlperiode": "Wahlperiode",
                },
                markers=True,
            )
            st.plotly_chart(fig_mwp, width="stretch")

            if not df_peaks.empty:
                with st.expander("Peak-Sitzungen je Wahlperiode"):
                    st.dataframe(df_peaks, width="stretch")
        else:
            st.info("Keine Multi-WP-Daten für dieses Schlagwort gefunden.")

    st.divider()

    # ── Themen-Sentiment-Korrelation ───────────────────────────────────────────
    st.subheader("🔥 Reizwort-Analyse: Keyword vs. Aggression")
    st.markdown(
        "Welche Schlagwörter lösen im Schnitt **aggressivere** Reaktionen aus? "
        "Vergleich der Interjektions-Stimmung bei Reden *mit* vs. *ohne* das Keyword."
    )

    polarizing_input = st.text_input(
        "Schlagwörter zum Vergleich (kommagetrennt)",
        value="Migration, Klimaschutz, Wirtschaft, AfD, Bürgergeld",
        key="polarizing_kw",
    )
    polarizing_keywords = [k.strip() for k in polarizing_input.split(",") if k.strip()]

    if polarizing_keywords:
        with get_session() as session:
            tk_polar = ThemenKarriere(session)
            df_polar = tk_polar.most_polarizing_keywords(
                polarizing_keywords, wahlperiode=selected_wp
            )

        if not df_polar.empty:
            fig_polar = px.bar(
                df_polar,
                x="keyword",
                y="delta_aggression",
                color="delta_aggression",
                color_continuous_scale="RdYlGn_r",
                title="Reizwort-Index: Aggressions-Delta über Baseline",
                labels={
                    "keyword": "Schlagwort",
                    "delta_aggression": "Aggressions-Delta (Keyword − Baseline)",
                },
                text_auto=".3f",
            )
            fig_polar.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_polar, width="stretch")

            with st.expander("Details je Schlagwort"):
                for kw in polarizing_keywords:
                    with get_session() as session:
                        df_corr = ThemenKarriere(session).keyword_aggression_correlation(
                            kw, wahlperiode=selected_wp
                        )
                    if not df_corr.empty:
                        st.write(f"**{kw}**")
                        st.dataframe(df_corr, width="stretch")
        else:
            st.info(
                "Keine Aggression-Korrelations-Daten gefunden. "
                "Bitte zuerst die NLP-Sentiment-Pipeline ausführen."
            )
    else:
        st.warning("Bitte mindestens ein Schlagwort eingeben.")


def render_interaktions_netzwerk(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("🕸️ Interaktions-Netzwerk")
    st.markdown(
        "Adjazenzmatrix: **Zeilen** = Redende Fraktion, **Spalten** = Störende Fraktion. "
        "Eigen-Interaktionen (Beifall der eigenen Fraktion) sind ausgeblendet. "
        "Werte = Anzahl Zwischenrufe (oder Aggressions-Intensität)."
    )

    score_weighted = st.toggle("Aggressions-gewichtet (−avg Sentiment statt Anzahl)", value=False)
    per_capita = st.toggle("Pro-Kopf-Normalisierung (÷ Fraktionsgröße)", value=False)

    with get_session() as session:
        netzwerk = InteraktionsNetzwerk(session)
        matrix = netzwerk.adjacency_matrix(
            datum_von=datum_von,
            datum_bis=datum_bis,
            score_weighted=score_weighted,
            wahlperiode=selected_wp,
            per_capita=per_capita,
        )
        edge_df = netzwerk.edge_list(
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=selected_wp,
        )

    if matrix.empty:
        st.info("Keine Netzwerk-Daten vorhanden.")
    else:
        color_label = "Aggressions-Intensität" if score_weighted else "Anzahl Zwischenrufe"
        if per_capita:
            color_label += " pro Mitglied"
        display_matrix = matrix if score_weighted else np.log1p(matrix)
        if not score_weighted:
            color_label = f"log(1 + {color_label})"
        fig_heatmap = px.imshow(
            display_matrix,
            color_continuous_scale="Reds" if score_weighted else "Blues",
            title="Fraktions-Interaktionsmatrix",
            labels={"x": "Störende Fraktion", "y": "Redende Fraktion", "color": color_label},
            aspect="auto",
        )
        st.plotly_chart(fig_heatmap, width="stretch")

        with st.expander("Edge-Liste (für Gephi / NetworkX Export)"):
            st.dataframe(edge_df, width="stretch")
            csv = edge_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Edge-Liste als CSV herunterladen",
                data=csv,
                file_name="openparlament_edges.csv",
                mime="text/csv",
            )
            graphml_bytes = netzwerk.to_graphml_bytes(
                datum_von=datum_von,
                datum_bis=datum_bis,
                wahlperiode=selected_wp,
            )
            st.download_button(
                "⬇️ Graph als GraphML herunterladen (NetworkX / yEd)",
                data=graphml_bytes,
                file_name="openparlament_graph.graphml",
                mime="application/xml",
            )
            gexf_bytes = netzwerk.to_gexf_bytes(
                datum_von=datum_von,
                datum_bis=datum_bis,
                wahlperiode=selected_wp,
            )
            st.download_button(
                "⬇️ Graph als GEXF herunterladen (Gephi)",
                data=gexf_bytes,
                file_name="openparlament_graph.gexf",
                mime="application/xml",
            )

    st.divider()

    # ── Netzwerk-Evolution ────────────────────────────────────────────────────
    st.subheader("📅 Netzwerk-Evolution")
    st.markdown(
        "Wie entwickeln sich die **Fraktionsbeziehungen** im Lauf einer Wahlperiode? "
        "Wähle ein Zeitfenster und navigiere mit dem Schieberegler durch die Geschichte."
    )

    evo_col1, evo_col2 = st.columns(2)
    evo_window = evo_col1.selectbox(
        "Zeitfenster",
        ["quarter", "year"],
        format_func=lambda x: "Quartal (3 Monate)" if x == "quarter" else "Jahr",
        key="evo_window",
    )
    evo_weighted = evo_col2.toggle(
        "Aggressions-gewichtet", value=True, key="evo_weighted"
    )

    with get_session() as session:
        netzwerk_evo = InteraktionsNetzwerk(session)
        evo_windows = netzwerk_evo.adjacency_matrix_by_window(
            wahlperiode=selected_wp,
            window=evo_window,
            score_weighted=evo_weighted,
        )

    if not evo_windows:
        st.info(
            "Keine Netzwerk-Evolutionsdaten vorhanden. "
            "Bitte zuerst Protokolle importieren."
        )
    else:
        evo_labels = list(evo_windows.keys())
        evo_idx = st.slider(
            "Zeitfenster auswählen",
            min_value=0,
            max_value=len(evo_labels) - 1,
            value=len(evo_labels) - 1,
            format="%d",
            key="evo_slider",
        )
        selected_label = evo_labels[evo_idx]
        st.caption(f"**Zeitfenster:** {selected_label}")
        evo_matrix = evo_windows[selected_label]
        evo_display_matrix = evo_matrix if evo_weighted else np.log1p(evo_matrix)
        evo_color_label = "Aggressions-Score" if evo_weighted else "log(1 + Anzahl Zwischenrufe)"

        fig_evo_heat = px.imshow(
            evo_display_matrix,
            color_continuous_scale="Reds" if evo_weighted else "Blues",
            title=f"Interaktions-Matrix: {selected_label}",
            labels={
                "x": "Störende Fraktion",
                "y": "Redende Fraktion",
                "color": evo_color_label,
            },
            aspect="auto",
        )
        st.plotly_chart(fig_evo_heat, width="stretch")

        # Interactive Plotly network graph for the selected window
        st.subheader(f"Interaktives Netzwerk-Diagramm: {selected_label}")
        if evo_weighted:
            st.markdown(
                "Knoten = Fraktionen. **Kantendicke** = Aggressions-Score-Betrag. "
                "**Kantenfarbe**: rot = hohe Aggression, grün = kooperativ/positiv."
            )
        else:
            st.markdown(
                "Knoten = Fraktionen. **Kantendicke** = Anzahl Zwischenrufe. "
                "**Kantenfarbe**: dunkel = mehr Interaktion, hell = weniger."
            )

        import math
        # Build positions using a deterministic sorted order so the layout
        # stays stable as the user moves the timeline slider.
        factions = sorted(
            set(evo_matrix.index.tolist()) | set(evo_matrix.columns.tolist())
        )
        n_fac = len(factions)
        pos = {
            f: (math.cos(2 * math.pi * i / max(n_fac, 1)),
                math.sin(2 * math.pi * i / max(n_fac, 1)))
            for i, f in enumerate(factions)
        }

        import plotly.graph_objects as go

        edge_traces = []
        abs_max = max(abs(evo_matrix.values).max(), 1e-9)
        for src in evo_matrix.index:
            for tgt in evo_matrix.columns:
                val = evo_matrix.loc[src, tgt]
                if val == 0:
                    continue
                x0, y0 = pos.get(src, (0, 0))
                x1, y1 = pos.get(tgt, (0, 0))
                # Width always based on absolute magnitude (volume).
                width = max(1, abs(val) / abs_max * 8)
                # Colour encodes sign: positive aggression → red, negative → green.
                if evo_weighted:
                    norm = abs(val) / abs_max
                    r_c = int(200 * norm + 55) if val > 0 else 55
                    g_c = 55 if val > 0 else int(200 * norm + 55)
                else:
                    norm = abs(val) / abs_max
                    r_c = int(200 * norm + 55)
                    g_c = int(200 * (1 - norm) + 55)
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line={"width": width, "color": f"rgb({r_c},{g_c},80)"},
                        hoverinfo="text",
                        text=f"{src} → {tgt}: {val:.2f}",
                        showlegend=False,
                    )
                )

        node_x = [pos[f][0] for f in factions]
        node_y = [pos[f][1] for f in factions]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=factions,
            textposition="top center",
            hoverinfo="text",
            marker={
                "size": 18,
                "color": "#1f77b4",
                "line": {"width": 2, "color": "white"},
            },
            showlegend=False,
        )

        fig_network = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=f"Interaktions-Netzwerk: {selected_label}",
                showlegend=False,
                hovermode="closest",
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
            ),
        )
        st.plotly_chart(fig_network, width="stretch")


def render_ton_analyse(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("🎭 Ton-Analyse")
    st.markdown(
        "Wie aggressiv, sarkastisch oder humorvoll sind die Zwischenrufe? "
        "Welche Fraktion fällt durch welchen Ton auf?"
    )

    with get_session() as session:
        ton_analyzer = TonAnalyse(session)
        df_ton_frak = ton_analyzer.ton_by_fraktion(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )
        df_ton_trend = ton_analyzer.ton_trend(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )

    if df_ton_frak.empty:
        st.info(
            "Keine Ton-Daten vorhanden. Bitte zuerst die NLP-Pipeline (ToneClassifier) ausführen."
        )
    else:
        fig_ton = px.bar(
            df_ton_frak,
            x="fraktion",
            y="anzahl",
            color="ton_label",
            barmode="stack",
            title="Ton-Verteilung nach Fraktion",
            labels={
                "fraktion": "Fraktion",
                "anzahl": "Anzahl Zwischenrufe",
                "ton_label": "Ton",
            },
            color_discrete_map={
                "Aggression": "#d62728",
                "Sarkasmus": "#ff7f0e",
                "Humor": "#2ca02c",
                "Neutral": "#aec7e8",
            },
        )
        fig_ton.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ton, width="stretch")

        if not df_ton_trend.empty:
            st.subheader("Ton-Trend über die Zeit")
            fig_ton_trend = px.line(
                df_ton_trend,
                x="datum",
                y="anzahl",
                color="ton_label",
                title="Ton-Label-Häufigkeit pro Sitzung",
                labels={
                    "datum": "Sitzungsdatum",
                    "anzahl": "Anzahl Zwischenrufe",
                    "ton_label": "Ton",
                },
                markers=True,
                color_discrete_map={
                    "Aggression": "#d62728",
                    "Sarkasmus": "#ff7f0e",
                    "Humor": "#2ca02c",
                    "Neutral": "#aec7e8",
                },
            )
            st.plotly_chart(fig_ton_trend, width="stretch")


def render_adressaten_analyse(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("🎯 Adressaten-Analyse")
    st.markdown(
        "Wer wird am häufigsten in Zwischenrufen adressiert? "
        "Welche Fraktion nimmt welche Ziele ins Visier?"
    )

    top_n_adr = st.slider("Top N Adressaten", 5, 30, 10, key="adr_top_n")

    with get_session() as session:
        adr_analyzer = AdressatenAnalyse(session)
        df_top_adr = adr_analyzer.top_adressaten(
            n=top_n_adr,
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )
        df_frak_targets = adr_analyzer.fraktion_targets_fraktion(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )

    if df_top_adr.empty:
        st.info(
            "Keine Adressaten-Daten vorhanden. Bitte zuerst die NLP-Pipeline (AddresseeDetector) ausführen."
        )
    else:
        fig_adr = px.bar(
            df_top_adr,
            x="adressat",
            y="anzahl",
            title=f"Top {top_n_adr} Adressaten von Zwischenrufen",
            labels={"adressat": "Adressat", "anzahl": "Anzahl Erwähnungen"},
            color="anzahl",
            color_continuous_scale="Blues",
        )
        fig_adr.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_adr, width="stretch")

    if not df_frak_targets.empty:
        st.subheader("Wer zielt auf wen? (Fraktion → Adressat)")
        pivot = df_frak_targets.pivot_table(
            index="fraktion",
            columns="adressat",
            values="anzahl",
            aggfunc="sum",
        )
        fig_heat = px.imshow(
            np.log1p(pivot),
            color_continuous_scale="Reds",
            title="Ziel-Matrix: Störende Fraktion vs. Adressat",
            labels={"x": "Adressat", "y": "Störende Fraktion", "color": "log(1 + Anzahl)"},
            aspect="auto",
        )
        st.plotly_chart(fig_heat, width="stretch")


def render_scraping_monitor() -> None:
    st.header("📊 Scraping-Monitor")
    st.markdown(
        "Übersicht über den aktuellen Datenbestand und den Fortschritt der NLP-Pipeline."
    )

    with get_session() as session:
        monitor = ScrapingMonitor(session)
        df_overview = monitor.overview()
        df_zr_stats = monitor.zwischenruf_stats()
        df_recent = monitor.recent_sitzungen(n=25)

    if df_overview.empty:
        st.info("Die Datenbank ist leer. Bitte zuerst den Scraper ausführen.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Sitzungen gesamt", int(df_overview["sitzungen"].sum()))
        col2.metric("Reden gesamt", int(df_overview["reden"].sum()))
        if not df_zr_stats.empty:
            col3.metric("Zwischenrufe gesamt", int(df_zr_stats["gesamt"].sum()))

        st.subheader("Datenbestand nach Wahlperiode")
        fig_overview = px.bar(
            df_overview,
            x="wahlperiode",
            y=["sitzungen", "reden"],
            barmode="group",
            title="Sitzungen und Reden je Wahlperiode",
            labels={
                "wahlperiode": "Wahlperiode",
                "value": "Anzahl",
                "variable": "Kategorie",
            },
        )
        st.plotly_chart(fig_overview, width="stretch")

        if not df_zr_stats.empty:
            st.subheader("NLP-Pipeline-Abdeckung")
            df_display = df_zr_stats[
                ["wahlperiode", "gesamt", "mit_sentiment_pct", "mit_ton_label_pct", "mit_adressaten_pct"]
            ].copy()
            df_display.columns = [
                "Wahlperiode", "Zwischenrufe", "Sentiment (%)", "Ton-Label (%)", "Adressaten (%)"
            ]
            st.dataframe(df_display, width="stretch")

        if not df_recent.empty:
            st.subheader("Zuletzt importierte Sitzungen")
            st.dataframe(df_recent, width="stretch")


def render_db_uebersicht() -> None:
    st.header("🗄️ Datenbank-Übersicht")
    st.markdown(
        "Visualisierung der Datenbankarchitektur, Zeilenanzahl und Spaltenschemas. "
        "Alle Informationen werden live aus der aktiven SQLite-Datenbank gelesen."
    )

    from src.database import get_engine
    engine = get_engine()

    # ── DB-Dateigröße ──────────────────────────────────────────────────────────
    db_url = str(engine.url)
    db_path_str = db_url.replace("sqlite:///", "")
    try:
        db_size_bytes = os.path.getsize(db_path_str)
        if db_size_bytes >= 1024 ** 2:
            db_size_str = f"{db_size_bytes / 1024**2:.1f} MB"
        else:
            db_size_str = f"{db_size_bytes / 1024:.1f} KB"
    except OSError:
        db_size_str = "in-memory / unbekannt"

    # ── Zeilenzähler ──────────────────────────────────────────────────────────
    with get_session() as session:
        counts = {
            "Sitzungen":     session.execute(select(func.count()).select_from(Sitzung)).scalar() or 0,
            "Redner":        session.execute(select(func.count()).select_from(Redner)).scalar() or 0,
            "Reden":         session.execute(select(func.count()).select_from(Rede)).scalar() or 0,
            "Zwischenrufe":  session.execute(select(func.count()).select_from(Zwischenruf)).scalar() or 0,
        }

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📁 DB-Größe", db_size_str)
    c2.metric("📋 Sitzungen", f"{counts['Sitzungen']:,}")
    c3.metric("🎤 Reden", f"{counts['Reden']:,}")
    c4.metric("💬 Zwischenrufe", f"{counts['Zwischenrufe']:,}")
    c5.metric("👤 Redner", f"{counts['Redner']:,}")

    st.divider()
    
    # Create two columns of equal width
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        # ── Datenvolumen & Relationen (Sankey) ────────────────────────────────────
        st.subheader("🌊 Datenvolumen & Relationen (Sankey)")
        st.markdown(
            "Veranschaulicht die Größenverhältnisse zwischen den Tabellen. "
            "Breite der Flüsse = logarithmisch skalierte Zeilenanzahl."
        )

        _nodes = ["Sitzung", "Redner", "Rede", "Zwischenruf"]
        _node_idx = {n: i for i, n in enumerate(_nodes)}
        _links = [
            ("Sitzung",  "Rede",         counts["Reden"]),
            ("Redner",   "Rede",         counts["Reden"]),
            ("Rede",     "Zwischenruf",  counts["Zwischenrufe"]),
        ]

        import math
        _src, _tgt, _val, _lbl = [], [], [], []
        for src, tgt, val in _links:
            _src.append(_node_idx[src])
            _tgt.append(_node_idx[tgt])
            _val.append(max(1, math.log1p(val)))
            _lbl.append(f"{val:,} Einträge")

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=20,
                thickness=24,
                line=dict(color="#3B82F6", width=1.5),
                label=_nodes,
                color=["#1D4ED8", "#059669", "#7C3AED", "#EA580C"],
            ),
            link=dict(
                source=_src,
                target=_tgt,
                value=_val,
                label=_lbl,
                color=["rgba(59,130,246,0.35)", "rgba(124,58,237,0.35)", "rgba(234,88,12,0.35)"],
            ),
        ))
        fig_sankey.update_layout(
            title_text="Datenbankbeziehungen (Sitzung → Rede → Zwischenruf)",
            font_size=14,
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sankey, width="stretch")

    with col_right:
        # ── Entitäts-Beziehungs-Diagramm (ERD) ───────────────────────────────────
        from streamlit_mermaid import st_mermaid
        st.subheader("🗂️ Entitäts-Beziehungs-Diagramm (ERD)")
        st.markdown(
            "Technischer Bauplan der Datenbank mit allen Tabellen, Spalten, "
            "Datentypen, Primary Keys (PK) und Foreign Keys (FK)."
        )
        
        # Define the mermaid code without the markdown fences
        mermaid_code = """
        %%{init: {'er': {'layoutDirection': 'LR'}}}%%
        erDiagram
            SITZUNGEN ||--o{ REDEN : "enthaelt (1:n)"
            REDNER    ||--o{ REDEN : "haelt (1:n)"
            REDEN     ||--o{ ZWISCHENRUFE : "provoziert (1:n)"

            SITZUNGEN {
                int  sitzungs_id  PK
                int  wahlperiode
                int  sitzungsnr
                date datum
                str  wochentag
                str  titel
                int  gesamtwortzahl
            }

            REDNER {
                int  redner_id    PK
                str  bundestag_id
                str  vorname
                str  nachname
                str  titel
                str  fraktion
                str  partei
            }

            REDEN {
                int   rede_id           PK
                str   bundestag_rede_id
                int   sitzung_id        FK
                int   redner_id         FK
                text  text
                str   tagesordnungspunkt
                int   wortanzahl
                float sentiment_score
                str   ton_label
                json  tone_scores
                str   adressaten
            }

            ZWISCHENRUFE {
                int   ruf_id         PK
                int   rede_id        FK
                text  text
                str   fraktion
                float sentiment_score
                str   kategorie
                str   ton_label
                json  tone_scores
                str   adressaten
            }
        """
        
        # Render the diagram
        st_mermaid(mermaid_code, height="500px")  # Optional: match the height of the Sankey chart

    st.divider()
    st.subheader("📐 Schema-Übersicht")

    inspector = sa_inspect(engine)
    table_names = inspector.get_table_names()

    schema_rows = []
    for tbl in table_names:
        cols = inspector.get_columns(tbl)
        for col in cols:
            schema_rows.append({
                "Tabelle":    tbl,
                "Spalte":     col["name"],
                "Typ":        str(col["type"]),
                "Nullable":   "✓" if col.get("nullable", True) else "✗",
                "PK":         "🔑" if col.get("primary_key") else "",
            })

    if schema_rows:
        df_schema = pd.DataFrame(schema_rows)
        selected_table = st.selectbox(
            "Tabelle auswählen",
            ["Alle"] + sorted(table_names),
            key="db_schema_table_sel",
        )
        if selected_table != "Alle":
            df_schema = df_schema[df_schema["Tabelle"] == selected_table]
        st.dataframe(df_schema, width="stretch", hide_index=True)

    st.divider()


    # ── Fremdschlüssel-Übersicht ────────────────────────────────────────────────
    st.subheader("🔑 Fremdschlüssel & Constraints")
    fk_rows = []
    for tbl in table_names:
        for fk in inspector.get_foreign_keys(tbl):
            fk_rows.append({
                "Tabelle":             tbl,
                "Spalte(n)":          ", ".join(fk["constrained_columns"]),
                "→ Referenz-Tabelle": fk["referred_table"],
                "→ Spalte(n)":        ", ".join(fk["referred_columns"]),
                "Constraint-Name":    fk.get("name", ""),
            })
        for uc in inspector.get_unique_constraints(tbl):
            fk_rows.append({
                "Tabelle":             tbl,
                "Spalte(n)":          ", ".join(uc["column_names"]),
                "→ Referenz-Tabelle": "(UNIQUE constraint)",
                "→ Spalte(n)":        "",
                "Constraint-Name":    uc.get("name", ""),
            })
    if fk_rows:
        st.dataframe(pd.DataFrame(fk_rows), width="stretch", hide_index=True)
    else:
        st.info("Keine Fremdschlüssel-Constraints gefunden.")


def render_wahlperioden_vergleich() -> None:
    st.header("⚖️ Wahlperioden-Vergleich")
    st.markdown(
        "Vergleiche Aggressivität, Ton und Aktivität **über alle Wahlperioden** hinweg. "
        "Die Daten für alle Legislaturperioden in der Datenbank werden gemeinsam ausgewertet."
    )

    with get_session() as session:
        wp_analyzer = WahlperiodenVergleich(session)
        df_agg_wp = wp_analyzer.aggression_by_wp()
        df_ton_wp = wp_analyzer.ton_by_wp()
        df_activity_wp = wp_analyzer.activity_by_wp()

    if df_agg_wp.empty and df_activity_wp.empty and df_ton_wp.empty:
        st.info(
            "Keine Vergleichsdaten vorhanden. "
            "Bitte zuerst mehrere Wahlperioden mit dem Scraper importieren."
        )
    else:
        # ── Activity overview ─────────────────────────────────────────────
        if not df_activity_wp.empty:
            st.subheader("Parlamentarische Aktivität je Wahlperiode")
            col_a, col_b = st.columns(2)
            fig_act_sz = px.bar(
                df_activity_wp,
                x="wahlperiode",
                y="sitzungen",
                title="Anzahl Sitzungen je Wahlperiode",
                labels={"wahlperiode": "Wahlperiode", "sitzungen": "Sitzungen"},
                text_auto=True,
            )
            col_a.plotly_chart(fig_act_sz, width="stretch")

            fig_act_zr = px.bar(
                df_activity_wp,
                x="wahlperiode",
                y="zwischenrufe_pro_rede",
                title="Zwischenrufe pro Rede",
                labels={
                    "wahlperiode": "Wahlperiode",
                    "zwischenrufe_pro_rede": "Zwischenrufe / Rede",
                },
                text_auto=True,
                color="zwischenrufe_pro_rede",
                color_continuous_scale="Reds",
            )
            col_b.plotly_chart(fig_act_zr, width="stretch")

            with st.expander("Aktivitätsdaten als Tabelle"):
                st.dataframe(df_activity_wp, width="stretch")

        # ── Aggression comparison ─────────────────────────────────────────
        if not df_agg_wp.empty:
            st.subheader("Aggressivität im Vergleich")
            fig_agg_wp = px.bar(
                df_agg_wp,
                x="wahlperiode",
                y="avg_aggression",
                title="Durchschnittlicher Aggressions-Score je Wahlperiode",
                labels={
                    "wahlperiode": "Wahlperiode",
                    "avg_aggression": "Ø Aggressions-Score",
                },
                text_auto=".3f",
                color="avg_aggression",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig_agg_wp, width="stretch")

            fig_neg_pct = px.bar(
                df_agg_wp,
                x="wahlperiode",
                y="anteil_negativ_pct",
                title="Anteil negativer Zwischenrufe (%) je Wahlperiode",
                labels={
                    "wahlperiode": "Wahlperiode",
                    "anteil_negativ_pct": "Anteil negativ (%)",
                },
                text_auto=True,
                color="anteil_negativ_pct",
                color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig_neg_pct, width="stretch")

        # ── Tone comparison ───────────────────────────────────────────────
        if not df_ton_wp.empty:
            st.subheader("Ton-Verteilung im Vergleich")
            fig_ton_wp = px.bar(
                df_ton_wp,
                x="wahlperiode",
                y="anteil_pct",
                color="ton_label",
                barmode="stack",
                title="Ton-Label-Verteilung je Wahlperiode (%)",
                labels={
                    "wahlperiode": "Wahlperiode",
                    "anteil_pct": "Anteil (%)",
                    "ton_label": "Ton",
                },
                color_discrete_map={
                    "Aggression": "#d62728",
                    "Sarkasmus": "#ff7f0e",
                    "Humor": "#2ca02c",
                    "Neutral": "#aec7e8",
                },
            )
            st.plotly_chart(fig_ton_wp, width="stretch")


def render_top_analyse(selected_wp: int) -> None:
    st.header("🏛️ Tagesordnungspunkt-Analyse")
    st.markdown(
        "Welche **Tagesordnungspunkte** provozieren die meisten negativen Reaktionen? "
        "Analyse nach Aggressions-Intensität und Kategorie-Verteilung pro TOP."
    )

    top_n_top = st.slider("Top N Tagesordnungspunkte", 5, 30, 15, key="top_n_top")
    min_reden_top = st.slider(
        "Mindestanzahl Reden (Qualitätsfilter)", 1, 20, 3, key="min_reden_top"
    )

    with get_session() as session:
        top_analyzer = TOPAnalyse(session)
        df_top_agg = top_analyzer.aggression_by_top(
            n=top_n_top, wahlperiode=selected_wp, min_reden=min_reden_top
        )
        df_top_kat = top_analyzer.kategorie_by_top(
            n=top_n_top, wahlperiode=selected_wp, min_reden=min_reden_top
        )

    if df_top_agg.empty:
        st.info(
            "Keine Tagesordnungspunkt-Daten vorhanden. "
            "Bitte zuerst Protokolle mit dem Scraper importieren und die NLP-Pipeline ausführen."
        )
    else:
        fig_top_agg = px.bar(
            df_top_agg,
            x="avg_aggression",
            y="tagesordnungspunkt",
            orientation="h",
            color="anteil_negativ_pct",
            color_continuous_scale="Reds",
            title=f"Top {top_n_top} Reizthemen nach Aggressions-Score",
            labels={
                "avg_aggression": "Ø Aggressions-Score",
                "tagesordnungspunkt": "Tagesordnungspunkt",
                "anteil_negativ_pct": "Anteil negativ (%)",
            },
        )
        fig_top_agg.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_top_agg, width="stretch")

        with st.expander("Daten als Tabelle"):
            st.dataframe(df_top_agg, width="stretch")

    if not df_top_kat.empty:
        st.subheader("Kategorie-Verteilung je Tagesordnungspunkt")
        fig_top_kat = px.bar(
            df_top_kat,
            x="tagesordnungspunkt",
            y="anzahl",
            color="kategorie",
            barmode="stack",
            title="Reaktions-Kategorien nach Tagesordnungspunkt",
            labels={
                "tagesordnungspunkt": "TOP",
                "anzahl": "Anzahl",
                "kategorie": "Kategorie",
            },
        )
        fig_top_kat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_kat, width="stretch")


def render_reaktions_analyse(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("👏 Reaktions-Analyse")
    st.markdown(
        "Jenseits des Sentiment-Scores: Wer produziert **Beifall**, wer produziert "
        "**Widerspruch**? Welche Fraktions-Paare sind besonders höflich – oder besonders feindselig?"
    )

    mode_toggle = st.radio(
        "Perspektive",
        ["Gegeben (Störende Fraktion)", "Empfangen (Redende Fraktion)"],
        horizontal=True,
        key="reaktion_mode",
    )
    reaktion_mode = "given" if mode_toggle.startswith("Gegeben") else "received"

    with get_session() as session:
        kat_analyzer = KategorieAnalyse(session)
        df_kat_frak = kat_analyzer.kategorie_by_fraktion(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
            mode=reaktion_mode,
        )
        df_civility = kat_analyzer.beifall_widerspruch_ratio(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )
        df_lachen = kat_analyzer.lachen_by_redner(
            n=15,
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )

    if df_kat_frak.empty:
        st.info(
            "Keine Kategorie-Daten vorhanden. "
            "Bitte zuerst Protokolle importieren (Kategorie wird beim Parsing automatisch erkannt)."
        )
    else:
        fig_kat = px.bar(
            df_kat_frak,
            x="fraktion",
            y="anzahl",
            color="kategorie",
            barmode="stack",
            title=f"Kategorie-Verteilung nach Fraktion ({mode_toggle})",
            labels={
                "fraktion": "Fraktion",
                "anzahl": "Anzahl",
                "kategorie": "Kategorie",
            },
        )
        fig_kat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_kat, width="stretch")

    if not df_civility.empty:
        st.subheader("Höflichkeits-Index: Beifall/Widerspruch-Verhältnis (Fraktions-Paare)")
        pivot_civ = df_civility.pivot_table(
            index="interruptor_fraktion",
            columns="sprecher_fraktion",
            values="civility_index",
            aggfunc="mean",
        )
        log_pivot_civ = np.log1p(pivot_civ)
        fig_civ = px.imshow(
            log_pivot_civ,
            color_continuous_scale="RdYlGn",
            title="Civility-Index: Störende Fraktion → Redende Fraktion",
            labels={
                "x": "Redende Fraktion",
                "y": "Störende Fraktion",
                "color": "log(1 + Verhältnis)",
            },
            aspect="auto",
        )
        st.plotly_chart(fig_civ, width="stretch")

    if not df_lachen.empty:
        st.subheader("🤣 Die parlamentarischen Komiker (meiste Lachen-Reaktionen)")
        df_lachen["name"] = df_lachen["vorname"] + " " + df_lachen["nachname"]
        
        df_lachen["lachen_pro_rede"] = df_lachen.apply(
            lambda row: row["lachen_count"] / row["reden_count"] if row["reden_count"] > 0 else 0, 
            axis=1
        )
        
        # Optional: DataFrame nach dem neuen Wert sortieren, damit die höchsten Werte vorne stehen
        df_lachen = df_lachen.sort_values(by="lachen_pro_rede", ascending=False)

        fig_lachen = px.bar(
            df_lachen,
            x="name",
            y="lachen_pro_rede",
            color="fraktion",
            title="Abgeordnete mit den meisten Lachen-Reaktionen (Durchschnitt pro Rede)",
            labels={"name": "Abgeordnete/r", "lachen_pro_rede": "Lachen pro Rede (Ø)", "lachen_count": "Anzahl Lachen"},
        )
        fig_lachen.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lachen, width="stretch")


def render_redezeit_analyse(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("⏱️ Redezeit-Gerechtigkeit")
    st.markdown(
        "Wird Redezeit im Bundestag **proportional zur Fraktionsgröße** verteilt? "
        "Der Fairness-Index vergleicht den Wort-Anteil (NICHT DIE REDEZEIT!) jeder Fraktion mit ihrem Sitz-Anteil."
    )

    with get_session() as session:
        rz_analyzer = RedeZeitAnalyse(session)
        df_fairness = rz_analyzer.fairness_index(wahlperiode=selected_wp)
        df_verbose = rz_analyzer.top_redselige_redner(
            n=20,
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )
        df_scatter = rz_analyzer.wortanzahl_vs_zwischenrufe(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )

    if df_fairness.empty:
        st.info(
            "Keine Wortanzahl-Daten vorhanden. "
            "Bitte zuerst den Scraper ausführen."
        )
    else:
        fig_fairness = px.bar(
            df_fairness,
            x="fraktion",
            y="fairness_index",
            color="ueber_unterrepraesentation",
            title="Fairness-Index: Wort-Anteil ÷ Sitz-Anteil",
            labels={
                "fraktion": "Fraktion",
                "fairness_index": "Fairness-Index (1.0 = proportional)",
                "ueber_unterrepraesentation": "Status",
            },
            color_discrete_map={
                "Überrepräsentiert": "#d62728",
                "Proportional": "#2ca02c",
                "Unterrepräsentiert": "#1f77b4",
            },
        )
        fig_fairness.add_hline(y=1.0, line_dash="dash", line_color="gray",
                               annotation_text="Proportional (1.0)")
        fig_fairness.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_fairness, width="stretch")

        with st.expander("Fairness-Daten als Tabelle"):
            st.dataframe(df_fairness, width="stretch")

    if not df_verbose.empty:
        st.subheader("Top 20 gesprächigste Abgeordnete")
        df_verbose["name"] = df_verbose["vorname"] + " " + df_verbose["nachname"]
        fig_verbose = px.bar(
            df_verbose,
            x="name",
            y="total_worte",
            color="fraktion",
            title="Abgeordnete mit den meisten gesprochenen Wörtern",
            labels={"name": "Abgeordnete/r", "total_worte": "Gesamte Wörter"},
        )
        fig_verbose.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_verbose, width="stretch")

    if not df_scatter.empty:
        st.subheader("Wortanzahl vs. Negative Zwischenrufe")
        df_scatter["name"] = df_scatter["vorname"] + " " + df_scatter["nachname"]
        df_scatter_plot = df_scatter[df_scatter["total_worte"] > 0]
        excluded = len(df_scatter) - len(df_scatter_plot)
        if excluded:
            st.caption(f"ℹ️ {excluded} Abgeordnete ohne Wortanzahl-Daten ausgeblendet.")
        fig_scatter = px.scatter(
            df_scatter_plot,
            x="total_worte",
            y="neg_zwischenrufe",
            color="fraktion",
            hover_data=["name"],
            title="Wortanzahl vs. Negative Zwischenrufe pro Abgeordnetem",
            labels={
                "total_worte": "Gesamte Wörter",
                "neg_zwischenrufe": "Negative Zwischenrufe",
                "fraktion": "Fraktion",
            },
            log_x=True,
            log_y=True,
        )
        st.plotly_chart(fig_scatter, width="stretch")


def render_debattenklima(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("🌡️ Debattenklima-Index")
    st.markdown(
        "Wie *heiß* war das Parlament in jeder einzelnen Sitzung? "
        "Der **Temperatur-Index** kombiniert Sentiment, Aggressions-Labels, "
        "Unterbrechungsdichte und Unruhe-Anteil zu einem Composite-Score."
    )

    with get_session() as session:
        klima_analyzer = SitzungsKlima(session)
        df_klima = klima_analyzer.klima_per_sitzung(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )
        df_hottest = klima_analyzer.hottest_sessions(n=15, wahlperiode=selected_wp)

    if df_klima.empty:
        st.info(
            "Keine Klimadaten vorhanden. "
            "Bitte zuerst Protokolle importieren und die NLP-Pipeline ausführen."
        )
    else:
        df_klima_plot = df_klima.dropna(subset=["datum"])
        df_klima_plot = df_klima_plot.sort_values("datum")

        # Rolling 10-session average
        df_klima_plot["rolling_avg"] = (
            df_klima_plot["temperatur_index"].rolling(window=10, min_periods=1).mean()
        )

        fig_klima = px.line(
            df_klima_plot,
            x="datum",
            y="temperatur_index",
            title="Parlamentarischer Temperatur-Index je Sitzung",
            labels={
                "datum": "Sitzungsdatum",
                "temperatur_index": "Temperatur-Index [0–1]",
            },
            markers=True,
        )
        fig_klima.add_scatter(
            x=df_klima_plot["datum"],
            y=df_klima_plot["rolling_avg"],
            mode="lines",
            name="Ø 10 Sitzungen",
            line={"dash": "dash", "color": "red", "width": 2},
        )
        st.plotly_chart(fig_klima, width="stretch")

    if not df_hottest.empty:
        st.subheader("🔥 Die 15 heißesten Sitzungen")
        st.dataframe(df_hottest, width="stretch")


def render_redner_profil(
    selected_wp: int,
    datum_von: Optional[datetime.date],
    datum_bis: Optional[datetime.date],
) -> None:
    st.header("🎤 Redner-Profil")
    st.markdown(
        "Das rhetorische Fingerabdruck-Profil jedes Abgeordneten. "
        "Basiert auf den **Ton-Scores** (JSON) die der ToneClassifier für jede Rede berechnet hat."
    )

    with get_session() as session:
        rp_analyzer = RednerProfil(session)
        df_rp_faction = rp_analyzer.faction_profile(
            wahlperiode=selected_wp,
            datum_von=datum_von,
            datum_bis=datum_bis,
        )

        # Build speaker selector list – filtered to current wahlperiode/dates
        # so the dropdown only contains MPs who actually have speeches in scope.
        # Query starts from Sitzung (filtered first) to reduce rows early.
        from sqlalchemy import select as sa_select
        speaker_stmt = (
            sa_select(Redner.redner_id, Redner.vorname, Redner.nachname, Redner.fraktion)
            .distinct()
            .select_from(Sitzung)
            .join(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
            .join(Redner, Redner.redner_id == Rede.redner_id)
            .order_by(Redner.nachname)
        )
        if selected_wp:
            speaker_stmt = speaker_stmt.where(Sitzung.wahlperiode == selected_wp)
        if datum_von:
            speaker_stmt = speaker_stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            speaker_stmt = speaker_stmt.where(Sitzung.datum <= datum_bis)
        redner_rows = session.execute(speaker_stmt).fetchall()

    redner_options = {
        f"{r[1]} {r[2]} ({r[3] or 'unbekannt'}) [ID: {r[0]}]": r[0]
        for r in redner_rows
    }

    if not df_rp_faction.empty:
        st.subheader("Rhetorisches Profil je Fraktion")
        fig_fp = px.bar(
            df_rp_faction.melt(
                id_vars="fraktion",
                value_vars=["Aggression", "Sarkasmus", "Humor", "Neutral"],
                var_name="Ton",
                value_name="Ø Wahrscheinlichkeit",
            ),
            x="fraktion",
            y="Ø Wahrscheinlichkeit",
            color="Ton",
            barmode="stack",
            title="Ø Ton-Profil je Fraktion (normiert auf Reden mit tone_scores)",
            labels={"fraktion": "Fraktion"},
            color_discrete_map={
                "Aggression": "#d62728",
                "Sarkasmus": "#ff7f0e",
                "Humor": "#2ca02c",
                "Neutral": "#aec7e8",
            },
        )
        fig_fp.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_fp, width="stretch")

    col_tone, _ = st.columns(2)

    tone_label_sel = col_tone.selectbox(
        "Ton-Label für Ranking",
        ["Aggression", "Sarkasmus", "Humor", "Neutral"],
        key="rp_tone_label",
    )

    with get_session() as session:
        rp_analyzer2 = RednerProfil(session)
        df_rp_top = rp_analyzer2.top_speakers_by_tone(
            ton_label=tone_label_sel,
            n=15,
            wahlperiode=selected_wp,
        )

    if not df_rp_top.empty:
        st.subheader(f"Top 15 Abgeordnete nach Ø {tone_label_sel}-Wahrscheinlichkeit")
        df_rp_top["name"] = df_rp_top["vorname"] + " " + df_rp_top["nachname"]
        fig_rp_top = px.bar(
            df_rp_top,
            x="name",
            y="avg_probability",
            color="fraktion",
            title=f"Abgeordnete mit höchstem Ø {tone_label_sel}-Score",
            labels={
                "name": "Abgeordnete/r",
                "avg_probability": f"Ø {tone_label_sel}-Wahrscheinlichkeit",
            },
        )
        fig_rp_top.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_rp_top, width="stretch")
    else:
        st.info(
            "Keine tone_scores-Daten vorhanden. "
            "Bitte zuerst die NLP-Pipeline (ToneClassifier) ausführen."
        )

    st.subheader("Einzel-Redner-Profil (Radar-Chart)")
    col_search, _ = st.columns(2)

    if redner_options:
        with col_search:
            col_input, col_select = st.columns([1, 2])

            with col_input:
                search_name = st.text_input(
                    "Suchen",
                    placeholder="Name..."
                )

            filtered_options = [
                name for name in redner_options.keys()
                if search_name.lower() in name.lower()
            ]

            with col_select:
                if filtered_options:
                    selected_redner_name = st.selectbox(
                        "Abgeordnete/r",
                        filtered_options,
                        key="rp_redner_sel"
                    )
                    selected_redner_id = redner_options[selected_redner_name]
                else:
                    st.warning("Kein passender Abgeordneter gefunden.")
                    return

        with get_session() as session:
            rp_analyzer3 = RednerProfil(session)
            df_profile = rp_analyzer3.speaker_profile(
                redner_id=selected_redner_id, wahlperiode=selected_wp
            )

        if not df_profile.empty:
            import plotly.graph_objects as go
            labels = df_profile["label"].tolist()
            values = df_profile["avg_probability"].tolist()
            # Close the polygon
            labels_closed = labels + [labels[0]]
            values_closed = values + [values[0]]
            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill="toself",
                    name=selected_redner_name,
                )
            )
            fig_radar.update_layout(
                polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                title=f"Rhetorisches Profil: {selected_redner_name}",
            )
            st.plotly_chart(fig_radar, width="stretch")
        else:
            st.info(
                "Keine tone_scores für diesen Abgeordneten vorhanden. "
                "Bitte zuerst die NLP-Pipeline (ToneClassifier) auf Reden ausführen."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – global filters
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Page wrappers & global filter state
# ─────────────────────────────────────────────────────────────────────────────
# These module-level names are reassigned by the sidebar widgets further below
# (which execute before pg.run()).  All page wrappers read the current values
# at call-time, so every page always sees up-to-date filter selections.

selected_wp: int = 20
datum_von: Optional[datetime.date] = None
datum_bis: Optional[datetime.date] = None
fraktion_filter: Optional[str] = None


def _page_startseite() -> None:
    render_startseite()


def _page_aggressions_radar() -> None:
    render_aggressions_radar(selected_wp, datum_von, datum_bis, fraktion_filter)


def _page_themen_trend() -> None:
    render_themen_trend(selected_wp)


def _page_interaktions_netzwerk() -> None:
    render_interaktions_netzwerk(selected_wp, datum_von, datum_bis)


def _page_ton_analyse() -> None:
    render_ton_analyse(selected_wp, datum_von, datum_bis)


def _page_adressaten_analyse() -> None:
    render_adressaten_analyse(selected_wp, datum_von, datum_bis)


def _page_scraping_monitor() -> None:
    render_scraping_monitor()


def _page_db_uebersicht() -> None:
    render_db_uebersicht()


def _page_wahlperioden_vergleich() -> None:
    render_wahlperioden_vergleich()


def _page_top_analyse() -> None:
    render_top_analyse(selected_wp)


def _page_reaktions_analyse() -> None:
    render_reaktions_analyse(selected_wp, datum_von, datum_bis)


def _page_redezeit_gerechtigkeit() -> None:
    render_redezeit_analyse(selected_wp, datum_von, datum_bis)


def _page_debattenklima() -> None:
    render_debattenklima(selected_wp, datum_von, datum_bis)


def _page_redner_profil() -> None:
    render_redner_profil(selected_wp, datum_von, datum_bis)


# ─────────────────────────────────────────────────────────────────────────────
# Native Streamlit navigation  (st.navigation / st.Page, requires ≥ 1.36)
# ─────────────────────────────────────────────────────────────────────────────

_P_START   = st.Page(_page_startseite,             title="Startseite",              icon="🏠", default=True)

_P_AGG     = st.Page(_page_aggressions_radar,      title="Aggressions-Radar",       icon="🔥")
_P_THEMEN  = st.Page(_page_themen_trend,           title="Themen-Trend",            icon="📈")
_P_NETZ    = st.Page(_page_interaktions_netzwerk,  title="Interaktions-Netzwerk",   icon="🕸️")
_P_TOP     = st.Page(_page_top_analyse,            title="Tagesordnungspunkte",     icon="🏛️")

_P_TON     = st.Page(_page_ton_analyse,            title="Ton-Analyse",             icon="🎭")
_P_ADR     = st.Page(_page_adressaten_analyse,     title="Adressaten-Analyse",      icon="🎯")
_P_REAK    = st.Page(_page_reaktions_analyse,      title="Reaktions-Analyse",       icon="👏")

_P_REDEN   = st.Page(_page_redezeit_gerechtigkeit, title="Redezeit-Gerechtigkeit",  icon="⏱️")
_P_KLIMA   = st.Page(_page_debattenklima,          title="Debattenklima-Index",     icon="🌡️")
_P_PROFIL  = st.Page(_page_redner_profil,          title="Redner-Profil",           icon="🎤")
_P_WP      = st.Page(_page_wahlperioden_vergleich, title="Wahlperioden-Vergleich",  icon="⚖️")

_P_MONITOR = st.Page(_page_scraping_monitor,       title="Scraping-Monitor",        icon="📊")
_P_DB      = st.Page(_page_db_uebersicht,          title="DB-Übersicht",            icon="🗄️")

# Populate the registry so _navigate_to() / render_startseite() can switch pages.
_PAGE_REGISTRY.update(
    {
        "aggressions-radar":      _P_AGG,
        "themen-trend":           _P_THEMEN,
        "interaktions-netzwerk":  _P_NETZ,
        "ton-analyse":            _P_TON,
        "adressaten-analyse":     _P_ADR,
        "scraping-monitor":       _P_MONITOR,
        "wahlperioden-vergleich": _P_WP,
        "top-analyse":            _P_TOP,
        "reaktions-analyse":      _P_REAK,
        "redezeit-gerechtigkeit": _P_REDEN,
        "debattenklima":          _P_KLIMA,
        "redner-profil":          _P_PROFIL,
        "db-uebersicht":          _P_DB,
    }
)

pg = st.navigation(
    {
        "Willkommen":          [_P_START],
        "Kern-Analysen":       [_P_AGG, _P_THEMEN, _P_NETZ, _P_TOP],
        "Sprache & Ton":       [_P_TON, _P_ADR, _P_REAK],
        "Parlaments-Metriken": [_P_REDEN, _P_KLIMA, _P_PROFIL, _P_WP],
        "Werkzeuge & Daten":   [_P_MONITOR, _P_DB],
    }
)
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – global filters  (run before pg.run() so page wrappers see values)
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🏛️ OpenParlament")
st.sidebar.markdown("*Demokratie-Mining für den Bundestag*")
st.sidebar.divider()

wahlperioden = _get_wahlperioden()
selected_wp = st.sidebar.selectbox("Wahlperiode", wahlperioden, index=len(wahlperioden) - 1)

date_min, date_max = _get_date_range(selected_wp)
datum_von = None
datum_bis = None
if date_min is not None and date_max is not None:
    datum_von = st.sidebar.date_input(
        "Von", value=date_min, min_value=date_min, max_value=date_max
    )
    datum_bis = st.sidebar.date_input(
        "Bis", value=date_max, min_value=date_min, max_value=date_max
    )

fraktionen = _get_fraktionen()
selected_fraktion = st.sidebar.selectbox(
    "Fraktion (Filter Targets)", ["Alle"] + fraktionen
)
fraktion_filter = None if selected_fraktion == "Alle" else selected_fraktion

st.sidebar.divider()
st.sidebar.caption("### **Quellen:**  ")
st.sidebar.markdown(
    """
    * [Bundestag Open Data](https://www.bundestag.de/services/opendata)  
    * 🔒 [GitHub (Repo is *private* at the moment)](https://github.com/Jonas-dpp/OpenParlament) 🔒 
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Run current page
# ─────────────────────────────────────────────────────────────────────────────

pg.run()

