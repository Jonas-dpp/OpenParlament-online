# OpenParlament: Der Bundestag-Scanner

**Demokratie-Mining — die digitale Röntgenaufnahme des Bundestags.**

[![Tests](https://github.com/Jonas-dpp/OpenParlament/actions/workflows/ci.yml/badge.svg)](https://github.com/Jonas-dpp/OpenParlament/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-3.0.0-brightgreen.svg)](pyproject.toml)
[![Schülerprojekt](https://img.shields.io/badge/🎓-Schülerprojekt-blueviolet.svg](README.md)

Willkommen bei **OpenParlament** – einem **Schülerprojekt** zur NLP-gestützten Analyse der Plenarprotokolle des Deutschen Bundestags.
Das Projekt macht verborgene politische Dynamiken über ein interaktives Streamlit-Dashboard sichtbar und wird
offen auf GitHub entwickelt.

> **Version 3.0.0** — Neu: *Redner-Vergleich* (Experiment N) und *Fraktions-Dynamik* (Experiment O)
> mit Sunburst-Charts, überlagerten Radar-Charts und monatlichen Zeitreihen.
> Komplettes Codebase-Cleanup und vollständige Dokumentations-Überarbeitung.

**Vierzehn Analysen + Dashboard-Tabs:**

| # | Experiment | Frage | Version |
|---|---|---|---|
| A | 🔥 Aggressions-Radar | Wer kassiert / verteilt die meisten negativen Zwischenrufe? | v1.0.0 |
| B | 📈 Themen-Trend | Wann dominieren welche Schlagworte die Debatte? (inkl. Multi-WP & Reizwort-Index) | v1.0.0 / v2.3.0 |
| C | 🕸️ Interaktions-Netzwerk | Welche Fraktionen unterbrechen sich wie oft? (inkl. NetworkX/Gephi-Export) | v1.0.0 |
| D | 🎭 Ton-Analyse | Wie ist der rhetorische Ton: Aggression, Sarkasmus, Humor oder Neutral? | v1.2.0 |
| E | 🎯 Adressaten-Erkennung | An wen richtet sich ein Zwischenruf oder eine Rede? | v1.2.0 |
| F | 📊 Scraping-Monitor | Wie ist der Datenbestand und NLP-Abdeckungsgrad? | v1.2.0 |
| G | ⚖️ Wahlperioden-Vergleich | Wie unterscheiden sich verschiedene Legislaturperioden in Ton & Aktivität? | v2.0.0 |
| H | 🏛️ Tagesordnungspunkt-Analyse | Welche Agenda-Items provozieren die meisten negativen Reaktionen? | v2.2.0 |
| I | 👏 Reaktions-Analyse | Wer produziert Beifall, wer produziert Widerspruch? Civility-Index pro Fraktions-Paar. | v2.2.0 |
| M | ⏱️ Redezeit-Gerechtigkeit | Wird Redezeit proportional zur Fraktionsgröße verteilt? | v2.2.0 |
| L | 🌡️ Debattenklima-Index | Wie heiß war das Parlament pro Sitzung? (Composite Temperatur-Index) | v2.4.0 |
| K | 🎤 Redner-Profil | Was ist das rhetorische DNA-Profil jedes Abgeordneten? | v2.4.0 |
| N | 👥 Redner-Vergleich | Vergleiche zwei Abgeordnete direkt: Ton, Aggression, Redeaktivität | **v3.0.0** |
| O | 📡 Fraktions-Dynamik | Sunburst & Zeitreihen: Wie entwickelt sich der Ton der Fraktionen? | **v3.0.0** |
| P | 🗄️ DB-Übersicht | Datenbankschema, Zeilenzähler, Sankey-Datenfluss & ERD | v2.5.0 |

Das System arbeitet **100 % lokal** (SQLite), ohne API-Kosten und nutzt quelloffene KI-Modelle (HuggingFace).

---

## ⚠️ Disclaimer

OpenParlament ist ein **Schülerprojekt** zu Lehr- und Forschungszwecken. Die
Analyseergebnisse basieren auf automatisierten NLP-Verfahren und stellen **keine politische
Stellungnahme** dar. Die Nutzung erfolgt auf eigene Verantwortung.

## 📜 Datenquelle & Lizenz

Alle Plenarprotokolle stammen aus dem offenen Datenportal des Deutschen Bundestags:

> **Bundestag Open Data** — <https://www.bundestag.de/services/opendata>  
> Lizenz der Rohdaten: [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0)

Der Quellcode dieses Projekts steht unter der **GNU Affero General Public License v3 (AGPL-3.0-or-later)**.
Eine Kopie der Lizenz findet sich in der Datei [`LICENSE`](LICENSE) im Repository-Wurzelverzeichnis.

---

## 📁 Projektstruktur

```
OpenParlament/
├── src/
│   ├── models.py       # SQLAlchemy ORM (Sitzung, Redner, Rede, Zwischenruf)
│   ├── database.py     # DB-Engine & Session-Management
│   ├── parser.py       # Bundestag-XML-Parser (OOP)
│   ├── scraper.py      # Open-Data-Scraper
│   ├── nlp.py          # NLP-Engines: SentimentEngine, ToneClassifier, AddresseeDetector
│   ├── analytics.py    # Analyse-Module A–M (12 Analytics-Klassen)
│   └── app.py          # Streamlit-Dashboard (native st.navigation, gruppierte Sidebar)
├── scripts/
│   ├── db_init.py      # Datenbank initialisieren
│   ├── db_patch.py     # Fehlende Spalten nachträglich hinzufügen & datum/wochentag backfüllen
│   ├── import_xmls.py  # Lokale XML-Dateien importieren & fehlende Datumsfelder backfüllen
│   ├── run_scraper.py  # Protokolle herunterladen, importieren & optional NLP-Scoring
│   ├── run_nlp_cli.py  # NLP-Scoring auf bestehenden DB-Daten (standalone CLI)
│   └── nlp_session.py  # NLPSession-Kontext (CUDA-Erkennung, Engine-Lifecycle)
├── tests/
│   └── test_core.py    # Unit-Tests (237 Tests)
├── data/               # SQLite-DB und heruntergeladene XML-Dateien
├── docs/               # Lastenheft, Pflichtenheft, Project-Tracker
├── pyproject.toml      # Projekt-Konfiguration & Dependency-Deklaration
└── requirements.txt
```

---

## 💻 Quick Start

### 1. Repository klonen & Umgebung einrichten

```bash
git clone https://github.com/Jonas-dpp/OpenParlament.git
cd OpenParlament
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Datenbank initialisieren

```bash
python scripts/db_init.py
```

### 3. Protokolle herunterladen & importieren

```bash
# 20. Wahlperiode, max. 5 Seiten der Listing-Page (ca. 50 Protokolle)
python scripts/run_scraper.py --wahlperiode 20 --max-pages 5

# Inkl. automatischer NLP-Analyse (Sentiment, Ton, Adressaten)
python scripts/run_scraper.py --wahlperiode 20 --nlp

# GPU-Beschleunigung + Half-Precision (CUDA):
python scripts/run_scraper.py --wahlperiode 20 --nlp --nlp-cuda --nlp-fp16

# NLP nachträglich auf bestehenden DB-Daten ausführen:
python scripts/run_nlp_cli.py --target all --batch-size 64
```

### 4. Dashboard starten

```bash
streamlit run src/app.py
```

Die App öffnet sich automatisch unter **http://localhost:8501**.

Die Navigation erfolgt über die **Sidebar** mit gruppierten Kategorien
(native `st.navigation`-API, Streamlit ≥ 1.36).

---

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## Direkt alles durchrattern:
```bash
pip install -r requirements.txt
python scripts/db_init.py
python scripts/run_scraper.py --wahlperiode 17 --delay 2 --max-pages 7 --nlp
python scripts/run_scraper.py --wahlperiode 18 --max-pages 7 --delay 2 --nlp
python scripts/run_scraper.py --wahlperiode 19 --max-pages 7 --delay 2 --nlp
python scripts/run_scraper.py --wahlperiode 20 --max-pages 7 --delay 2 --nlp
python scripts/run_scraper.py --wahlperiode 21 --max-pages 5 --delay 2 --nlp
streamlit run src/app.py
```
> Die Tests verwenden eine In-Memory-SQLite-Datenbank – kein torch/GPU nötig.

---

## 📖 Dokumentation

- 🎯 [Lastenheft (Anforderungen)](docs/Lastenheft.md)
- 🛠️ [Pflichtenheft (Technische Umsetzung)](docs/Pflichtenheft.md)
- 🗺️ [Project Tracker & Architecture](docs/Project.md)

---

## 🏗️ Tech Stack

| Schicht | Technologie | Zweck |
|---|---|---|
| Datenhaltung | SQLite + SQLAlchemy | Lokale relationale Datenbank, ORM |
| Datenbeschaffung | Requests + BeautifulSoup | Bundestag Open Data scraping |
| Parsing | BeautifulSoup (lxml) + Regex | XML-Protokoll-Zerlegung |
| NLP – Sentiment | HuggingFace `distilbert-base-multilingual-cased` | Sentiment-Analyse (lokal, offline) |
| NLP – Ton-Analyse | HuggingFace `mDeBERTa-v3` (Zero-Shot) | Klassifikation: Aggression / Sarkasmus / Humor / Neutral |
| NLP – Adressaten | Regel-Engine + HuggingFace `xlm-roberta-base-ner-hrl` NER | Erkennung von Fraktionen und Personen als Adressaten |
| Data Science | Pandas + NetworkX | Aggregation, Gruppenauswertungen, Graph-Export |
| Frontend | Streamlit ≥ 1.36 + Plotly | Interaktives Web-Dashboard mit nativer Navigation (`st.navigation`) |

## 📊 Analytics-Klassen

| Klasse | Tab | Beschreibung | Version |
|---|---|---|---|
| `AggressionsIndex` | A | Top-Empfänger und Top-Störer negativer Zwischenrufe | v0.8.0 |
| `ThemenKarriere` | B | Keyword-Trend, Multi-WP-Karriere, Reizwort-Korrelation | v0.8.0 / v2.3.0 |
| `InteraktionsNetzwerk` | C | Fraktions-Interaktionsmatrix + NetworkX/Gephi-Export | v0.8.0 |
| `TonAnalyse` | D | Ton-Label-Verteilung (Aggression/Sarkasmus/Humor/Neutral) | v1.2.0 |
| `AdressatenAnalyse` | E | Adressaten-Ranking und Fraktions-Adressierungs-Matrix | v1.2.0 |
| `ScrapingMonitor` | F | DB-Füllstand und NLP-Abdeckungsgrad | v1.2.0 |
| `WahlperiodenVergleich` | G | Vergleich von Aggression, Ton und Aktivität über Wahlperioden | v2.0.0 |
| `TOPAnalyse` | H | Tagesordnungspunkt-Hostility-Ranking (nutzt `Rede.tagesordnungspunkt`) | v2.2.0 |
| `KategorieAnalyse` | I | Kategorie-Verteilung & Civility-Index (nutzt `Zwischenruf.kategorie`) | v2.2.0 |
| `RedeZeitAnalyse` | M | Speech-Time-Fairness-Index (nutzt `Rede.wortanzahl`) | v2.2.0 |
| `SitzungsKlima` | L | Composite Temperatur-Index pro Sitzung | v2.4.0 |
| `RednerProfil` | K | Rhetorischer Fingerabdruck aus `Rede.tone_scores` JSON | v2.4.0 |
| `RednerVergleich` | N | Direktvergleich zweier Abgeordneter: Ton, Aggression, Rede-Aktivität | **v3.0.0** |
| `FraktionsDynamik` | O | Fraktions-Ton-Zeitreihe, Aggressions-Timeline, Sunburst-Hierarchie | **v3.0.0** |
| `DB-Übersicht` | P | Einblick in die Datenbankstruktur | v2.5.0 |

## 🗄️ DB-Übersicht (Tab P)

Die **DB-Übersicht** liefert einen vollständigen Einblick in die Datenbankstruktur:

| Abschnitt | Beschreibung |
|---|---|
| Metriken | Datenbankgröße, Zeilenzähler pro Tabelle |
| 🌊 Sankey-Diagramm | Datenvolumen & Relationen – logarithmisch skalierte Flussbreite |
| 🗂️ ERD (Mermaid) | Entitäts-Beziehungs-Diagramm mit allen Spalten, Typen, PKs und FKs |
| 📐 Schema-Übersicht | Live-Tabelle aller Spalten aus der aktiven Datenbank |
| 🔑 Fremdschlüssel | FK- und UNIQUE-Constraints tabellarisch |


---

## 📜 Changelog

### v3.0.0 (2026-04-07) — *Aktuelle Version*

**Neue Analysen (Experiment N & O):**
- **👥 Redner-Vergleich** (`RednerVergleich`): Zwei Abgeordnete direkt gegenüberstellen.
  - Überlappende Radar-Charts für Ton-Profile beider Sprecher
  - Divergenz-Balkendiagramm (A − B) je Ton-Label
  - Gruppenbalken für Redeaktivität und Aggressions-Exposition nebeneinander
- **📡 Fraktions-Dynamik** (`FraktionsDynamik`): Wie entwickelt sich der Ton der Fraktionen?
  - Sunburst-Hierarchie: Fraktion → Ton-Label → Anzahl (interaktiv)
  - Gestapeltes Flächendiagramm: monatliche Ton-Label-Häufigkeit je Fraktion
  - Linien-Zeitreihe: monatlicher Aggressions-Score je Fraktion

**UX & Codebase-Cleanup:**
- Startseite: v3.0.0-Badge, Schülerprojekt-Hinweis, Versions-Übersichtstabelle
- Alle 14 Analysen in der Übersichtstabelle auf der Startseite
- Sidebar-Gruppe *Parlaments-Metriken* erweitert um Redner-Vergleich
- Neue Sidebar-Gruppe *Werkzeuge & Daten* mit Fraktions-Dynamik, Scraping-Monitor, DB-Übersicht
- Modul-Docstrings in `analytics.py` und `app.py` aktualisiert

**Dokumentation:**
- `pyproject.toml` auf Version 3.0.0 aktualisiert
- `README.md`: neue Analysen, v3.0.0-Banner, Changelog-Abschnitt
- `docs/Project.md`: v3.0.0-Meilenstein eingetragen

---

### v2.5.0

- DB-Übersicht (Tab P): Sankey-Datenfluss, ERD (Mermaid), Schema-Inspector, Fremdschlüssel-Übersicht
- Sidebar-Spinner-Optimierungen für schnelleres initiales Rendern

### v2.4.0

- **🌡️ Debattenklima-Index** (`SitzungsKlima`): Composite Temperatur-Index pro Sitzung
- **🎤 Redner-Profil** (`RednerProfil`): Radar-Chart für rhetorischen Fingerabdruck; Top-Redner nach Ton-Label; Fraktions-Profil

### v2.3.0 (integriert in v2.x)

- Multi-WP Keyword-Trend (`ThemenKarriere.multi_wp_keyword_trend`)
- Reizwort-Index: Keyword-Aggressions-Korrelation (`most_polarizing_keywords`)
- Peak-Sitzungen je Wahlperiode

### v2.2.0

- **🏛️ Tagesordnungspunkte** (`TOPAnalyse`): Hostility-Ranking je Agenda-Item
- **👏 Reaktions-Analyse** (`KategorieAnalyse`): Beifall vs. Widerspruch, Civility-Index
- **⏱️ Redezeit-Gerechtigkeit** (`RedeZeitAnalyse`): Fairness-Index, Wortanzahl je Fraktion

### v2.0.0

- **⚖️ Wahlperioden-Vergleich** (`WahlperiodenVergleich`): Aggression, Ton und Aktivität über Legislaturperioden
- Netzwerk-Evolution (Zeitfenster-Schieberegler)
- Graph-Export: GraphML (NetworkX/yEd) und GEXF (Gephi)

### v1.2.0

- **🎭 Ton-Analyse** (`TonAnalyse`): Ton-Label-Verteilung nach Fraktion und Zeit
- **🎯 Adressaten-Erkennung** (`AdressatenAnalyse`): Top-Adressaten, Fraktions-Ziel-Matrix
- **📊 Scraping-Monitor** (`ScrapingMonitor`): NLP-Abdeckungsgrad, Übersicht nach Wahlperiode

### v1.0.0

- **🔥 Aggressions-Radar** (`AggressionsIndex`): Top-Targets und Top-Störer negativer Zwischenrufe
- **📈 Themen-Trend** (`ThemenKarriere`): Keyword-Häufigkeit normiert pro 1 000 Wörter
- **🕸️ Interaktions-Netzwerk** (`InteraktionsNetzwerk`): Fraktions-Adjazenzmatrix
