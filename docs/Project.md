# Agent Memory & Session Tracker

*Note for Dev/AI: Always review this file at the start of a session and update it before ending a session. Update the "Status" column in the Version Plan as milestones are reached.*

---

## Current Version: v2.5.0

### Completed Milestones
* [x] v0.0.0 — Project conceptualisation, Techstack defined (Python, SQLite, SQLAlchemy, Streamlit, HuggingFace).
* [x] v0.0.0 — Lastenheft und Pflichtenheft erstellt, Experimente (Aggressions-Index, Themen-Karriere, Netzwerk) definiert.
* [x] v0.1.0 — `requirements.txt` erstellt, Repository-Struktur (`src/`, `scripts/`, `data/`, `tests/`) angelegt.
* [x] v0.1.0 — SQLAlchemy ORM-Setup (`src/models.py` & `src/database.py`): Klassen `Sitzung`, `Redner`, `Rede`, `Zwischenruf` mit Relationships, vollständig OOP und typed (mapped_column).
* [x] v0.5.0 — `src/parser.py`: `BundestagXMLParser` parst strukturierte Bundestag-XML-Dateien (19./20. Wahlperiode) in ORM-Objekte.
* [x] v0.5.0 — `src/scraper.py`: `BundestagScraper` lädt Protokoll-XMLs von der Bundestag-Open-Data-Seite mit Paginierung und Duplikatschutz.
* [x] v0.5.0 — `scripts/run_scraper.py`: `ProtocolImporter` persistiert ParseResults in die DB mit Deduplizierung.
* [x] v0.8.0 — `src/nlp.py`: `SentimentEngine` kombiniert regelbasierte Schnellklassifikation mit HuggingFace-Inferenz (lazy loading, CPU-fähig).
* [x] v0.8.0 — `src/analytics.py`: `AggressionsIndex`, `ThemenKarriere`, `InteraktionsNetzwerk` — alle drei Kern-Experimente als Pandas-DataFrame-Ausgabe.
* [x] v1.0.0 — **Quality & Performance (Final Gate):**
  * Architektur-Konsistenz: Doppelte Imports in `nlp.py` entfernt; "Triple Crown" — `SentimentEngine`, `ToneClassifier` und `AddresseeDetector` folgen identischer interner Logik (regelbasierte Vorfilterung → `_neural_*_batch` → tqdm-Stream → Fehler-Fallback).
  * `AddresseeDetector` erhält `batch_size`-Parameter; `_neural_detect_batch()` extrahiert; `NLPSession` übergibt `batch_size` korrekt an alle drei Engines.
  * Shared-mutable-object-Bugs in Fehler-Fallbacks von `_neural_classify_batch`, `run_scraper.py` und `run_nlp_cli.py` behoben (List-Comprehension statt `* N`).
  * Dependency-Audit: Ungenutztes `networkx`-Paket aus `requirements.txt` entfernt.
  * Test-Suite auf 151 Tests erweitert: NLPSession-Lifecycle, neurale Fallback-Einzigartigkeit, Leerinput-Handling, Scraper-Timeouts, unerwartete NLP-Ausgabeformate.
  * `README.md` aktualisiert: korrekter NER-Modellname (`xlm-roberta-base-ner-hrl`), Projektstruktur mit `run_nlp_cli.py` / `nlp_session.py`, doppelter `--nlp`-Flag in Quick-Start entfernt.
* [x] v1.0.0 — `src/app.py`: Streamlit-Dashboard mit 3 Tabs (Aggressions-Radar, Themen-Trend, Interaktions-Netzwerk), globaler Sidebar-Filter.
* [x] v1.0.0 — `tests/test_core.py`: Vollständige Unit-Tests für DB, Models, Parser, NLP und Analytics (in-memory SQLite, kein torch nötig).
* [x] v1.0.0 — `README.md` und `docs/Project.md` aktualisiert.
* [x] v1.1.0 — **Scraper-Bugfixes:** `_OPENDATA_URL` ohne eingebettete Query-Parameter (vermeidet doppelte Parameter); Dateiname-Filter auf robustes Regex `^<wp>\d+\.xml$` umgestellt statt fragiler `startswith`-Prüfung.
* [x] v1.1.0 — **Neue NLP-Engines:** `ToneClassifier` (Aggression/Sarkasmus/Humor/Neutral via Zero-Shot-Klassifikation, `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) und `AddresseeDetector` (regelbasierte Fraktionserkennung + optionales NER via `deepset/gelectra-base-german-cased-ner`) in `src/nlp.py`.
* [x] v1.1.0 — **Modell-Erweiterung:** `Zwischenruf` erhält zwei neue Felder: `ton_label` (String, 64) und `adressaten` (String, 512, kommasepariert).
* [x] v1.1.0 — Tests für alle neuen Features (Scraper, ToneClassifier, AddresseeDetector, neue Modellfelder) – 45 Tests, alle grün.
* [x] v1.2.0 — **Wahlperioden-Bugfix:** `_get_date_range()` filtert nun nach gewählter Wahlperiode; `AggressionsIndex`, `InteraktionsNetzwerk` akzeptieren neues `wahlperiode`-Argument. Sidebar-Filter greifen jetzt in allen Tabs korrekt durch.
* [x] v1.2.0 — **Neue Analytics-Klassen:** `TonAnalyse` (Ton-Label-Verteilung nach Fraktion & Zeitverlauf), `AdressatenAnalyse` (Top-Adressaten + Fraktion-zu-Fraktion-Matrix), `ScrapingMonitor` (DB-Füllstand & NLP-Abdeckung).
* [x] v1.2.0 — **Dashboard-Erweiterung:** Tab D (🎭 Ton-Analyse), Tab E (🎯 Adressaten-Analyse), Tab F (📊 Scraping-Monitor) in `src/app.py` implementiert.
* [x] v2.0.0 — **CI/CD via GitHub Actions:** `.github/workflows/ci.yml` führt `pytest` auf Python 3.11 & 3.12 bei jedem Push und Pull-Request aus.
* [x] v2.0.0 — **`pyproject.toml`:** Saubere Projekt-Konfiguration mit Metadaten, Abhängigkeiten und `[tool.pytest.ini_options]` / `[tool.ruff]`.
* [x] v2.0.0 — **NetworkX/Gephi-Export:** `InteraktionsNetzwerk.to_networkx_graph()`, `.to_graphml_bytes()`, `.to_gexf_bytes()` exportieren den Interaktionsgraphen als `DiGraph` mit Kantenattributen (`weight`, `avg_sentiment`, `aggression_score`). Tab C bietet Download-Buttons für GraphML (NetworkX/yEd) und GEXF (Gephi).
* [x] v2.0.0 — **`WahlperiodenVergleich` Analytics:** Neue Klasse G in `analytics.py` mit drei Methoden: `aggression_by_wp()`, `ton_by_wp()`, `activity_by_wp()` — alle Wahlperioden vergleichend.
* [x] v2.0.0 — **Tab G (Wahlperioden-Vergleich):** Neuer Dashboard-Tab mit Aktivitäts-, Aggressions- und Ton-Vergleichsdiagrammen über alle Legislaturperioden.
* [x] v2.0.0 — **`networkx` zu `requirements.txt` hinzugefügt** (Version `>=3.2,<4.0`).
* [x] v2.0.0 — **Test-Suite auf 166 Tests erweitert:** `TestInteraktionsNetzwerkExport` (6 Tests) und `TestWahlperiodenVergleich` (9 Tests), alle grün.
* [x] v2.1.0 — **Sidebar-Navigation & URL-Routing:** `st.tabs` durch Sidebar-Radio-Navigation ersetzt. Alle 8 Seiten (Startseite + A–G) über URL-Query-Parameter erreichbar (z. B. `?page=themen-trend`). `DEFAULT_PAGE`-Konstante eingeführt; sicheres `.get()`-Lookup verhindert `ValueError` bei ungültigem Seitennamen.
* [x] v2.1.0 — **Startseite (Landing Page):** Neue Seite mit Projektbeschreibung, Schnelleinstieg-Buttons (Aggressions-Radar, Themen-Trend, Interaktions-Netzwerk) und Live-Datenbankstatistik.
* [x] v2.1.0 — **Render-Funktionen:** Jede Seite in eine dedizierte `render_*()`-Funktion extrahiert (`render_startseite`, `render_aggressions_radar`, `render_themen_trend`, `render_interaktions_netzwerk`, `render_ton_analyse`, `render_adressaten_analyse`, `render_scraping_monitor`, `render_wahlperioden_vergleich`). Hilfsfunktion `_navigate_to()` de-dupliziert Button-Click-Handler. Dispatch-Dict `_DISPATCH` ersetzt die if/elif-Kette.
* **Datenbeschaffung:** Das genaue URL-Muster der Bundestag Open-Data-Seite kann sich ändern. Bei Änderungen muss `_OPENDATA_URL` in `src/scraper.py` angepasst werden.
* **NLP-Performance:** Auf reinen CPU-Systemen dauert die Batch-Inferenz bei >10.000 Zwischenrufen einige Minuten. `batch_size` ist jetzt in allen drei Engines (`SentimentEngine`, `ToneClassifier`, `AddresseeDetector`) konfigurierbar — für CPU-Betrieb empfiehlt sich `--nlp-batch-size 32`, für GPU `--nlp-batch-size 128 --nlp-fp16`.
* **Zero-Shot-Genauigkeit:** `ToneClassifier` liefert bei kurzen Zwischenrufen (<5 Wörter) weniger verlässliche Ergebnisse. Für sehr kurze Texte ist die regelbasierte Schnellklassifikation die primäre Quelle.
* [x] v2.2.0 — **Neue Analytics-Klassen (Tab H, I, M):** `TOPAnalyse` (Tagesordnungspunkt-Hostility-Ranking), `KategorieAnalyse` (Kategorie-Verteilung & Civility-Index), `RedeZeitAnalyse` (Speech-Time-Fairness). Alle nutzen bereits vorhandene Felder `Rede.tagesordnungspunkt`, `Zwischenruf.kategorie`, `Rede.wortanzahl`.
* [x] v2.3.0 — **ThemenKarriere-Erweiterungen:** `multi_wp_keyword_trend()`, `keyword_peak_by_wp()`, `keyword_aggression_correlation()`, `most_polarizing_keywords()`. Tab B um Multi-WP-Zeitreihe und Reizwort-Panel erweitert.
* [x] v2.4.0 — **Neue Analytics-Klassen (Tab L, K):** `SitzungsKlima` (per-session Temperatur-Index), `RednerProfil` (Speaker-Fingerprint aus `Rede.tone_scores` JSON). Tab L (Debattenklima), Tab K (Redner-Profil) im Dashboard.
* [x] v2.2.0–v2.4.0 — **Test-Suite weiter ausgebaut**; alle grün. Vollständige Dokumentation (README, Pflichtenheft, Project.md) aktualisiert.
* [x] v2.5.0 — **Native Navigation:** `st.navigation` / `st.Page` (Streamlit ≥ 1.36) ersetzt den alten Radio-Button-Hack mit manuellen `st.query_params`. 13 Seiten in vier Kategorien gruppiert: *Willkommen*, *Kern-Analysen*, *Sprache & Ton*, *Parlaments-Metriken*. `_navigate_to()` nutzt jetzt `st.switch_page()`. Startseite visuell aufgewertet: Live-Metriken, Experimente-Tabelle, Tech-Stack-Übersicht und Quick-Start-Guide. `Rede`-Import-Bug behoben. `pyproject.toml` auf v2.5.0 / `streamlit>=1.36` angehoben.


---

## Recommended Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Datenbeschaffung** | `Python` + `Requests` | Automatisierter, ressourcenschonender Abruf der XML/JSON-Protokolle (DIP). |
| **Datenhaltung** | `SQLite` + `SQLAlchemy` | Keine Server- oder Cloud-Kosten. SQLAlchemy (ORM) ermöglicht objektorientierten Zugriff auf die Tabellen, was den Analyse-Code lesbar macht. |
| **Parsing** | `Regex` + `BeautifulSoup` | Bundestags-Protokolle sind unstrukturierter Text im XML-Gewand. Regex ist zwingend nötig, um das "Gemurmel" in Klammern vom Redetext zu trennen. |
| **NLP (Sentiment)** | `HuggingFace` | Lokale KI-Inferenz (z.B. *DistilBERT multilingual*). Vermeidet API-Kosten (wie bei OpenAI) und garantiert den Offline-Betrieb. |
| **Data Science** | `Pandas` | Aggregation der Daten (Gruppieren nach Datum, Fraktion) im Arbeitsspeicher, bevor sie an das Frontend geschickt werden. |
| **Frontend (GUI)** | `Streamlit` | Erlaubt die Erstellung interaktiver Web-Dashboards direkt aus Python-Code, ohne HTML/JS/CSS schreiben zu müssen. Perfekt für schnelle Data-Science-GUIs. |
| **Visualisierung** | `Plotly` | Native Integration in Streamlit, interaktive Charts (Zoom, Hover-Infos), was für die Zeitreihenanalyse unerlässlich ist. |

---

## Complete Version Plan (v0.0.0 → v2.6.0)

| Version | Focus Area | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **v0.0.0** | Setup & Planung | Repo scaffolding, OOA/OOD Modelle definiert, Lasten/Pflichtenheft. | ✅ Done |
| **v0.1.0** | Proof of Concept | `requirements.txt`, SQLAlchemy Models (`models.py`, `database.py`), `scripts/db_init.py`. | ✅ Done |
| **v0.5.0** | Data Pipeline | `parser.py` (BundestagXMLParser), `scraper.py` (BundestagScraper), `run_scraper.py` (ProtocolImporter). | ✅ Done |
| **v0.8.0** | MVP GUI & Analytics | `nlp.py` (SentimentEngine), `analytics.py` (AggressionsIndex, ThemenKarriere, InteraktionsNetzwerk). | ✅ Done |
| **v1.0.0** | Release | `app.py` (Streamlit-Dashboard mit 3 Tabs), Unit-Tests, vollständige Dokumentation. | ✅ Done |
| **v1.1.0** | Scraper-Robustheit & Erweiterte NLP | Scraper-Bugfixes (URL, Regex-Filter); `ToneClassifier` (Aggression/Sarkasmus/Humor); `AddresseeDetector`; neue DB-Felder. | ✅ Done |
| **v1.2.0** | Dashboard-Erweiterung | Wahlperioden-Bugfix; Tabs D (Ton-Analyse), E (Adressaten-Analyse), F (Scraping-Monitor); `TonAnalyse`, `AdressatenAnalyse`, `ScrapingMonitor`. | ✅ Done |
| **v2.0.0** | Vollständige Wahlperioden-Analyse | Multi-Wahlperioden-Vergleich (Tab G); NetworkX/Gephi-Export; CI/CD via GitHub Actions; `pyproject.toml`. | ✅ Done |
| **v2.1.0** | Navigation & UX | Startseite (Landing Page); Sidebar-Radio-Navigation; URL-Query-Parameter-Routing (`?page=...`); `render_*()`-Funktionen; Dispatch-Dict; `DEFAULT_PAGE`-Konstante; `_navigate_to()`-Helper. | ✅ Done |
| **v2.2.0** | Untapped Data Analytics | `TOPAnalyse` (Tab H), `KategorieAnalyse` (Tab I), `RedeZeitAnalyse` (Tab M). Nutzt bereits gespeicherte Felder. | ✅ Done |
| **v2.3.0** | Longitudinal & Sentiment Upgrades | `ThemenKarriere.multi_wp_keyword_trend`, `keyword_aggression_correlation`, `most_polarizing_keywords`. Tab B erweitert. | ✅ Done |
| **v2.4.0** | Advanced Analytics | `SitzungsKlima` (Tab L: Temperatur-Index), `RednerProfil` (Tab K: Speaker DNA). | ✅ Done |
| **v2.5.0** | Native Navigation & Enhanced Home | `st.navigation` / `st.Page` (Streamlit ≥ 1.36) mit 4 gruppierten Sidebar-Kategorien; `st.switch_page()` via `_PAGE_REGISTRY`; visuell aufgewertete Startseite (Live-Metriken, Experimente-Tabelle, Tech-Stack). | ✅ Done |
| **v2.6.0** | Data Export & Speaker Comparison | CSV/Excel-Export für alle Analyse-Ergebnisse; neuer „Redner-Vergleich"-Tab (side-by-side Speaker Comparison); Volltext-Suche über alle Reden. | 🔜 Planned |


---

## Architecture Summary

```text
┌──────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (Backend / Offline)                           │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ Bundestag API   │───>│ BundestagScraper                │  │
│  │ (Open Data)     │    │ src/scraper.py                  │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
│                                  │                           │
│                                  v                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ SentimentEngine │<───│ BundestagXMLParser              │  │
│  │ ToneClassifier  │    │ src/parser.py                   │  │
│  │ AddresseeDetect │    └─────────────────────────────────┘  │
│  │ src/nlp.py      │                                         │
│           │                      │                           │
│           └──────────────────────v                           │
│                         [ SQLite Database ]                  │
│                         via SQLAlchemy ORM                   │
│                         (src/models.py + database.py)        │
└──────────────────────────────────────────────────────────────┘
                                   │
                                   v
┌───────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER (Frontend)                                │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ AggressionsIndex │ ThemenKarriere │ InteraktionsNetzwerk │ │
│  │ TonAnalyse │ AdressatenAnalyse │ ScrapingMonitor         │ │
│  │ WahlperiodenVergleich │ TOPAnalyse │ KategorieAnalyse    │ │
│  │ RedeZeitAnalyse │ SitzungsKlima │ RednerProfil           │ │
│  │ src/analytics.py                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                  │                            │
│                                  v                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Streamlit Dashboard  src/app.py  (v2.5.0)                │ │
│  │ Native st.navigation / st.Page (Streamlit ≥ 1.36)        │ │
│  │ 13 Seiten in 4 Gruppen:                                  │ │
│  │  Willkommen:                                             │ │
│  │    🏠 Startseite  •  📊 Scraping-Monitor                 │ │
│  │  Kern-Analysen:                                          │ │
│  │    🔥 Aggressions-Radar  •  📈 Themen-Trend              │ │
│  │    🕸️ Interaktions-Netzwerk  •  🏛️ Tagesordnungspunkte   │ │
│  │  Sprache & Ton:                                          │ │
│  │    🎭 Ton-Analyse  •  🎯 Adressaten-Analyse              │ │
│  │    👏 Reaktions-Analyse                                  │ │
│  │  Parlaments-Metriken:                                    │ │
│  │    ⏱️ Redezeit-Gerechtigkeit  •  🌡️ Debattenklima-Index  │ │
│  │    🎤 Redner-Profil  •  ⚖️ Wahlperioden-Vergleich        │ │
│  │ PLANNED (v2.6.0):                                        │ │
│  │    🔍 Volltext-Suche  •  👥 Redner-Vergleich             │ │
│  │    📥 CSV/Excel-Export für alle Analysen                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1 — SQLite + SQLAlchemy statt reiner CSV-Dateien

Problem: Das bloße Speichern in Pandas-DataFrames oder CSVs führt bei hunderttausenden Zeilen zu speicherintensiven Ladevorgängen und erschwert das Filtern nach Beziehungen (z.B. "Zeige alle Zwischenrufe zu Reden von Person X").

Decision: Einsatz einer relationalen Datenbank (SQLite). SQLAlchemy abstrahiert die SQL-Befehle, sodass wir im Code streng objektorientiert (`rede.zwischenrufe`) arbeiten können.

### 2 — Lokales HuggingFace statt OpenAI API

**Problem:** Die Sentiment-Analyse von zehntausenden Zwischenrufen über eine kommerzielle API (OpenAI/DeepL) generiert hohe laufende Kosten und erfordert einen API-Key vom Endnutzer.

**Decision:** Nutzung vortrainierter deutscher NLP-Modelle via HuggingFace. Die Inferenz dauert zwar auf reinen CPU-Systemen länger, erfolgt aber als lokaler Batch-Prozess beim Datenbankaufbau. Im Streamlit-Dashboard sind die fertigen Scores dann latenzfrei abrufbar.

### 3 — Entkopplung von Pipeline und Frontend (Asynchrone Architektur)

**Problem:** Wenn das Streamlit-Dashboard bei jedem Aufruf das NLP-Modell bemüht oder die Rohtexte durchsucht, wird die GUI extrem langsam (Timeouts).

**Decision:** Strikte MVC-Trennung. Der Scraper/Parser und das NLP-Modell füllen die Datenbank in einem komplett asynchronen, vorgeschalteten Prozess (`scripts/`). Das Streamlit-Frontend (`src/app.py`) liest ausschließlich fertige, aggregierte Datenstrukturen über `src/analytics.py`.

### 4 — Regelbasierte + neuronale NLP-Hybridstrategie

**Problem:** Viele Bundestags-Zwischenrufe folgen Standardmustern ("Beifall bei der SPD"), die kein Transformer-Modell benötigen.

**Decision:** Alle drei NLP-Engines (`SentimentEngine`, `ToneClassifier`, `AddresseeDetector`) verwenden dasselbe zweistufige Muster: Zuerst regelbasierte Schnellklassifikation (schnell, deterministisch). Nur unaufgelöste Texte werden an das HuggingFace-Modell übergeben (`_neural_*_batch`). Dies spart erheblich Rechenzeit. Alle drei Engines akzeptieren einen `batch_size`-Parameter für kontrollierten GPU/CPU-Durchsatz.

### 5 — Untapped-Data-First für v2.2.0

**Ansatz:** Bevor neue Datenquellen erschlossen oder ML-Modelle eingeführt werden, wurden in v2.2.0 zuerst alle bereits vorhandenen, gespeicherten Felder analysiert. `Rede.tagesordnungspunkt`, `Zwischenruf.kategorie` und `Rede.wortanzahl` waren seit v0.5.0 vorhanden, aber noch nicht ausgewertet. Durch `TOPAnalyse`, `KategorieAnalyse` und `RedeZeitAnalyse` wurden diese Felder in vollem Umfang genutzt.

### 6 — Composite-Index statt einzelner Metriken (SitzungsKlima)

**Problem:** Keine einzelne Metrik (Sentiment, Lautstärke, Tone-Label) erfasst das parlamentarische Klima vollständig.

**Decision:** `SitzungsKlima` kombiniert vier normierte Komponenten (heat_sent, Aggressions-Anteil, Unterbrechungsdichte, Unruhe-Anteil) per Min-Max-Normierung zu einem einzigen Composite-Score `temperatur_index ∈ [0,1]`. Das erlaubt intuitive Zeitreihen- und Anomalie-Darstellungen.
