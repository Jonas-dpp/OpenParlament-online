# 🛠️ Pflichtenheft: Demokratie-Mining

## 1. Einleitung & Systemzweck
Das System "Demokratie-Mining" dient der systematischen, quantitativen Analyse der parlamentarischen Kommunikation im Deutschen Bundestag (Fokus auf eine oder mehrere volle Wahlperioden).
Die Software soll nicht nur abbilden, was gesagt wird, sondern durch Data Mining **implizites Verhalten** sichtbar machen. Das System liefert die technische Infrastruktur, um folgende übergeordnete Fragen datenbasiert zu beantworten:
* Wer triggert wen?
* Welche Themen werden "totgeschwiegen" oder dominieren?
* Wie aggressiv ist die Debattenkultur objektiv messbar?
* Welche Tagesordnungspunkte provozieren strukturell die meisten negativen Reaktionen?
* Wird Redezeit fair proportional zur Fraktionsgröße verteilt?
* Wie entwickeln sich parlamentarische Beziehungen über eine gesamte Wahlperiode?

## 2. Systemarchitektur & Tech-Stack
Das System nutzt ausschließlich kostenlose Open-Source-Tools und setzt sich wie folgt zusammen:

| Schicht / Phase | Tool / Technologie | Technischer Zweck |
| :--- | :--- | :--- |
| **Datenbeschaffung** | `Python` + `Requests` | Automatisierter Abruf der XML/JSON-Protokolle (Bundestag-DIP / Open Discourse). |
| **Datenhaltung** | `SQLite` + `SQLAlchemy` | Lokale, leichtgewichtige relationale Datenbank für Rohdaten und geparste OOP-Objekte. |
| **Parsing-Engine** | `Regex` + `BeautifulSoup` | Trennung von Sprecher, Fraktion, Redetext und in Klammern gesetzten Zwischenrufen. |
| **Analyse & Logik** | `Pandas` | Statistische Auswertung, Zeitreihenanalyse und In-Memory-Korrelationen. |
| **NLP (KI)** | `HuggingFace` (German BERT / mDeBERTa) | Lokale, automatisierte Stimmungsanalyse (Sentiment) und Ton-Klassifikation (Aggression/Sarkasmus/Humor/Neutral). |
| **Frontend (GUI)** | `Streamlit` + `Plotly` | Interaktives Dashboard zur Visualisierung der Daten und Trends im Browser. |
| **Graph-Export** | `NetworkX` / `Gephi` | Erstellung von Kanten und Knoten für die Netzwerkanalyse (Matrix-Export). |

## 3. Datenmodell (OOD & Datenbank-Schema)
Das System extrahiert und speichert folgende Kernentitäten:
* **`Sitzung`**: Speichert Metadaten (`sitzungs_id`, `datum`, `wahlperiode`, `gesamtwortzahl`).
* **`Redner`**: Speichert Personendaten (`redner_id`, `vorname`, `nachname`, `fraktion`, `partei`).
* **`Rede`**: Speichert den eigentlichen Beitrag (`rede_id`, `text`, `wortanzahl`, `tagesordnungspunkt`, `tone_scores` JSON, Fremdschlüssel zu `Sitzung` und `Redner`).
* **`Zwischenruf`**: Speichert verbale oder non-verbale Einwürfe während einer Rede (`ruf_id`, `text`, `sentiment_score`, `ton_label`, `adressaten`, `kategorie`, `fraktion`, Fremdschlüssel zu `Rede`).

### Wichtige Felder für die neuen Analysen (v2.2.0+)
| Feld | Modell | Gefüllt durch | Genutzt in |
| :--- | :--- | :--- | :--- |
| `tagesordnungspunkt` | `Rede` | Parser (aus XML) | TOPAnalyse (Tab H) |
| `wortanzahl` | `Rede` | Parser (aus XML) | RedeZeitAnalyse (Tab M), RednerProfil (Tab K) |
| `tone_scores` | `Rede` | ToneClassifier (NLP-Pipeline) | RednerProfil (Tab K) |
| `kategorie` | `Zwischenruf` | Parser (Muster: "Beifall", "Widerspruch", "Lachen") | KategorieAnalyse (Tab I), TOPAnalyse (Tab H), SitzungsKlima (Tab L) |
| `ton_label` | `Zwischenruf` | ToneClassifier (NLP-Pipeline) | SitzungsKlima (Tab L) |
| `sentiment_score` | `Zwischenruf` | SentimentEngine (NLP-Pipeline) | Alle Tabs |

## 4. Experimente & Datenanalysen

### A. Der "Aggressions-Index" (Zwischenruf-Mining) – Tab A
**Ziel:** Messung der Diskussionshärte und Identifikation der Störer und Zielscheiben.
* **Analytics-Klasse:** `AggressionsIndex`
* **Verwendete Daten:** `Zwischenruf.sentiment_score`, `Rede`, `Redner.fraktion`.
* **Ausgabe:** Rangliste der häufigsten Opfer und Störer nach Aggressions-Score.

### B. Die "Themen-Karriere" (Trend-Mining) – Tab B
**Ziel:** Messung von politischer Aufmerksamkeit und Identifikation von Themen-Zyklen.
* **Analytics-Klasse:** `ThemenKarriere`
* **Methoden:** `keyword_trend()`, `multi_wp_keyword_trend()`, `keyword_peak_by_wp()`, `keyword_aggression_correlation()`, `most_polarizing_keywords()`.
* **Ausgabe:** Zeitreihen-Analyse (Line-Charts), Multi-Wahlperioden-Karriere, Reizwort-Index.

### C. Das "Netzwerk der Feindschaften" (Graph-Mining) – Tab C
**Ziel:** Visualisierung der inter- und intrafraktionellen Dynamiken.
* **Analytics-Klasse:** `InteraktionsNetzwerk`
* **Methoden:** `adjacency_matrix()`, `edge_list()`, `to_networkx_graph()`, `to_graphml_bytes()`, `to_gexf_bytes()`, `adjacency_matrix_by_window()`.
* **Ausgabe:** Adjazenzmatrix-Heatmap, GraphML/GEXF-Export, interaktives Plotly-Netzwerkdiagramm mit Timeline-Slider (Netzwerk-Evolution, v2.4.0).

### D. Ton-Analyse – Tab D
**Analytics-Klasse:** `TonAnalyse`. Verteilung der Ton-Labels (Aggression/Sarkasmus/Humor/Neutral) nach Fraktion und Zeitverlauf.

### E. Adressaten-Erkennung – Tab E
**Analytics-Klasse:** `AdressatenAnalyse`. Top-Adressaten und Fraktions-Adressierungs-Matrix.

### F. Scraping-Monitor – Tab F
**Analytics-Klasse:** `ScrapingMonitor`. DB-Füllstand und NLP-Abdeckungsgrad.

### G. Wahlperioden-Vergleich – Tab G
**Analytics-Klasse:** `WahlperiodenVergleich`. Vergleich von Aggression, Ton und Aktivität über Wahlperioden hinweg.

### H. Tagesordnungspunkt-Analyse – Tab H (v2.2.0)
**Ziel:** Welche Agenda-Items provozieren strukturell die meisten negativen Reaktionen?
* **Analytics-Klasse:** `TOPAnalyse`
* **Methoden:** `aggression_by_top()`, `kategorie_by_top()`.
* **Verwendete Daten:** `Rede.tagesordnungspunkt`, `Zwischenruf.sentiment_score`, `Zwischenruf.kategorie`.
* **Ausgabe:** Horizontales Balkendiagramm der Top-N-Reizthemen, gestapeltes Balkendiagramm der Kategorie-Verteilung pro TOP.

### I. Reaktions-Analyse (Kategorie-Index) – Tab I (v2.2.0)
**Ziel:** Wer produziert Beifall, wer produziert Widerspruch? Civility-Index pro Fraktions-Paar.
* **Analytics-Klasse:** `KategorieAnalyse`
* **Methoden:** `kategorie_by_fraktion()`, `beifall_widerspruch_ratio()`, `lachen_by_redner()`.
* **Verwendete Daten:** `Zwischenruf.kategorie`.
* **Ausgabe:** Gestapeltes Balkendiagramm, Civility-Heatmap, Ranking der parlamentarischen Komiker.

### M. Redezeit-Gerechtigkeit – Tab M (v2.2.0)
**Ziel:** Wird Redezeit proportional zur Fraktionsgröße verteilt?
* **Analytics-Klasse:** `RedeZeitAnalyse`
* **Methoden:** `wortanzahl_by_fraktion()`, `fairness_index()`, `top_redselige_redner()`, `wortanzahl_vs_zwischenrufe()`.
* **Verwendete Daten:** `Rede.wortanzahl`, `FACTION_SIZES_BY_WAHLPERIODE`.
* **Ausgabe:** Fairness-Index-Balkendiagramm (1.0 = proportional), Scatter-Plot Wortanzahl vs. Negative Zwischenrufe.

### L. Debattenklima-Index – Tab L (v2.4.0)
**Ziel:** Wie heiß war das Parlament pro Sitzung?
* **Analytics-Klasse:** `SitzungsKlima`
* **Methoden:** `klima_per_sitzung()`, `hottest_sessions()`.
* **Composite-Index:** `temperatur_index ∈ [0,1]` aus Aggressions-Sentiment, Aggressions-Label-Anteil, Unterbrechungsdichte und Unruhe-Anteil (Min-Max-normiert).
* **Ausgabe:** Zeitreihen-Chart mit rollierendem 10-Sitzungs-Durchschnitt, Tabelle der 15 heißesten Sitzungen.

### K. Redner-Profil (Speaker DNA) – Tab K (v2.4.0)
**Ziel:** Rhetorischer Fingerabdruck jedes Abgeordneten.
* **Analytics-Klasse:** `RednerProfil`
* **Methoden:** `speaker_profile()`, `top_speakers_by_tone()`, `faction_profile()`.
* **Verwendete Daten:** `Rede.tone_scores` (JSON: Aggression/Sarkasmus/Humor/Neutral-Wahrscheinlichkeiten).
* **Ausgabe:** Radar-Chart für Einzel-Profil, Balkendiagramm nach Fraktion, Top-15-Ranking je Ton-Label.

## 5. GUI & Frontend-Anforderungen
Die Streamlit-App muss alle Analysen für den Endnutzer ohne Programmierkenntnisse bedienbar machen.

Die Navigation erfolgt über die **Sidebar** mit gruppierten Kategorien (native `st.navigation`-API, Streamlit ≥ 1.36). Seiten können programmatisch über `st.switch_page()` angesteuert werden.
Die Streamlit-App bietet zur Zeit dreizehn Seiten (13), aufgeteilt in vier Gruppen:
| Gruppe | Seiten |
| :--- | :--- |
| Willkommen | 🏠 Startseite, 📊 Scraping-Monitor |
| Kern-Analysen | 🔥 Aggressions-Radar, 📈 Themen-Trend, 🕸️ Interaktions-Netzwerk, 🏛️ Tagesordnungspunkte |
| Sprache & Ton | 🎭 Ton-Analyse, 🎯 Adressaten-Analyse, 👏 Reaktions-Analyse |
| Parlaments-Metriken | ⏱️ Redezeit-Gerechtigkeit, 🌡️ Debattenklima-Index, 🎤 Redner-Profil, ⚖️ Wahlperioden-Vergleich |

Die Schlüssel-Visualisierungen je Seite:

| Seite | Analytics-Klasse | Schlüssel-Visualisierungen |
| :--- | :--- | :--- |
| 🔥 Aggressions-Radar | `AggressionsIndex` | Balken-, Streudiagramme |
| 📈 Themen-Trend | `ThemenKarriere` | Zeitreihen, Multi-WP-Chart, Reizwort-Panel |
| 🕸️ Interaktions-Netzwerk | `InteraktionsNetzwerk` | Heatmap, GraphML/GEXF-Export, interaktives Netzwerk + Timeline-Slider |
| 🎭 Ton-Analyse | `TonAnalyse` | Gestapeltes Balkendiagramm, Zeitreihe |
| 🎯 Adressaten-Analyse | `AdressatenAnalyse` | Heatmap, Ranking |
| 📊 Scraping-Monitor | `ScrapingMonitor` | Füllstand-Tabellen |
| ⚖️ Wahlperioden-Vergleich | `WahlperiodenVergleich` | Vergleichsbalken |
| 🏛️ Tagesordnungspunkte | `TOPAnalyse` | Horizontales Balkendiagramm, gestapeltes Balkendiagramm |
| 👏 Reaktions-Analyse | `KategorieAnalyse` | Gestapeltes Balkendiagramm, Civility-Heatmap |
| ⏱️ Redezeit-Gerechtigkeit | `RedeZeitAnalyse` | Fairness-Index, Scatter-Plot |
| 🌡️ Debattenklima-Index | `SitzungsKlima` | Zeitreihen-Chart, Anomalie-Tabelle |
| 🎤 Redner-Profil | `RednerProfil` | Radar-Chart, Balkendiagramm |
| 🏠 Startseite | — | Live-Metriken, Experimente-Tabelle, Quick-Start-Guide |

*Geplante Erweiterungen (v2.6.0):* **Redner-Vergleich** (side-by-side Speaker Comparison), **Volltext-Suche** über alle Reden, **CSV/Excel-Export** für alle Analyse-Ergebnisse.

