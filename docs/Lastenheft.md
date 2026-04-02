# 🎯 Lastenheft: Demokratie-Mining

**Projekt:** Demokratie-Mining – Die digitale Röntgenaufnahme des Bundestags  
**Dokumentenart:** Lastenheft (Anforderungsspezifikation)  
**Datum:** 18.03.2026  

---

## 1. Einführung und Zielsetzung
Das Projekt "Demokratie-Mining" zielt darauf ab, die parlamentarische Kommunikation im Deutschen Bundestag quantitativ und qualitativ messbar zu machen. Das System soll eine Brücke zwischen riesigen, unstrukturierten Textmengen (Sitzungsprotokollen) und dem Endnutzer (Bürger, Journalisten, Forscher) schlagen. 
Hauptziel ist es, verborgene politische Dynamiken – wie die Aggressivität der Debattenkultur, thematische Konjunkturen und fraktionsübergreifende Netzwerke – durch Data Mining sichtbar zu machen und über eine intuitive grafische Benutzeroberfläche (GUI) bereitzustellen.

## 2. Ist-Zustand (Ausgangssituation)
Der Deutsche Bundestag veröffentlicht alle Sitzungsprotokolle als Open Data (DIP / Open Discourse) im XML- oder Text-Format. 
* Diese Dokumente umfassen pro Wahlperiode zehntausende Seiten. 
* Eine systematische Auswertung, wer wen wie oft unterbricht oder wie sich die Stimmung zu bestimmten Themen entwickelt, ist händisch nicht leistbar.
* Bisherige Analysen stützen sich oft auf "Gefühle" oder anekdotische Evidenz, statt auf harte Daten.

## 3. Soll-Zustand (Vision)
Es soll eine Softwareanwendung entwickelt werden, die den gesamten Prozess von der Datenbeschaffung bis zur Visualisierung automatisiert. Die Anwendung arbeitet rein lokal (offline-fähig nach dem Download) und verursacht **keine laufenden Kosten** für API-Aufrufe oder Server-Infrastruktur. Der Endnutzer bedient die Software über ein modernes, interaktives Web-Dashboard im Browser, ohne selbst programmieren zu müssen.

## 4. Funktionale Anforderungen (Muss-Kriterien)

Das System muss zwingend folgende Funktionen bereitstellen:

* **FA-10: Automatisierte Datenbeschaffung (Scraping)**
  * Das System muss in der Lage sein, Plenarprotokolle einer gesamten Wahlperiode automatisiert von offiziellen Schnittstellen herunterzuladen.
* **FA-20: Intelligentes Text-Parsing**
  * Rohe XML-Dokumente müssen in strukturierte Entitäten zerlegt werden: Sitzung, Redner, Rede und Zwischenruf.
  * Das System muss explizit erkennen, wer einen Zwischenruf getätigt hat (Person oder Fraktion).
* **FA-30: Natural Language Processing (NLP)**
  * Das System muss den Text von Zwischenrufen automatisch analysieren und einem Sentiment (Stimmungswert von negativ bis positiv) zuordnen.
* **FA-40: Interaktives Dashboard (GUI)**
  * Nutzer müssen Daten über Dropdowns und Slider filtern können (nach Datum, Fraktion, Politiker).
* **FA-50: Analyse-Modul "Aggressions-Index"**
  * Generierung einer Rangliste: Wer erhält die meisten negativen Zwischenrufe? Wer teilt am meisten aus?
* **FA-60: Analyse-Modul "Themen-Karriere"**
  * Freitextsuche für Schlagworte (z. B. "KI", "Klimaschutz") mit grafischer Ausgabe der Worthäufigkeit über die Zeit (relativ zur Gesamtwortzahl).
* **FA-70: Analyse-Modul "Netzwerk der Feindschaften"**
  * Berechnung einer Adjazenzmatrix (Wer unterbricht wen?), um Interaktionen zwischen Fraktionen als Netzwerk darzustellen.

## 5. Nicht-funktionale Anforderungen

* **NFA-10: Kostenfreiheit (Zero-Cost-Policy)**
  * Das Projekt darf ausschließlich kostenlose Open-Source-Software und -Modelle nutzen. Kostenpflichtige Cloud-APIs (wie OpenAI) sind strikt ausgeschlossen.
* **NFA-20: Performance und Usability**
  * Das Laden des Dashboards und das Filtern von Graphen darf nicht länger als 3 Sekunden dauern.
  * Rechenintensive NLP-Prozesse müssen im Hintergrund ablaufen und dürfen die Benutzeroberfläche nicht blockieren.
* **NFA-30: Datenhaltung**
  * Alle Daten müssen in einer lokalen, relationalen Datei-Datenbank (SQLite) gespeichert werden. Eine aufwendige Server-Installation (z.B. PostgreSQL) entfällt, um die Einstiegshürde für Endnutzer gering zu halten.
* **NFA-40: Wartbarkeit**
  * Der Code ist streng nach dem Prinzip der objektorientierten Programmierung (OOP) und dem Model-View-Controller (MVC) Muster zu strukturieren.

## 6. Lieferumfang
1. Ein Python-basiertes Backend (Scraper, Parser, NLP-Engine).
2. Ein initialisiertes SQLite-Datenbankschema.
3. Ein Frontend (Streamlit-Applikation).
4. Vollständige Dokumentation (README, Setup-Guide) zur lokalen Ausführung der Software.
