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
"""
from __future__ import annotations

from sqlalchemy import select as _sa_select

# ── Helper: get_or_create_redner ─────────────────────────────────────────────
def get_or_create_redner(session, redner_id, **kwargs):
    """
    Check if Redner exists by external bundestag_id, else create.
    Uses SQLAlchemy 2.0 syntax.
    """
    # Lookup: Search by bundestag_id (SQLAlchemy 2.0 style)
    instance = session.execute(
        _sa_select(Redner).where(Redner.bundestag_id == redner_id)
    ).scalar_one_or_none()
    
    if instance:
        return instance, False

    # Creation: Filter and Map
    valid_fields = {c.name for c in Redner.__table__.columns}
    # Exclude internal autoincrement PK to let DB handle it
    internal_pk = {c.name for c in Redner.__table__.primary_key.columns}
    
    # Filter kwargs to only valid fields, excluding the PK and the raw redner_id if it's there
    filtered_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in valid_fields and k not in internal_pk and k != "redner_id"
    }

    # Ensure the external ID is assigned to the correct column
    if "bundestag_id" in valid_fields:
        filtered_kwargs["bundestag_id"] = redner_id

    instance = Redner(**filtered_kwargs)
    session.add(instance)
    return instance, True
"""
Bundestag XML protocol parser for OpenParlament.

The Bundestag publishes plenary protocols as structured XML files (since the
19th Wahlperiode / 2017).  This module provides a single public entry-point:

        parser = BundestagXMLParser()
        result = parser.parse_file(Path("data/20001.xml"))

The returned ``ParseResult`` contains all ``Sitzung``, ``Redner``, ``Rede``
and ``Zwischenruf`` objects ready to be persisted via SQLAlchemy.

XML structure overview (20th Wahlperiode)
─────────────────────────────────────────
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
        <tagesordnungspunkt id="...">
            <rede id="ID205000100">
                <p klasse="redner">
                    <redner id="11004759">
                        <name>
                            <titel>Dr.</titel>
                            <vorname>Hans</vorname>
                            <nachname>Mustermann</nachname>
                            <fraktion>SPD</fraktion>
                        </name>
                    </redner>
                </p>
                <p klasse="J_1">Text …</p>
                <kommentar>(Beifall bei der SPD)</kommentar>
            </rede>
        </tagesordnungspunkt>
    </sitzungsverlauf>
</dbtplenarprotokoll>
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from src.models import Rede, Redner, Sitzung, Zwischenruf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for parse results (detached from DB session)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    """Aggregates all objects extracted from a single XML protocol file."""

    sitzung: Sitzung
    redner: List[Redner] = field(default_factory=list)
    reden: List[Rede] = field(default_factory=list)
    zwischenrufe: List[Zwischenruf] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Regular expressions
# ─────────────────────────────────────────────────────────────────────────────

# Matches the content of a <kommentar> tag, e.g. "(Beifall bei der SPD)"
_RE_KOMMENTAR = re.compile(r"\(([^)]+)\)", re.UNICODE)

# Extracts the faction from interjection text, e.g. "Beifall bei der SPD"
_RE_FRAKTION = re.compile(
    r"\b(?:bei|von|der|des|den)\s+(?:der\s+)?([A-ZÄÖÜ][A-ZÄÖÜa-zäöüß/\-]+(?:\s+[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß/\-]+){0,3})\b",
    re.UNICODE,
)

# Known German Bundestag faction short names (used for normalisation)
_KNOWN_FRAKTIONEN = {
    "SPD", "CDU", "CSU", "CDU/CSU", "Grünen", "GRÜNE", "BÜNDNIS 90/DIE GRÜNEN",
    "FDP", "AfD", "LINKE", "DIE LINKE", "BSW", "SSW",
}

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d")

_GERMAN_WEEKDAYS = [
    "Montag", "Dienstag", "Mittwoch",
    "Donnerstag", "Freitag", "Samstag", "Sonntag",
]


def _parse_date(raw: str) -> Optional[date]:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def weekday_german(d: date) -> str:
    """Return the German name of the weekday for *d* (e.g. ``'Dienstag'``)."""
    return _GERMAN_WEEKDAYS[d.weekday()]


# Keep the private alias for internal uses within this module.
_weekday_german = weekday_german


# ─────────────────────────────────────────────────────────────────────────────
# BundestagXMLParser
# ─────────────────────────────────────────────────────────────────────────────

class BundestagXMLParser:
    """Parse a Bundestag plenary protocol XML file into ORM objects.

    The parser is stateless and thread-safe; a single instance can be reused
    across many files.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def parse_file(self, path: Path, session=None) -> ParseResult:
        """Parse the XML at *path* and return a :class:`ParseResult`.

        :param path: Absolute or relative path to the XML file.
        :param session: Optional SQLAlchemy session for deduplication.
        :raises FileNotFoundError: if the file does not exist.
        :raises ValueError: if the file cannot be parsed as a Bundestag protocol.
        """
        if not path.exists():
            raise FileNotFoundError(f"Protocol file not found: {path}")

        with open(path, "rb") as fh:
            soup = BeautifulSoup(fh, "lxml-xml")

        return self._parse_soup(soup, session=session)

    def parse_bytes(self, data: bytes, session=None) -> ParseResult:
        """Parse an XML protocol from a raw bytes object."""
        soup = BeautifulSoup(data, "lxml-xml")
        return self._parse_soup(soup, session=session)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_soup(self, soup: BeautifulSoup, session=None) -> ParseResult:
        sitzung = self._extract_sitzung(soup)
        redner_map: Dict[str, Redner] = {}  # bundestag_id → Redner
        reden: List[Rede] = []
        alle_zwischenrufe: List[Zwischenruf] = []

        for top_tag in soup.find_all("tagesordnungspunkt"):
            top_label: str = top_tag.get("id", "")
            for rede_tag in top_tag.find_all("rede"):
                redner, new = self._extract_redner(rede_tag, redner_map, session=session)
                if new:
                    redner_map[redner.bundestag_id or redner.vollname] = redner

                rede, zwischenrufe = self._extract_rede(
                    rede_tag, sitzung, redner, top_label
                )
                reden.append(rede)
                alle_zwischenrufe.extend(zwischenrufe)

        # Update total word count for the Sitzung.
        sitzung.gesamtwortzahl = sum(r.wortanzahl for r in reden)

        return ParseResult(
            sitzung=sitzung,
            redner=list(redner_map.values()),
            reden=reden,
            zwischenrufe=alle_zwischenrufe,
        )

    # ── Sitzung ───────────────────────────────────────────────────────────────

    def _extract_sitzung(self, soup: BeautifulSoup) -> Sitzung:
        # ── Wahlperiode & Sitzungsnr ─────────────────────────────────────────
        # Scope to <plenarprotokoll-nummer> so the duplicate <sitzungsnr> that
        # appears inside <sitzungstitel> in real Bundestag XML doesn't shadow
        # the authoritative value.
        pnr_tag = soup.find("plenarprotokoll-nummer")
        wahlperiode = (self._text(pnr_tag, "wahlperiode") if pnr_tag else None) \
                      or self._text(soup, "wahlperiode") or "0"
        sitzungsnr  = (self._text(pnr_tag, "sitzungsnr")  if pnr_tag else None) \
                      or self._text(soup, "sitzungsnr")  or "0"

        # ── Date ─────────────────────────────────────────────────────────────
        # Real Bundestag XML (19th WP onwards) places the date in THREE places:
        #   1. Root element attribute:  <dbtplenarprotokoll sitzung-datum="24.06.2025">
        #   2. Tag attribute:           <datum date="24.06.2025">Dienstag, …</datum>
        #      inside <veranstaltungsdaten> inside <kopfdaten>
        #   3. Tag text content (test fixtures / older formats):
        #                               <datum>17.11.2021</datum>
        # The text content of the real <datum> tag is a German long-form date
        # ("Dienstag, den 24. Juni 2025") which does NOT match any of our date
        # format strings – hence the previous parser always produced datum=None.
        # Priority: root attribute → tag attribute → tag text content.
        datum_raw = ""

        # 1) Root element attribute
        root_tag = soup.find("dbtplenarprotokoll")
        if root_tag is not None:
            datum_raw = root_tag.get("sitzung-datum", "") or ""

        # 2) <datum date="…"> attribute inside <kopfdaten>
        if not datum_raw:
            kopfdaten_tag = soup.find("kopfdaten")
            if kopfdaten_tag is not None:
                datum_tag = kopfdaten_tag.find("datum")
                if datum_tag is not None:
                    datum_raw = datum_tag.get("date", "") or datum_tag.get_text(strip=True)

        # 3) Global fallback (older/simplified XMLs without the attribute)
        if not datum_raw:
            datum_tag = soup.find("datum")
            if datum_tag is not None:
                datum_raw = datum_tag.get("date", "") or datum_tag.get_text(strip=True)

        # ── Sitzungstitel ────────────────────────────────────────────────────
        # Real Bundestag XML: <sitzungstitel><sitzungsnr>12</sitzungsnr>. Sitzung</sitzungstitel>
        # Use get_text() so embedded child tags (like <sitzungsnr>) contribute
        # their text, giving "12. Sitzung" rather than just ". Sitzung".
        titel: Optional[str] = None
        sitzungstitel_tag = soup.find("sitzungstitel")
        if sitzungstitel_tag is not None:
            titel = sitzungstitel_tag.get_text(strip=True) or None

        # ── sitzungsnr normalisation ─────────────────────────────────────────
        # May contain non-numeric suffixes like '125 (neu)'; extract leading digits
        snr_match = re.search(r'\d+', sitzungsnr)
        if snr_match:
            snr_int = int(snr_match.group())
            if snr_match.group() != sitzungsnr.strip():
                logger.warning("Non-standard sitzungsnr %r; using %d", sitzungsnr, snr_int)
        else:
            logger.warning("Could not parse sitzungsnr %r; defaulting to 0", sitzungsnr)
            snr_int = 0

        parsed_datum = _parse_date(datum_raw) if datum_raw else None

        return Sitzung(
            wahlperiode=int(wahlperiode),
            sitzungsnr=snr_int,
            datum=parsed_datum,
            wochentag=_weekday_german(parsed_datum) if parsed_datum is not None else None,
            titel=titel,
        )

    # ── Redner ────────────────────────────────────────────────────────────────

    def _extract_redner(
        self, rede_tag: Tag, existing: Dict[str, Redner], session=None
    ) -> Tuple[Redner, bool]:
        """Return (Redner, is_new) for the speaker of this <rede> block."""
        redner_tag = rede_tag.find("redner")
        if redner_tag is None:
            # Fallback: anonymous speaker
            anon_key = "__anon__"
            if anon_key in existing:
                return existing[anon_key], False
            r = Redner(vorname="Unbekannt", nachname="Redner")
            r.bundestag_id = anon_key
            return r, True

        bt_id: str = redner_tag.get("id", "")
        if bt_id and bt_id in existing:
            return existing[bt_id], False

        name_tag = redner_tag.find("name")
        vorname = self._text(name_tag, "vorname") or "" if name_tag else ""
        nachname = self._text(name_tag, "nachname") or "Unbekannt" if name_tag else "Unbekannt"
        titel = self._text(name_tag, "titel") if name_tag else None
        fraktion = self._text(name_tag, "fraktion") if name_tag else None
        partei = self._text(name_tag, "partei") if name_tag else None

        # Use get_or_create_redner for DB/session lookup and creation
        if session is not None and bt_id:
            redner, is_new = get_or_create_redner(
                session,
                redner_id=bt_id,
                bundestag_id=bt_id,
                vorname=vorname,
                nachname=nachname,
                titel=titel or None,
                fraktion=fraktion,
                partei=partei,
            )
            return redner, is_new

        # Fallback: create new Redner if no session
        redner = Redner(
            bundestag_id=bt_id or None,
            vorname=vorname,
            nachname=nachname,
            titel=titel or None,
            fraktion=fraktion,
            partei=partei,
        )
        return redner, True

    # ── Rede & Zwischenrufe ────────────────────────────────────────────────────

    def _extract_rede(
        self,
        rede_tag: Tag,
        sitzung: Sitzung,
        redner: Redner,
        top_label: str,
    ) -> Tuple[Rede, List[Zwischenruf]]:
        rede_bt_id: str = rede_tag.get("id", "")
        text_parts: List[str] = []
        zwischenrufe: List[Zwischenruf] = []

        for child in rede_tag.children:
            if not isinstance(child, Tag):
                continue

            tag_name = child.name
            if tag_name == "p":
                klasse = child.get("klasse", "")
                if klasse == "redner":
                    continue  # skip speaker header
                raw_text = child.get_text(separator=" ", strip=True)
                if raw_text:
                    text_parts.append(raw_text)

            elif tag_name == "kommentar":
                raw_kommentar = child.get_text(strip=True)
                zwr = self._parse_kommentar(raw_kommentar)
                if zwr:
                    zwischenrufe.append(zwr)

        full_text = "\n".join(text_parts)
        word_count = len(full_text.split())

        rede = Rede(
            bundestag_rede_id=rede_bt_id or None,
            sitzung=sitzung,
            redner=redner,
            text=full_text,
            tagesordnungspunkt=top_label or None,
            wortanzahl=word_count,
        )
        for zwr in zwischenrufe:
            zwr.rede = rede
        return rede, zwischenrufe

    # ── Kommentar / Zwischenruf parsing ───────────────────────────────────────

    def _parse_kommentar(self, raw: str) -> Optional[Zwischenruf]:
        """Extract a :class:`Zwischenruf` from a raw kommentar string."""
        # Strip surrounding parens if present.
        match = _RE_KOMMENTAR.search(raw)
        text = match.group(1).strip() if match else raw.strip("() ")
        if not text:
            return None

        fraktion = self._detect_fraktion(text)
        kategorie = self._classify_kommentar(text)

        return Zwischenruf(text=text, fraktion=fraktion, kategorie=kategorie)

    @staticmethod
    def _detect_fraktion(text: str) -> Optional[str]:
        """Heuristically extract the faction from a kommentar text."""
        # Fast path: check if a known faction name appears verbatim.
        upper = text.upper()
        for frak in _KNOWN_FRAKTIONEN:
            if frak.upper() in upper:
                return frak
        # Fallback: regex extraction.
        m = _RE_FRAKTION.search(text)
        return m.group(1) if m else None

    @staticmethod
    def _classify_kommentar(text: str) -> str:
        """Assign a human-readable category to the kommentar text."""
        t = text.lower()
        if "beifall" in t:
            return "Beifall"
        if "lachen" in t or "heiterkeit" in t:
            return "Lachen"
        if "widerspruch" in t:
            return "Widerspruch"
        if "zuruf" in t or "ruft" in t:
            return "Zwischenruf"
        if any(w in t for w in ("unruhe", "lärm", "tumult")):
            return "Unruhe"
        if any(w in t for w in ("sehr richtig", "bravo", "gut so", "richtig")):
            return "Zustimmung"
        return "Sonstiges"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _text(tag: Optional[Tag], child_name: str) -> Optional[str]:
        """Safely extract the text content of a child tag."""
        if tag is None:
            return None
        child = tag.find(child_name)
        return child.get_text(strip=True) if child else None
