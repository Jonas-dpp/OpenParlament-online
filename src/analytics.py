"""
Analytics module for OpenParlament.

Provides analysis classes corresponding to the core experiments:

A. AggressionsIndex    – Who receives / produces the most negative interjections?
B. ThemenKarriere      – Keyword frequency over time (normalised per session).
C. InteraktionsNetzwerk – Adjacency matrix of inter-faction interruptions + NetworkX/Gephi export.
D. TonAnalyse          – Tone-label distribution (Aggression/Sarkasmus/Humor/Neutral).
E. AdressatenAnalyse   – Who gets targeted by which factions?
F. ScrapingMonitor     – Database fill-state and NLP-coverage overview.
G. WahlperiodenVergleich – Cross-legislature comparison of key metrics.
H. TOPAnalyse          – Agenda-item (Tagesordnungspunkt) hostility heat-map.  [v2.2.0]
I. KategorieAnalyse    – Zwischenruf category distribution (Applaus vs. Disruption). [v2.2.0]
M. RedeZeitAnalyse     – Speech-time fairness relative to faction size. [v2.2.0]
L. SitzungsKlima       – Per-session parliamentary temperature index. [v2.4.0]
K. RednerProfil        – Speaker rhetorical fingerprint from tone_scores JSON. [v2.4.0]

All classes operate on a SQLAlchemy session and return Pandas DataFrames
for direct use in the Streamlit frontend.
"""

from __future__ import annotations

import io
import json
import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from src.models import Rede, Redner, Sitzung, Zwischenruf
from src.nlp import _FRAKTION_CLEANUP_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalise_faction(name: Optional[str]) -> Optional[str]:
    """Map a raw faction string to its canonical form.

    Uses the same normalization dictionary as the NLP pipeline so that
    variants like "DIE LINKE" / "LINKE" / "LINKEN" all collapse to
    "Die Linke" and "GRÜNEN" / "BÜNDNIS 90/DIE GRÜNEN" both become
    "Bündnis 90/Die Grünen".
    """
    if not name:
        return name
    return _FRAKTION_CLEANUP_MAP.get(name.upper(), name)


def _parse_adressaten(raw: str) -> List[str]:
    """Parse an *adressaten* value stored in the database.

    The NLP pipeline serialises addressee lists via ``json.dumps``, so values
    look like ``'["SPD", "CDU/CSU"]'``.  Older / test data may be stored as a
    plain string (e.g. ``"SPD"``).  Both formats are handled gracefully.
    """
    raw = raw.strip() if raw else ""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        # Scalar JSON value (e.g. a bare string like '"SPD"')
        return [str(parsed).strip()] if str(parsed).strip() else []
    except (json.JSONDecodeError, ValueError):
        # Fallback: plain comma-separated string
        return [part.strip() for part in raw.split(",") if part.strip()]


# Faction sizes (number of seats) per Wahlperiode.
# Values reflect the initial composition after each federal election.
# Used to normalise raw interjection counts to a per-capita aggression metric.
# "Die Linke" covers the PDS era too because _canonicalise_faction maps PDS → Die Linke.
FACTION_SIZES_BY_WAHLPERIODE: Dict[int, Dict[str, int]] = {
    13: {  # 1994–1998
        "CDU/CSU": 294,
        "SPD": 252,
        "Bündnis 90/Die Grünen": 49,
        "FDP": 47,
        "Die Linke": 30,   # PDS
    },
    14: {  # 1998–2002
        "SPD": 298,
        "CDU/CSU": 245,
        "Bündnis 90/Die Grünen": 47,
        "FDP": 43,
        "Die Linke": 36,   # PDS
    },
    15: {  # 2002–2005
        "SPD": 251,
        "CDU/CSU": 248,
        "Bündnis 90/Die Grünen": 55,
        "FDP": 47,
    },
    16: {  # 2005–2009
        "CDU/CSU": 226,
        "SPD": 222,
        "FDP": 61,
        "Die Linke": 54,
        "Bündnis 90/Die Grünen": 51,
    },
    17: {  # 2009–2013
        "CDU/CSU": 237,
        "SPD": 146,
        "FDP": 93,
        "Die Linke": 76,
        "Bündnis 90/Die Grünen": 68,
    },
    18: {  # 2013–2017
        "CDU/CSU": 311,
        "SPD": 193,
        "Die Linke": 64,
        "Bündnis 90/Die Grünen": 63,
    },
    19: {  # 2017–2021
        "CDU/CSU": 246,
        "SPD": 153,
        "AfD": 94,
        "FDP": 80,
        "Die Linke": 69,
        "Bündnis 90/Die Grünen": 67,
    },
    20: {  # 2021–2025
        "SPD": 206,
        "CDU/CSU": 197,
        "Bündnis 90/Die Grünen": 118,
        "FDP": 92,
        "AfD": 83,
        "Die Linke": 39,
        "BSW": 10,
        "SSW": 1,
    },
    21: {  # 2025–
        "CDU/CSU": 208,
        "AfD": 152,
        "SPD": 120,
        "Bündnis 90/Die Grünen": 85,
        "Die Linke": 64,
        "SSW": 1,
    },
}

# Backward-compatible alias pointing at the most recent known Wahlperiode.
FACTION_SIZES: Dict[str, int] = FACTION_SIZES_BY_WAHLPERIODE[max(FACTION_SIZES_BY_WAHLPERIODE)]


def _get_faction_sizes(wahlperiode: Optional[int]) -> Dict[str, int]:
    """Return the faction-size lookup table for *wahlperiode*.

    Falls back to the most recent known Wahlperiode when *wahlperiode* is
    ``None`` or not yet present in :data:`FACTION_SIZES_BY_WAHLPERIODE`.
    """
    if wahlperiode is not None and wahlperiode in FACTION_SIZES_BY_WAHLPERIODE:
        return FACTION_SIZES_BY_WAHLPERIODE[wahlperiode]
    return FACTION_SIZES_BY_WAHLPERIODE[max(FACTION_SIZES_BY_WAHLPERIODE)]


# ─────────────────────────────────────────────────────────────────────────────
# A. Aggressions-Index
# ─────────────────────────────────────────────────────────────────────────────

class AggressionsIndex:
    """Compute aggression-related statistics from interjection data.

    All methods return Pandas DataFrames for easy downstream use.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def top_targets(
        self,
        n: int = 20,
        fraktion_filter: Optional[str] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the *n* speakers who received the most negative interjections.

        Columns: nachname, vorname, fraktion, neg_zwischenrufe, wortanzahl,
                 neg_pro_100_worte.

        :param n: Number of top results.
        :param fraktion_filter: Restrict to speakers of this faction.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        :param wahlperiode: Restrict to a specific Wahlperiode.
        """
        stmt = (
            select(
                Redner.nachname,
                Redner.vorname,
                Redner.fraktion,
                func.count(Zwischenruf.ruf_id).label("neg_zwischenrufe"),
                func.sum(Rede.wortanzahl).label("wortanzahl"),
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.sentiment_score < -0.3)
        )
        stmt = self._apply_filters(stmt, fraktion_filter, datum_von, datum_bis, wahlperiode)
        stmt = stmt.group_by(Redner.redner_id).order_by(
            func.count(Zwischenruf.ruf_id).desc()
        ).limit(n)

        rows = self._db.execute(stmt).fetchall()
        df = pd.DataFrame(rows, columns=["nachname", "vorname", "fraktion",
                                         "neg_zwischenrufe", "wortanzahl"])
        df["neg_pro_100_worte"] = (
            df["neg_zwischenrufe"] / df["wortanzahl"].replace(0, 1) * 100
        )
        # Ensure numeric dtype before rounding
        df["neg_pro_100_worte"] = pd.to_numeric(df["neg_pro_100_worte"], errors="coerce").round(2)
        return df

    def top_interruptors(
        self,
        n: int = 20,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the *n* factions that produce the most negative interjections.

        Columns: fraktion, neg_zwischenrufe, alle_zwischenrufe, anteil_negativ.
        """
        stmt = (
            select(
                Zwischenruf.fraktion,
                func.count(Zwischenruf.ruf_id).label("alle"),
                func.sum(
                    case((Zwischenruf.sentiment_score < -0.3, 1), else_=0)
                ).label("neg"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.fraktion.isnot(None))
        )
        stmt = self._apply_filters(stmt, None, datum_von, datum_bis, wahlperiode)
        stmt = stmt.group_by(Zwischenruf.fraktion).order_by(
            func.count(Zwischenruf.ruf_id).desc()
        ).limit(n)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["fraktion", "alle_zwischenrufe", "neg_zwischenrufe", "anteil_negativ"])
        
        df = pd.DataFrame(rows, columns=["fraktion", "alle_zwischenrufe", "neg_zwischenrufe"])
        
        # Consistent type casting to prevent math errors
        df["alle_zwischenrufe"] = pd.to_numeric(df["alle_zwischenrufe"], errors="coerce").fillna(0).astype(int)
        df["neg_zwischenrufe"] = pd.to_numeric(df["neg_zwischenrufe"], errors="coerce").fillna(0).astype(int)
        
        # Calculate percentage with protection against DivisionByZero
        df["anteil_negativ"] = (
            df["neg_zwischenrufe"] / df["alle_zwischenrufe"].replace(0, 1) * 100
        ).round(1)
        
        return df

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_filters(self, stmt, fraktion_filter, datum_von, datum_bis, wahlperiode=None):
        if fraktion_filter:
            stmt = stmt.where(Redner.fraktion == fraktion_filter)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        return stmt


# ─────────────────────────────────────────────────────────────────────────────
# B. Themen-Karriere
# ─────────────────────────────────────────────────────────────────────────────

class ThemenKarriere:
    """Track how often a keyword appears in speeches over time.

    Normalises raw counts by the total word count of each session to avoid
    bias from long vs. short sitting days.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def keyword_trend(
        self,
        keyword: str,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return a time-series DataFrame with normalised keyword frequency.

        Columns: datum, sitzungsnr, rohanzahl, gesamtwortzahl, normiert.

        :param keyword: The search term (case-insensitive, regex-safe).
        :param wahlperiode: Optional filter to a single Wahlperiode.
        """
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        stmt = (
            select(
                Sitzung.datum,
                Sitzung.sitzungsnr,
                Sitzung.gesamtwortzahl,
                Rede.text,
            )
            .join(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        stmt = stmt.order_by(Sitzung.datum)

        rows = self._db.execute(stmt).fetchall()

        # Aggregate per session.
        agg: Dict[Tuple, List] = {}
        for datum, sitzungsnr, gesamtwortzahl, text in rows:
            key = (datum, sitzungsnr, gesamtwortzahl)
            count = len(pattern.findall(text or ""))
            if key not in agg:
                agg[key] = 0
            agg[key] += count

        records = [
            {
                "datum": k[0],
                "sitzungsnr": k[1],
                "gesamtwortzahl": k[2],
                "rohanzahl": v,
                "normiert": round(v / max(k[2], 1) * 1000, 4),  # per 1 000 words
            }
            for k, v in agg.items()
        ]
        return pd.DataFrame(records)

    def multi_wp_keyword_trend(self, keyword: str) -> pd.DataFrame:
        """Return normalised keyword frequency across *all* Wahlperioden.

        Columns: wahlperiode, datum, sitzungsnr, gesamtwortzahl, rohanzahl, normiert.

        Each row represents one session.  Callers can colour-code by
        ``wahlperiode`` to produce multi-trace longitudinal charts.

        :param keyword: The search term (case-insensitive, regex-safe).
        """
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        # Pre-filter in SQL so that only speeches mentioning the keyword are
        # loaded into Python.  This avoids transferring the full text corpus
        # across the SQLAlchemy boundary when the keyword is rare.
        stmt = (
            select(
                Sitzung.wahlperiode,
                Sitzung.datum,
                Sitzung.sitzungsnr,
                Sitzung.gesamtwortzahl,
                Rede.text,
            )
            .join(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
            .where(Rede.text.ilike(f"%{keyword}%"))
            .order_by(Sitzung.wahlperiode, Sitzung.datum)
        )
        rows = self._db.execute(stmt).fetchall()

        # Aggregate per (wahlperiode, sitzungsnr, datum, gesamtwortzahl).
        agg: Dict[Tuple, int] = {}
        for wp, datum, sitzungsnr, gesamtwortzahl, text in rows:
            key = (wp, datum, sitzungsnr, gesamtwortzahl)
            count = len(pattern.findall(text or ""))
            agg[key] = agg.get(key, 0) + count

        records = [
            {
                "wahlperiode": k[0],
                "datum": k[1],
                "sitzungsnr": k[2],
                "gesamtwortzahl": k[3],
                "rohanzahl": v,
                "normiert": round(v / max(k[3], 1) * 1000, 4),
            }
            for k, v in agg.items()
        ]
        return pd.DataFrame(records)

    def keyword_peak_by_wp(self, keyword: str) -> pd.DataFrame:
        """Return the peak normalised frequency and its date per Wahlperiode.

        Columns: wahlperiode, peak_normiert, peak_datum.

        Useful for pinpointing *when* a topic had its parliamentary moment in
        each legislative term.

        :param keyword: The search term (case-insensitive, regex-safe).
        """
        df = self.multi_wp_keyword_trend(keyword)
        if df.empty:
            return pd.DataFrame(columns=["wahlperiode", "peak_normiert", "peak_datum"])

        idx = df.groupby("wahlperiode")["normiert"].idxmax()
        peak = df.loc[idx, ["wahlperiode", "normiert", "datum"]].copy()
        peak.columns = ["wahlperiode", "peak_normiert", "peak_datum"]
        return peak.reset_index(drop=True)

    def keyword_aggression_correlation(
        self,
        keyword: str,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compare the interjection sentiment during keyword-mentioning speeches
        against the baseline (speeches that do *not* mention the keyword).

        Columns: group ("Keyword", "Baseline"), avg_sentiment, std_sentiment,
                 anzahl_zwischenrufe, avg_aggression_score.

        ``avg_aggression_score`` = ``−avg_sentiment``.

        :param keyword: The search term (case-insensitive, regex-safe).
        :param wahlperiode: Optional Wahlperiode filter.
        """
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        stmt = (
            select(
                Rede.rede_id,
                Rede.text,
                Zwischenruf.sentiment_score,
            )
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.sentiment_score.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["group", "avg_sentiment", "std_sentiment",
                         "anzahl_zwischenrufe", "avg_aggression_score"]
            )

        df = pd.DataFrame(rows, columns=["rede_id", "text", "sentiment_score"])
        df["has_keyword"] = df["text"].apply(
            lambda t: bool(pattern.search(t or ""))
        )

        records = []
        for label, flag in (("Keyword", True), ("Baseline", False)):
            subset = df[df["has_keyword"] == flag]["sentiment_score"]
            if subset.empty:
                records.append({
                    "group": label,
                    "avg_sentiment": None,
                    "std_sentiment": None,
                    "anzahl_zwischenrufe": 0,
                    "avg_aggression_score": None,
                })
            else:
                avg = subset.mean()
                records.append({
                    "group": label,
                    "avg_sentiment": round(avg, 4),
                    "std_sentiment": round(subset.std(), 4),
                    "anzahl_zwischenrufe": int(len(subset)),
                    "avg_aggression_score": round(-avg, 4),
                })
        return pd.DataFrame(records)

    def most_polarizing_keywords(
        self,
        keyword_list: List[str],
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Rank keywords by how much they raise the aggression level above baseline.

        For each keyword the *delta* = keyword avg_aggression − baseline avg_aggression
        is computed.  A higher delta means speeches about that keyword attract
        more hostile reactions than average.

        Columns: keyword, delta_aggression, keyword_avg_aggression,
                 baseline_avg_aggression, anzahl_zwischenrufe.

        :param keyword_list: List of search terms to compare.
        :param wahlperiode: Optional Wahlperiode filter.
        """
        records = []
        for kw in keyword_list:
            df = self.keyword_aggression_correlation(kw, wahlperiode=wahlperiode)
            if df.empty:
                continue
            kw_row = df[df["group"] == "Keyword"]
            bl_row = df[df["group"] == "Baseline"]
            if kw_row.empty or bl_row.empty:
                continue
            kw_agg = kw_row.iloc[0]["avg_aggression_score"]
            bl_agg = bl_row.iloc[0]["avg_aggression_score"]
            if kw_agg is None or bl_agg is None:
                continue
            records.append({
                "keyword": kw,
                "delta_aggression": round(kw_agg - bl_agg, 4),
                "keyword_avg_aggression": kw_agg,
                "baseline_avg_aggression": bl_agg,
                "anzahl_zwischenrufe": int(kw_row.iloc[0]["anzahl_zwischenrufe"]),
            })
        if not records:
            return pd.DataFrame(
                columns=["keyword", "delta_aggression", "keyword_avg_aggression",
                         "baseline_avg_aggression", "anzahl_zwischenrufe"]
            )
        return (
            pd.DataFrame(records)
            .sort_values("delta_aggression", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# C. Interaktions-Netzwerk
# ─────────────────────────────────────────────────────────────────────────────

class InteraktionsNetzwerk:
    """Build a faction-level interruption adjacency matrix.

    Rows    = speaker faction (who is being interrupted).
    Columns = interruptor faction (who is interrupting).

    Faction names are normalised to canonical forms so that variants
    ("DIE LINKE", "LINKE") collapse to a single entry. Self-interactions
    (same faction applauding their own speaker) are excluded by default
    because they inflate the diagonal and are not meaningful as an
    aggression signal.

    When ``score_weighted=True`` the matrix shows an *aggression score*
    defined as ``−avg_sentiment``.  Negative sentiment (Widerspruch,
    Zwischenrufe) yields a high positive aggression score (red), while
    positive sentiment (Beifall) yields a low / near-zero score.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def adjacency_matrix(
        self,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        score_weighted: bool = False,
        wahlperiode: Optional[int] = None,
        exclude_self: bool = True,
        per_capita: bool = False,
    ) -> pd.DataFrame:
        """Return the faction × faction adjacency matrix.

        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        :param score_weighted: If True, values = aggression score (−avg_sentiment).
                               If False, values = interjection count.
        :param wahlperiode: Restrict to a specific Wahlperiode.
        :param exclude_self: If True (default), remove same-faction entries so
                             that the matrix diagonal stays at zero.
        :param per_capita: If True, divide each cell by the interruptor faction's
                           known member count so that smaller factions are not
                           artificially disadvantaged by their size.
        :return: Pandas DataFrame (factions as both index and columns), sorted
                 by total interaction volume descending on both axes so that
                 the most active factions appear top-left and low-volume
                 outliers are pushed to the right/bottom.
        """
        # Always fetch count AND avg_sentiment; both are needed for proper
        # weighted re-aggregation after faction-name normalisation.
        stmt = (
            select(
                Redner.fraktion.label("sprecher_fraktion"),
                Zwischenruf.fraktion.label("interruptor_fraktion"),
                func.count(Zwischenruf.ruf_id).label("anzahl"),
                func.avg(Zwischenruf.sentiment_score).label("avg_sentiment"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Redner, Redner.redner_id == Rede.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Redner.fraktion.isnot(None))
            .where(Zwischenruf.fraktion.isnot(None))
        )
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        stmt = stmt.group_by("sprecher_fraktion", "interruptor_fraktion")

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=["sprecher_fraktion", "interruptor_fraktion", "anzahl", "avg_sentiment"],
        )

        # ── 1. Normalise faction names ────────────────────────────────────
        df["sprecher_fraktion"] = df["sprecher_fraktion"].map(_canonicalise_faction)
        df["interruptor_fraktion"] = df["interruptor_fraktion"].map(_canonicalise_faction)

        # ── 2. Exclude self-interactions (diagonal) ───────────────────────
        if exclude_self:
            df = df[df["sprecher_fraktion"] != df["interruptor_fraktion"]]
        if df.empty:
            return pd.DataFrame()

        # ── 3. Re-aggregate with weighted average for sentiment ───────────
        # Compute weighted sentiment sum so that merging variants
        # (e.g. "LINKE" + "DIE LINKE") produces a proper weighted mean.
        df = df.assign(weighted_sent=df["avg_sentiment"].fillna(0) * df["anzahl"])
        df = df.infer_objects(copy=False)
        grp = (
            df.groupby(["sprecher_fraktion", "interruptor_fraktion"])
            .agg(total_count=("anzahl", "sum"), total_weighted=("weighted_sent", "sum"))
            .reset_index()
        )

        if score_weighted:
            # Aggression score = −weighted_avg_sentiment
            # Negative sentiment (interruptions/opposition) → positive value → red.
            grp["wert"] = -(grp["total_weighted"] / grp["total_count"].replace(0, 1))
        else:
            grp["wert"] = grp["total_count"]

        # ── 4. Per-capita normalisation ───────────────────────────────────
        if per_capita:
            faction_sizes = _get_faction_sizes(wahlperiode)
            grp["wert"] = grp.apply(
                lambda row: row["wert"] / faction_sizes.get(row["interruptor_fraktion"], 1),
                axis=1,
            )

        matrix = grp.pivot_table(
            index="sprecher_fraktion",
            columns="interruptor_fraktion",
            values="wert",
            fill_value=0,
            aggfunc="sum",
        )
        # Sort rows and columns by their total volume (descending) so that the
        # most active factions appear in the top-left corner and low-volume
        # outliers are pushed to the bottom-right.
        row_order = matrix.sum(axis=1).sort_values(ascending=False).index
        col_order = matrix.sum(axis=0).sort_values(ascending=False).index
        return matrix.loc[row_order, col_order]

    def edge_list(
        self,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Return an edge list suitable for NetworkX or Gephi export.

        Columns: source, target, weight, avg_sentiment, aggression_score.

        ``aggression_score`` = ``−avg_sentiment`` (clipped to ≥ 0), so that
        hostile interjections get a higher score than applause.
        """
        stmt = (
            select(
                Redner.fraktion.label("source"),
                Zwischenruf.fraktion.label("target"),
                func.count(Zwischenruf.ruf_id).label("weight"),
                func.avg(Zwischenruf.sentiment_score).label("avg_sentiment"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Redner, Redner.redner_id == Rede.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Redner.fraktion.isnot(None))
            .where(Zwischenruf.fraktion.isnot(None))
        )
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        stmt = stmt.group_by("source", "target")

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["source", "target", "weight", "avg_sentiment", "aggression_score"])

        df = pd.DataFrame(rows, columns=["source", "target", "weight", "avg_sentiment"])

        # ── 1. Normalise faction names ────────────────────────────────────
        df["source"] = df["source"].map(_canonicalise_faction)
        df["target"] = df["target"].map(_canonicalise_faction)

        # ── 2. Exclude self-interactions ──────────────────────────────────
        if exclude_self:
            df = df[df["source"] != df["target"]]
        if df.empty:
            return pd.DataFrame(columns=["source", "target", "weight", "avg_sentiment", "aggression_score"])

        # ── 3. Re-aggregate with weighted average for sentiment ───────────
        df = df.assign(weighted_sent=df["avg_sentiment"].fillna(0) * df["weight"])
        df = df.infer_objects(copy=False)
        agg = (
            df.groupby(["source", "target"])
            .agg(weight=("weight", "sum"), weighted_sent=("weighted_sent", "sum"))
            .reset_index()
        )
        # Zero-count rows cannot occur here (SQL COUNT always ≥ 1 for matched rows),
        # but guard with replace(0, 1) defensively.
        agg["avg_sentiment"] = (agg["weighted_sent"] / agg["weight"].replace(0, 1)).round(3)
        # aggression_score = −avg_sentiment, clipped to ≥ 0.
        # Positive sentiment (Beifall, avg > 0) maps to 0 (not aggressive).
        # Negative sentiment (Widerspruch, avg < 0) maps to |avg| > 0 (aggressive).
        agg["aggression_score"] = (-agg["avg_sentiment"]).clip(lower=0).round(3)
        agg = agg.drop(columns=["weighted_sent"]).sort_values("weight", ascending=False)

        return agg

    def to_networkx_graph(
        self,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
        exclude_self: bool = True,
    ) -> nx.DiGraph:
        """Build a directed NetworkX graph from the edge list.

        Nodes are faction names; edges carry ``weight``, ``avg_sentiment``,
        and ``aggression_score`` attributes.  The graph can be exported to
        GraphML or GEXF with ``nx.write_graphml`` / ``nx.write_gexf``.

        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        :param wahlperiode: Restrict to a specific Wahlperiode.
        :param exclude_self: Exclude self-loops (same-faction edges).
        :return: Directed weighted graph.
        """
        edge_df = self.edge_list(
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=wahlperiode,
            exclude_self=exclude_self,
        )
        G = nx.DiGraph()
        if edge_df.empty:
            return G
        for _, row in edge_df.iterrows():
            G.add_edge(
                row["source"],
                row["target"],
                weight=float(row["weight"]),
                avg_sentiment=float(row["avg_sentiment"]),
                aggression_score=float(row["aggression_score"]),
            )
        return G

    def to_graphml_bytes(
        self,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
    ) -> bytes:
        """Return the interaction graph serialised as GraphML (bytes).

        Suitable for use with ``st.download_button`` or direct file I/O.
        """
        G = self.to_networkx_graph(
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=wahlperiode,
        )
        buf = io.BytesIO()
        nx.write_graphml(G, buf)
        return buf.getvalue()

    def to_gexf_bytes(
        self,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        wahlperiode: Optional[int] = None,
    ) -> bytes:
        """Return the interaction graph serialised as GEXF (bytes).

        GEXF is natively supported by Gephi for richer visualisations.
        """
        G = self.to_networkx_graph(
            datum_von=datum_von,
            datum_bis=datum_bis,
            wahlperiode=wahlperiode,
        )
        buf = io.BytesIO()
        nx.write_gexf(G, buf)
        return buf.getvalue()

    def adjacency_matrix_by_window(
        self,
        wahlperiode: Optional[int] = None,
        window: str = "quarter",
        score_weighted: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Return a time-series of adjacency matrices, one per time window.

        The result is a dict mapping a human-readable label (e.g. "Q1 2022")
        to the corresponding adjacency matrix (same format as
        :meth:`adjacency_matrix`).  Empty windows are omitted.
        Insertion order is preserved (Python 3.7+) so iteration is
        chronological.

        This powers the *Netzwerk-Evolution* view in the dashboard where a
        timeline slider lets users animate how inter-faction hostilities shift
        across a legislative term.

        :param wahlperiode: Restrict to a specific Wahlperiode.
        :param window: Granularity – ``"quarter"`` (3-month) or ``"year"``.
            Any other value raises :class:`ValueError`.
        :param score_weighted: If True use aggression score; if False use
            interjection count.  Defaults to True for the evolution view.
        :return: ``dict[label, DataFrame]`` sorted chronologically.
        :raises ValueError: If *window* is not ``"quarter"`` or ``"year"``.
        """
        if window not in ("quarter", "year"):
            raise ValueError(
                f"Unsupported window {window!r}; expected 'quarter' or 'year'."
            )
        # Fetch the full date span for the requested Wahlperiode.
        stmt_dates = (
            select(
                func.min(Sitzung.datum).label("min_datum"),
                func.max(Sitzung.datum).label("max_datum"),
            )
        )
        if wahlperiode:
            stmt_dates = stmt_dates.where(Sitzung.wahlperiode == wahlperiode)
        row = self._db.execute(stmt_dates).fetchone()
        if not row or row[0] is None:
            return {}

        min_date: date = row[0]
        max_date: date = row[1]

        # Build a list of (start, end, label) tuples covering the date range.
        windows: List[Tuple[date, date, str]] = []
        current = date(min_date.year, min_date.month, 1)
        while current <= max_date:
            if window == "year":
                win_start = date(current.year, 1, 1)
                win_end = date(current.year, 12, 31)
                label = str(current.year)
                # Advance by one year
                current = date(current.year + 1, 1, 1)
            else:
                # Quarter: round current month down to the first month of the quarter.
                q_start_month = ((current.month - 1) // 3) * 3 + 1
                win_start = date(current.year, q_start_month, 1)
                # Quarter end: last day of the third month in the quarter
                q_end_month = q_start_month + 2
                q_end_year = current.year
                if q_end_month == 12:
                    win_end = date(q_end_year, 12, 31)
                else:
                    win_end = date(q_end_year, q_end_month + 1, 1) - timedelta(days=1)
                quarter_num = (q_start_month - 1) // 3 + 1
                label = f"Q{quarter_num} {current.year}"
                # Advance to the next quarter
                next_month = q_start_month + 3
                if next_month > 12:
                    current = date(current.year + 1, next_month - 12, 1)
                else:
                    current = date(current.year, next_month, 1)

            if win_end < min_date or win_start > max_date:
                continue
            windows.append((win_start, win_end, label))

        result: Dict[str, pd.DataFrame] = {}
        for win_start, win_end, label in windows:
            matrix = self.adjacency_matrix(
                datum_von=win_start,
                datum_bis=win_end,
                score_weighted=score_weighted,
                wahlperiode=wahlperiode,
            )
            if not matrix.empty:
                result[label] = matrix
        return result


# ─────────────────────────────────────────────────────────────────────────────
# D. Ton-Analyse
# ─────────────────────────────────────────────────────────────────────────────

class TonAnalyse:
    """Analyse tone-label distribution (Aggression/Sarkasmus/Humor/Neutral).

    Uses the ``ton_label`` field populated by ``ToneClassifier`` during the
    NLP pipeline run.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def ton_by_fraktion(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return tone-label counts grouped by interrupting faction.

        Columns: fraktion, ton_label, anzahl.
        """
        stmt = (
            select(
                Zwischenruf.fraktion,
                Zwischenruf.ton_label,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.fraktion.isnot(None))
            .where(Zwischenruf.ton_label.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Zwischenruf.fraktion, Zwischenruf.ton_label)

        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(rows, columns=["fraktion", "ton_label", "anzahl"])

    def ton_trend(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return tone-label counts per session date for trend analysis.

        Columns: datum, ton_label, anzahl.
        """
        stmt = (
            select(
                Sitzung.datum,
                Zwischenruf.ton_label,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.ton_label.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Sitzung.datum, Zwischenruf.ton_label).order_by(Sitzung.datum)

        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(rows, columns=["datum", "ton_label", "anzahl"])


# ─────────────────────────────────────────────────────────────────────────────
# E. Adressaten-Analyse
# ─────────────────────────────────────────────────────────────────────────────

class AdressatenAnalyse:
    """Analyse addressee patterns from interjection data.

    Uses the ``adressaten`` field populated by ``AddresseeDetector`` during the
    NLP pipeline run.  The field is persisted as a JSON-serialised list string
    (e.g. ``'["SPD", "CDU/CSU"]'``) via ``json.dumps``; use
    :func:`_parse_adressaten` to decode it rather than splitting on commas.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def top_adressaten(
        self,
        n: int = 20,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return the *n* most frequently targeted factions / persons.

        Columns: adressat, anzahl.
        """
        stmt = (
            select(
                Zwischenruf.adressaten,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.adressaten.isnot(None))
            .where(Zwischenruf.adressaten != "")
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Zwischenruf.adressaten).order_by(
            func.count(Zwischenruf.ruf_id).desc()
        )

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["adressat", "anzahl"])

        df = pd.DataFrame(rows, columns=["adressaten_raw", "anzahl"])
        # adressaten is stored as a JSON-serialised list – parse and explode into individual rows.
        records: List[Dict] = []
        for _, row in df.iterrows():
            for adressat in _parse_adressaten(row["adressaten_raw"]):
                if adressat:
                    records.append({"adressat": adressat, "anzahl": row["anzahl"]})

        if not records:
            return pd.DataFrame(columns=["adressat", "anzahl"])

        result = pd.DataFrame(records)
        return (
            result.groupby("adressat")["anzahl"]
            .sum()
            .reset_index()
            .sort_values("anzahl", ascending=False)
            .head(n)
        )

    def fraktion_targets_fraktion(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return a heatmap-ready DataFrame: which faction targets which faction.

        Columns: fraktion (interruptor), adressat, anzahl.
        """
        stmt = (
            select(
                Zwischenruf.fraktion,
                Zwischenruf.adressaten,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.fraktion.isnot(None))
            .where(Zwischenruf.adressaten.isnot(None))
            .where(Zwischenruf.adressaten != "")
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Zwischenruf.fraktion, Zwischenruf.adressaten)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["fraktion", "adressat", "anzahl"])

        df = pd.DataFrame(rows, columns=["fraktion", "adressaten_raw", "anzahl"])
        records: List[Dict] = []
        for _, row in df.iterrows():
            for adressat in _parse_adressaten(row["adressaten_raw"]):
                if adressat:
                    records.append({
                        "fraktion": row["fraktion"],
                        "adressat": adressat,
                        "anzahl": row["anzahl"],
                    })

        if not records:
            return pd.DataFrame(columns=["fraktion", "adressat", "anzahl"])

        result = pd.DataFrame(records)
        return (
            result.groupby(["fraktion", "adressat"])["anzahl"]
            .sum()
            .reset_index()
            .sort_values("anzahl", ascending=False)
        )


# ─────────────────────────────────────────────────────────────────────────────
# F. Scraping-Monitor
# ─────────────────────────────────────────────────────────────────────────────

class ScrapingMonitor:
    """Provide an overview of the current database fill-state and NLP coverage."""

    def __init__(self, session: Session) -> None:
        self._db = session

    def overview(self) -> pd.DataFrame:
        """Return session / speech counts grouped by Wahlperiode.

        Columns: wahlperiode, sitzungen, reden.
        """
        stmt = (
            select(
                Sitzung.wahlperiode,
                func.count(Sitzung.sitzungs_id.distinct()).label("sitzungen"),
                func.count(Rede.rede_id).label("reden"),
            )
            .outerjoin(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
            .group_by(Sitzung.wahlperiode)
            .order_by(Sitzung.wahlperiode)
        )
        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(rows, columns=["wahlperiode", "sitzungen", "reden"])

    def zwischenruf_stats(self) -> pd.DataFrame:
        """Return interjection counts and NLP-coverage per Wahlperiode.

        Columns: wahlperiode, gesamt, mit_sentiment, mit_ton_label,
                 mit_adressaten, mit_sentiment_pct, mit_ton_label_pct,
                 mit_adressaten_pct.
        """
        stmt = (
            select(
                Sitzung.wahlperiode,
                func.count(Zwischenruf.ruf_id).label("gesamt"),
                func.sum(
                    case((Zwischenruf.sentiment_score.isnot(None), 1), else_=0)
                ).label("mit_sentiment"),
                func.sum(
                    case((Zwischenruf.ton_label.isnot(None), 1), else_=0)
                ).label("mit_ton_label"),
                func.sum(
                    case((Zwischenruf.adressaten.isnot(None), 1), else_=0)
                ).label("mit_adressaten"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .group_by(Sitzung.wahlperiode)
            .order_by(Sitzung.wahlperiode)
        )
        rows = self._db.execute(stmt).fetchall()
        df = pd.DataFrame(
            rows,
            columns=["wahlperiode", "gesamt", "mit_sentiment", "mit_ton_label", "mit_adressaten"],
        )
        for col in ["gesamt", "mit_sentiment", "mit_ton_label", "mit_adressaten"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["mit_sentiment", "mit_ton_label", "mit_adressaten"]:
            df[f"{col}_pct"] = (
                df[col] / df["gesamt"].replace(0, 1) * 100
            ).round(1)
        return df

    def recent_sitzungen(self, n: int = 10) -> pd.DataFrame:
        """Return the *n* most recently scraped sessions.

        Columns: wahlperiode, sitzungsnr, datum, titel, gesamtwortzahl, reden.
        """
        stmt = (
            select(
                Sitzung.wahlperiode,
                Sitzung.sitzungsnr,
                Sitzung.datum,
                Sitzung.titel,
                Sitzung.gesamtwortzahl,
                func.count(Rede.rede_id).label("reden"),
            )
            .outerjoin(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
            .group_by(Sitzung.sitzungs_id)
            .order_by(Sitzung.datum.desc().nullslast())
            .limit(n)
        )
        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(
            rows,
            columns=["wahlperiode", "sitzungsnr", "datum", "titel", "gesamtwortzahl", "reden"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# G. WahlperiodenVergleich
# ─────────────────────────────────────────────────────────────────────────────

class WahlperiodenVergleich:
    """Cross-legislature comparison of key metrics.

    Compares aggression levels, tone-label distributions, interjection
    volumes, and speech activity across multiple Wahlperioden so that
    users can spot trends and structural differences between legislative
    terms directly in the dashboard.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def aggression_by_wp(self) -> pd.DataFrame:
        """Return the average aggression score (−avg_sentiment) per Wahlperiode.

        Columns: wahlperiode, avg_aggression, neg_zwischenrufe, gesamt_zwischenrufe,
                 anteil_negativ_pct.

        ``avg_aggression`` is the mean of ``−sentiment_score`` across all
        interjections with a non-NULL sentiment score.  Higher → more hostile.
        ``anteil_negativ_pct`` is the share of interjections with a negative
        sentiment (sentiment_score < 0).
        """
        stmt = (
            select(
                Sitzung.wahlperiode,
                func.count(Zwischenruf.ruf_id).label("gesamt"),
                func.sum(
                    case((Zwischenruf.sentiment_score < 0, 1), else_=0)
                ).label("neg"),
                func.avg(Zwischenruf.sentiment_score).label("avg_sent"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.sentiment_score.isnot(None))
            .group_by(Sitzung.wahlperiode)
            .order_by(Sitzung.wahlperiode)
        )
        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["wahlperiode", "avg_aggression", "neg_zwischenrufe",
                         "gesamt_zwischenrufe", "anteil_negativ_pct"]
            )
        df = pd.DataFrame(rows, columns=["wahlperiode", "gesamt", "neg", "avg_sent"])
        df["avg_aggression"] = (-df["avg_sent"]).round(3)
        df["anteil_negativ_pct"] = (df["neg"] / df["gesamt"].replace(0, 1) * 100).round(1)
        return df.rename(columns={"gesamt": "gesamt_zwischenrufe", "neg": "neg_zwischenrufe"})[
            ["wahlperiode", "avg_aggression", "neg_zwischenrufe",
             "gesamt_zwischenrufe", "anteil_negativ_pct"]
        ]

    def ton_by_wp(self) -> pd.DataFrame:
        """Return tone-label distribution per Wahlperiode (percentages).

        Columns: wahlperiode, ton_label, anzahl, anteil_pct.
        """
        stmt = (
            select(
                Sitzung.wahlperiode,
                Zwischenruf.ton_label,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.ton_label.isnot(None))
            .group_by(Sitzung.wahlperiode, Zwischenruf.ton_label)
            .order_by(Sitzung.wahlperiode, Zwischenruf.ton_label)
        )
        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["wahlperiode", "ton_label", "anzahl", "anteil_pct"])
        df = pd.DataFrame(rows, columns=["wahlperiode", "ton_label", "anzahl"])
        totals = df.groupby("wahlperiode")["anzahl"].transform("sum")
        df["anteil_pct"] = (df["anzahl"] / totals.replace(0, 1) * 100).round(1)
        return df

    def activity_by_wp(self) -> pd.DataFrame:
        """Return high-level activity statistics per Wahlperiode.

        Columns: wahlperiode, sitzungen, reden, zwischenrufe,
                 zwischenrufe_pro_rede, worte_pro_sitzung.
        """
        stmt_sz = (
            select(
                Sitzung.wahlperiode,
                func.count(Sitzung.sitzungs_id.distinct()).label("sitzungen"),
                func.count(Rede.rede_id).label("reden"),
            )
            .outerjoin(Rede, Rede.sitzung_id == Sitzung.sitzungs_id)
            .group_by(Sitzung.wahlperiode)
            .order_by(Sitzung.wahlperiode)
        )
        rows_sz = self._db.execute(stmt_sz).fetchall()
        if not rows_sz:
            return pd.DataFrame(
                columns=["wahlperiode", "sitzungen", "reden", "zwischenrufe",
                         "zwischenrufe_pro_rede", "worte_pro_sitzung"]
            )
        df_sz = pd.DataFrame(rows_sz, columns=["wahlperiode", "sitzungen", "reden"])

        stmt_wt = (
            select(
                Sitzung.wahlperiode,
                func.sum(Sitzung.gesamtwortzahl).label("gesamtworte"),
            )
            .group_by(Sitzung.wahlperiode)
        )
        rows_wt = self._db.execute(stmt_wt).fetchall()
        df_wt = pd.DataFrame(rows_wt, columns=["wahlperiode", "gesamtworte"]) if rows_wt else pd.DataFrame(columns=["wahlperiode", "gesamtworte"])
        df_sz = df_sz.merge(df_wt, on="wahlperiode", how="left")

        stmt_zr = (
            select(
                Sitzung.wahlperiode,
                func.count(Zwischenruf.ruf_id).label("zwischenrufe"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .group_by(Sitzung.wahlperiode)
        )
        rows_zr = self._db.execute(stmt_zr).fetchall()
        df_zr = pd.DataFrame(rows_zr, columns=["wahlperiode", "zwischenrufe"]) if rows_zr else pd.DataFrame(columns=["wahlperiode", "zwischenrufe"])

        df = df_sz.merge(df_zr, on="wahlperiode", how="left")
        df["zwischenrufe"] = df["zwischenrufe"].fillna(0).astype(int)
        df["zwischenrufe_pro_rede"] = (
            df["zwischenrufe"] / df["reden"].replace(0, 1)
        ).round(2)
        df["worte_pro_sitzung"] = (
            df["gesamtworte"].fillna(0) / df["sitzungen"].replace(0, 1)
        ).round(0).astype(int)
        return df[["wahlperiode", "sitzungen", "reden", "zwischenrufe",
                   "zwischenrufe_pro_rede", "worte_pro_sitzung"]]


# ─────────────────────────────────────────────────────────────────────────────
# H. TOPAnalyse  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

class TOPAnalyse:
    """Analyse how hostile parliament is towards different agenda items (TOPs).

    Uses the ``Rede.tagesordnungspunkt`` field (populated at parse time) together
    with the ``Zwischenruf.sentiment_score`` and ``Zwischenruf.kategorie`` fields
    to answer: *which policy topics structurally provoke the most hostile reactions?*
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def aggression_by_top(
        self,
        n: int = 20,
        wahlperiode: Optional[int] = None,
        min_reden: int = 3,
    ) -> pd.DataFrame:
        """Return the *n* most combative agenda items ranked by average aggression.

        Only TOPs with at least *min_reden* speeches are included to filter out
        one-off agenda items with insufficient statistical mass.

        Columns: tagesordnungspunkt, avg_aggression, neg_zwischenrufe,
                 gesamt_zwischenrufe, anteil_negativ_pct, anzahl_reden.

        :param n: Number of top results to return.
        :param wahlperiode: Optional Wahlperiode filter.
        :param min_reden: Minimum number of speeches required for inclusion.
        """
        stmt = (
            select(
                Rede.tagesordnungspunkt,
                func.count(Zwischenruf.ruf_id).label("gesamt_zwr"),
                func.sum(
                    case((Zwischenruf.sentiment_score < -0.3, 1), else_=0)
                ).label("neg_zwr"),
                func.avg(Zwischenruf.sentiment_score).label("avg_sent"),
                func.count(Rede.rede_id.distinct()).label("anzahl_reden"),
            )
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Rede.tagesordnungspunkt.isnot(None))
            .where(Zwischenruf.sentiment_score.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        stmt = stmt.group_by(Rede.tagesordnungspunkt)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["tagesordnungspunkt", "avg_aggression", "neg_zwischenrufe",
                         "gesamt_zwischenrufe", "anteil_negativ_pct", "anzahl_reden"]
            )

        df = pd.DataFrame(
            rows,
            columns=["tagesordnungspunkt", "gesamt_zwischenrufe", "neg_zwischenrufe",
                     "avg_sent", "anzahl_reden"],
        )
        # Cast numerics
        for col in ["gesamt_zwischenrufe", "neg_zwischenrufe", "anzahl_reden"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Filter TOPs with too few speeches to be statistically meaningful
        df = df[df["anzahl_reden"] >= min_reden]
        if df.empty:
            return pd.DataFrame(
                columns=["tagesordnungspunkt", "avg_aggression", "neg_zwischenrufe",
                         "gesamt_zwischenrufe", "anteil_negativ_pct", "anzahl_reden"]
            )

        df["avg_aggression"] = (-pd.to_numeric(df["avg_sent"], errors="coerce")).round(3)
        df["anteil_negativ_pct"] = (
            df["neg_zwischenrufe"] / df["gesamt_zwischenrufe"].replace(0, 1) * 100
        ).round(1)
        return (
            df[["tagesordnungspunkt", "avg_aggression", "neg_zwischenrufe",
                "gesamt_zwischenrufe", "anteil_negativ_pct", "anzahl_reden"]]
            .sort_values("avg_aggression", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def kategorie_by_top(
        self,
        n: int = 20,
        wahlperiode: Optional[int] = None,
        min_reden: int = 3,
    ) -> pd.DataFrame:
        """Return interjection category distribution per agenda item.

        Columns: tagesordnungspunkt, kategorie, anzahl.

        Restricted to the top *n* most active agenda items (by total interjection
        count) so that the chart remains readable.

        :param n: Number of top agenda items to include.
        :param wahlperiode: Optional Wahlperiode filter.
        :param min_reden: Minimum speech count for inclusion.
        """
        stmt = (
            select(
                Rede.tagesordnungspunkt,
                Zwischenruf.kategorie,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
                func.count(Rede.rede_id.distinct()).label("anzahl_reden"),
            )
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Rede.tagesordnungspunkt.isnot(None))
            .where(Zwischenruf.kategorie.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        stmt = stmt.group_by(Rede.tagesordnungspunkt, Zwischenruf.kategorie)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(columns=["tagesordnungspunkt", "kategorie", "anzahl"])

        df = pd.DataFrame(
            rows,
            columns=["tagesordnungspunkt", "kategorie", "anzahl", "anzahl_reden"],
        )
        df["anzahl"] = pd.to_numeric(df["anzahl"], errors="coerce").fillna(0).astype(int)
        df["anzahl_reden"] = pd.to_numeric(df["anzahl_reden"], errors="coerce").fillna(0).astype(int)

        # Keep only TOPs with sufficient speech coverage
        valid_tops = (
            df.groupby("tagesordnungspunkt")["anzahl_reden"]
            .max()
            .loc[lambda s: s >= min_reden]
            .index
        )
        df = df[df["tagesordnungspunkt"].isin(valid_tops)]
        if df.empty:
            return pd.DataFrame(columns=["tagesordnungspunkt", "kategorie", "anzahl"])

        # Restrict to top-n TOPs by total interjection volume
        top_tops = (
            df.groupby("tagesordnungspunkt")["anzahl"]
            .sum()
            .nlargest(n)
            .index
        )
        return (
            df[df["tagesordnungspunkt"].isin(top_tops)][
                ["tagesordnungspunkt", "kategorie", "anzahl"]
            ]
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# I. KategorieAnalyse  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

# Category labels stored in Zwischenruf.kategorie
_POSITIVE_KATEGORIEN = {"Beifall", "Zustimmung"}
_NEGATIVE_KATEGORIEN = {"Widerspruch", "Unruhe"}


class KategorieAnalyse:
    """Analyse interjection category distribution and civility ratios.

    Moves beyond raw sentiment scores to use the human-readable ``kategorie``
    field (populated at parse time from explicit markers in the protocol XML,
    e.g. "(Beifall bei der SPD)").

    Categories:
        Positive: Beifall, Zustimmung
        Negative: Widerspruch, Unruhe
        Neutral:  Lachen, Zwischenruf, Sonstiges
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def kategorie_by_fraktion(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
        mode: str = "given",
    ) -> pd.DataFrame:
        """Return category counts per faction.

        :param mode: ``"given"`` = categories produced by a faction (interruptor),
                     ``"received"`` = categories received by the *speaking* faction.
        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.

        Columns: fraktion, kategorie, anzahl.
        :raises ValueError: If *mode* is not ``"given"`` or ``"received"``.
        """
        if mode == "given":
            fraktion_col = Zwischenruf.fraktion
            stmt = (
                select(
                    fraktion_col.label("fraktion"),
                    Zwischenruf.kategorie,
                    func.count(Zwischenruf.ruf_id).label("anzahl"),
                )
                .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
                .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
                .where(fraktion_col.isnot(None))
                .where(Zwischenruf.kategorie.isnot(None))
            )
        elif mode == "received":
            fraktion_col = Redner.fraktion
            stmt = (
                select(
                    fraktion_col.label("fraktion"),
                    Zwischenruf.kategorie,
                    func.count(Zwischenruf.ruf_id).label("anzahl"),
                )
                .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
                .join(Redner, Redner.redner_id == Rede.redner_id)
                .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
                .where(fraktion_col.isnot(None))
                .where(Zwischenruf.kategorie.isnot(None))
            )
        else:
            raise ValueError(
                f"Unsupported mode {mode!r}; expected 'given' or 'received'."
            )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(fraktion_col, Zwischenruf.kategorie)

        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(rows, columns=["fraktion", "kategorie", "anzahl"])

    def beifall_widerspruch_ratio(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return the civility index for each faction-pair.

        For each (interruptor_fraktion, speaking_fraktion) pair the *civility
        index* is defined as:

            civility = (Beifall + Zustimmung) / (Widerspruch + Unruhe + 1)

        A value > 1 means the pair exchanges more applause than disruption.
        A value < 1 means mostly hostile interactions.

        Columns: interruptor_fraktion, sprecher_fraktion, beifall, widerspruch,
                 civility_index.
        """
        stmt = (
            select(
                Zwischenruf.fraktion.label("interruptor_fraktion"),
                Redner.fraktion.label("sprecher_fraktion"),
                Zwischenruf.kategorie,
                func.count(Zwischenruf.ruf_id).label("anzahl"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Redner, Redner.redner_id == Rede.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.fraktion.isnot(None))
            .where(Redner.fraktion.isnot(None))
            .where(Zwischenruf.kategorie.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(
            Zwischenruf.fraktion, Redner.fraktion, Zwischenruf.kategorie
        )

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["interruptor_fraktion", "sprecher_fraktion",
                         "beifall", "widerspruch", "civility_index"]
            )

        df = pd.DataFrame(
            rows,
            columns=["interruptor_fraktion", "sprecher_fraktion", "kategorie", "anzahl"],
        )
        df["anzahl"] = pd.to_numeric(df["anzahl"], errors="coerce").fillna(0).astype(int)

        # Pivot to wide format: one row per faction-pair
        df["pos"] = df["kategorie"].isin(_POSITIVE_KATEGORIEN).astype(int) * df["anzahl"]
        df["neg"] = df["kategorie"].isin(_NEGATIVE_KATEGORIEN).astype(int) * df["anzahl"]
        pair = (
            df.groupby(["interruptor_fraktion", "sprecher_fraktion"])
            .agg(beifall=("pos", "sum"), widerspruch=("neg", "sum"))
            .reset_index()
        )
        pair["civility_index"] = (pair["beifall"] / (pair["widerspruch"] + 1)).round(3)
        return pair.sort_values("civility_index", ascending=False).reset_index(drop=True)

    def lachen_by_redner(
        self,
        n: int = 15,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return the top *n* speakers ranked by "Lachen" interjections received.

        Columns: nachname, vorname, fraktion, lachen_count.

        :param n: Number of top speakers to return.
        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        """
        stmt = (
            select(
                Redner.nachname,
                Redner.vorname,
                Redner.fraktion,
                func.count(Zwischenruf.ruf_id).label("lachen_count"),
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.kategorie == "Lachen")
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = (
            stmt.group_by(Redner.redner_id)
            .order_by(func.count(Zwischenruf.ruf_id).desc())
            .limit(n)
        )

        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(rows, columns=["nachname", "vorname", "fraktion", "lachen_count"])


# ─────────────────────────────────────────────────────────────────────────────
# M. RedeZeitAnalyse  (v2.2.0)
# ─────────────────────────────────────────────────────────────────────────────

class RedeZeitAnalyse:
    """Analyse speech-time distribution and fairness relative to faction size.

    Uses ``Rede.wortanzahl`` (stored at parse time) together with the hardcoded
    ``FACTION_SIZES_BY_WAHLPERIODE`` to compute a *fairness index*:

        fairness_index = (faction_word_share%) / (faction_seat_share%)

    A value of 1.0 means perfectly proportional speech time.
    Values > 1.0 indicate over-representation; < 1.0 under-representation.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def wortanzahl_by_fraktion(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return total and average word count per faction.

        Columns: fraktion, total_worte, avg_worte_pro_rede, anzahl_reden.
        """
        stmt = (
            select(
                Redner.fraktion,
                func.sum(Rede.wortanzahl).label("total_worte"),
                func.avg(Rede.wortanzahl).label("avg_worte"),
                func.count(Rede.rede_id).label("anzahl_reden"),
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Redner.fraktion.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Redner.fraktion).order_by(
            func.sum(Rede.wortanzahl).desc()
        )

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["fraktion", "total_worte", "avg_worte_pro_rede", "anzahl_reden"]
            )
        df = pd.DataFrame(
            rows, columns=["fraktion", "total_worte", "avg_worte_pro_rede", "anzahl_reden"]
        )
        df["fraktion"] = df["fraktion"].map(_canonicalise_faction)
        return df.groupby("fraktion", as_index=False).agg(
            total_worte=("total_worte", "sum"),
            avg_worte_pro_rede=("avg_worte_pro_rede", "mean"),
            anzahl_reden=("anzahl_reden", "sum"),
        ).sort_values("total_worte", ascending=False).reset_index(drop=True)

    def fairness_index(
        self,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the speech-time fairness index per faction.

        Compares the faction's share of total Bundestag words against its
        proportional seat share.

        Columns: fraktion, total_worte, wort_anteil_pct, sitz_anteil_pct,
                 fairness_index, ueber_unterrepraesentation.

        ``fairness_index`` = wort_anteil_pct / sitz_anteil_pct.
        ``ueber_unterrepraesentation`` is a human-readable label:
            "Überrepräsentiert" if > 1.1, "Unterrepräsentiert" if < 0.9, else "Proportional".

        :param wahlperiode: Wahlperiode to use for seat-size lookup and filtering.
        """
        df = self.wortanzahl_by_fraktion(wahlperiode=wahlperiode)
        if df.empty:
            return pd.DataFrame(
                columns=["fraktion", "total_worte", "wort_anteil_pct",
                         "sitz_anteil_pct", "fairness_index",
                         "ueber_unterrepraesentation"]
            )

        total_words = df["total_worte"].sum()
        df["wort_anteil_pct"] = (df["total_worte"] / max(total_words, 1) * 100).round(2)

        faction_sizes = _get_faction_sizes(wahlperiode)
        total_seats = sum(faction_sizes.values())
        df["sitze"] = df["fraktion"].map(faction_sizes)
        df = df[df["sitze"].notna()].copy()
        df["sitze"] = df["sitze"].astype(int)
        df["sitz_anteil_pct"] = (df["sitze"] / max(total_seats, 1) * 100).round(2)
        df["fairness_index"] = (
            df["wort_anteil_pct"] / df["sitz_anteil_pct"].replace(0, 1)
        ).round(3)

        def _label(v: float) -> str:
            if v > 1.1:
                return "Überrepräsentiert"
            if v < 0.9:
                return "Unterrepräsentiert"
            return "Proportional"

        df["ueber_unterrepraesentation"] = df["fairness_index"].apply(_label)
        return df[
            ["fraktion", "total_worte", "wort_anteil_pct", "sitz_anteil_pct",
             "fairness_index", "ueber_unterrepraesentation"]
        ].reset_index(drop=True)

    def top_redselige_redner(
        self,
        n: int = 20,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return the top *n* most verbose MPs by total word count.

        Columns: nachname, vorname, fraktion, total_worte, anzahl_reden.

        :param n: Number of top speakers to return.
        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        """
        stmt = (
            select(
                Redner.nachname,
                Redner.vorname,
                Redner.fraktion,
                func.sum(Rede.wortanzahl).label("total_worte"),
                func.count(Rede.rede_id).label("anzahl_reden"),
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = (
            stmt.group_by(Redner.redner_id)
            .order_by(func.sum(Rede.wortanzahl).desc())
            .limit(n)
        )

        rows = self._db.execute(stmt).fetchall()
        return pd.DataFrame(
            rows, columns=["nachname", "vorname", "fraktion", "total_worte", "anzahl_reden"]
        )

    def wortanzahl_vs_zwischenrufe(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return a scatter-ready DataFrame: word count vs. negative interjections
        received per speaker.

        Columns: nachname, vorname, fraktion, total_worte, neg_zwischenrufe,
                 neg_pro_100_worte.

        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        """
        stmt = (
            select(
                Redner.nachname,
                Redner.vorname,
                Redner.fraktion,
                func.sum(Rede.wortanzahl).label("total_worte"),
                func.count(Zwischenruf.ruf_id).label("neg_zwischenrufe"),
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Zwischenruf, Zwischenruf.rede_id == Rede.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.sentiment_score < -0.3)
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = stmt.group_by(Redner.redner_id)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["nachname", "vorname", "fraktion",
                         "total_worte", "neg_zwischenrufe", "neg_pro_100_worte"]
            )
        df = pd.DataFrame(
            rows,
            columns=["nachname", "vorname", "fraktion", "total_worte", "neg_zwischenrufe"],
        )
        df["neg_pro_100_worte"] = (
            df["neg_zwischenrufe"] / df["total_worte"].replace(0, 1) * 100
        ).round(2)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# L. SitzungsKlima  (v2.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class SitzungsKlima:
    """Compute a per-session composite parliamentary "temperature" index.

    The temperature index combines:
        - average interjection sentiment (inverted: lower sentiment → higher heat)
        - share of aggressive tone labels among interjections
        - number of interjections per speech (interruption density)
        - share of "Unruhe" (unrest) category interjections

    Each component is normalised to [0, 1] using the session-level extremes
    within the selected Wahlperiode so that the composite index is also in [0, 1].
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def klima_per_sitzung(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return per-session climate metrics.

        Columns: sitzungsnr, datum, avg_sentiment, anteil_aggression_pct,
                 zwischenrufe_pro_rede, anteil_unruhe_pct, temperatur_index.

        ``temperatur_index`` is a composite score in [0, 1] where 1.0 is the
        hottest session (most hostile) and 0.0 is the calmest.

        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        """
        stmt = (
            select(
                Sitzung.sitzungsnr,
                Sitzung.datum,
                func.avg(Zwischenruf.sentiment_score).label("avg_sent"),
                func.count(Zwischenruf.ruf_id).label("gesamt_zwr"),
                func.sum(
                    case((Zwischenruf.ton_label == "Aggression", 1), else_=0)
                ).label("agg_count"),
                func.sum(
                    case((Zwischenruf.kategorie == "Unruhe", 1), else_=0)
                ).label("unruhe_count"),
                func.count(Rede.rede_id.distinct()).label("anzahl_reden"),
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Zwischenruf.sentiment_score.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)
        stmt = (
            stmt.group_by(Sitzung.sitzungs_id)
            .order_by(Sitzung.datum)
        )

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["sitzungsnr", "datum", "avg_sentiment",
                         "anteil_aggression_pct", "zwischenrufe_pro_rede",
                         "anteil_unruhe_pct", "temperatur_index"]
            )

        df = pd.DataFrame(
            rows,
            columns=["sitzungsnr", "datum", "avg_sent", "gesamt_zwr",
                     "agg_count", "unruhe_count", "anzahl_reden"],
        )
        for col in ["gesamt_zwr", "agg_count", "unruhe_count", "anzahl_reden"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df["avg_sent"] = pd.to_numeric(df["avg_sent"], errors="coerce")

        df["avg_sentiment"] = df["avg_sent"].round(3)
        # heat_sent: higher = more hostile (invert sentiment)
        df["heat_sent"] = (-df["avg_sent"]).clip(lower=0)
        df["anteil_aggression_pct"] = (
            df["agg_count"] / df["gesamt_zwr"].replace(0, 1) * 100
        ).round(1)
        df["zwischenrufe_pro_rede"] = (
            df["gesamt_zwr"] / df["anzahl_reden"].replace(0, 1)
        ).round(2)
        df["anteil_unruhe_pct"] = (
            df["unruhe_count"] / df["gesamt_zwr"].replace(0, 1) * 100
        ).round(1)

        # Build composite temperatur_index (min-max normalise each component)
        def _minmax(series: pd.Series) -> pd.Series:
            lo, hi = series.min(), series.max()
            if hi == lo:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - lo) / (hi - lo)

        df["temperatur_index"] = (
            _minmax(df["heat_sent"])
            + _minmax(df["anteil_aggression_pct"])
            + _minmax(df["zwischenrufe_pro_rede"])
            + _minmax(df["anteil_unruhe_pct"])
        ) / 4

        df["temperatur_index"] = df["temperatur_index"].round(4)
        return df[
            ["sitzungsnr", "datum", "avg_sentiment",
             "anteil_aggression_pct", "zwischenrufe_pro_rede",
             "anteil_unruhe_pct", "temperatur_index"]
        ]

    def hottest_sessions(
        self,
        n: int = 15,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the *n* hottest (most hostile) sessions by temperatur_index.

        Columns: sitzungsnr, datum, temperatur_index, avg_sentiment,
                 anteil_aggression_pct, zwischenrufe_pro_rede, anteil_unruhe_pct.

        :param n: Number of hottest sessions to return.
        :param wahlperiode: Optional Wahlperiode filter.
        """
        df = self.klima_per_sitzung(wahlperiode=wahlperiode)
        if df.empty:
            return df
        return (
            df.sort_values("temperatur_index", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# K. RednerProfil  (v2.4.0)
# ─────────────────────────────────────────────────────────────────────────────

_TONE_LABELS_PROFILE = ["Aggression", "Sarkasmus", "Humor", "Neutral"]


class RednerProfil:
    """Map the unique rhetorical fingerprint of each MP from their tone_scores JSON.

    Uses the ``Rede.tone_scores`` field (a JSON dict of label→probability stored
    by ``ToneClassifier``) to build a 4-dimensional tone vector per speaker.

    Methods:
        speaker_profile      – Average tone profile for a single speaker.
        top_speakers_by_tone – Top MPs ranked by probability mass on a given label.
        faction_profile      – Average tone profile per faction.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def _fetch_tone_scores(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch all (redner_id, nachname, vorname, fraktion, tone_scores) rows."""
        stmt = (
            select(
                Redner.redner_id,
                Redner.nachname,
                Redner.vorname,
                Redner.fraktion,
                Rede.tone_scores,
            )
            .join(Rede, Rede.redner_id == Redner.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Rede.tone_scores.isnot(None))
        )
        if wahlperiode:
            stmt = stmt.where(Sitzung.wahlperiode == wahlperiode)
        if datum_von:
            stmt = stmt.where(Sitzung.datum >= datum_von)
        if datum_bis:
            stmt = stmt.where(Sitzung.datum <= datum_bis)

        rows = self._db.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["redner_id", "nachname", "vorname", "fraktion", "tone_scores"]
            )
        return pd.DataFrame(
            rows, columns=["redner_id", "nachname", "vorname", "fraktion", "tone_scores"]
        )

    def _expand_tone_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand the tone_scores JSON column into separate float columns."""
        for label in _TONE_LABELS_PROFILE:
            df[label] = df["tone_scores"].apply(
                lambda ts, lbl=label: float(ts.get(lbl, 0.0)) if isinstance(ts, dict) else 0.0
            )
        return df

    def speaker_profile(
        self,
        redner_id: int,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the average rhetorical tone profile for a single speaker.

        Columns: label, avg_probability.

        :param redner_id: The ``Redner.redner_id`` to profile.
        :param wahlperiode: Optional Wahlperiode filter.
        """
        df = self._fetch_tone_scores(wahlperiode=wahlperiode)
        df = df[df["redner_id"] == redner_id]
        if df.empty:
            return pd.DataFrame(columns=["label", "avg_probability"])
        df = self._expand_tone_scores(df)
        profile = {lbl: df[lbl].mean() for lbl in _TONE_LABELS_PROFILE}
        return pd.DataFrame(
            [{"label": k, "avg_probability": round(v, 4)} for k, v in profile.items()]
        )

    def top_speakers_by_tone(
        self,
        ton_label: str,
        n: int = 15,
        wahlperiode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return top *n* MPs ranked by their average probability for *ton_label*.

        Columns: nachname, vorname, fraktion, avg_probability.

        :param ton_label: One of "Aggression", "Sarkasmus", "Humor", "Neutral".
        :param n: Number of top speakers to return.
        :param wahlperiode: Optional Wahlperiode filter.
        """
        if ton_label not in _TONE_LABELS_PROFILE:
            return pd.DataFrame(columns=["nachname", "vorname", "fraktion", "avg_probability"])

        df = self._fetch_tone_scores(wahlperiode=wahlperiode)
        if df.empty:
            return pd.DataFrame(columns=["nachname", "vorname", "fraktion", "avg_probability"])
        df = self._expand_tone_scores(df)
        ranked = (
            df.groupby(["redner_id", "nachname", "vorname", "fraktion"])[ton_label]
            .mean()
            .reset_index()
            .rename(columns={ton_label: "avg_probability"})
            .sort_values("avg_probability", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        ranked["avg_probability"] = ranked["avg_probability"].round(4)
        return ranked[["nachname", "vorname", "fraktion", "avg_probability"]]

    def faction_profile(
        self,
        wahlperiode: Optional[int] = None,
        datum_von: Optional[date] = None,
        datum_bis: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return the average tone profile per faction.

        Columns: fraktion, Aggression, Sarkasmus, Humor, Neutral.

        :param wahlperiode: Optional Wahlperiode filter.
        :param datum_von: Start date filter.
        :param datum_bis: End date filter.
        """
        df = self._fetch_tone_scores(
            wahlperiode=wahlperiode, datum_von=datum_von, datum_bis=datum_bis
        )
        if df.empty:
            return pd.DataFrame(
                columns=["fraktion"] + _TONE_LABELS_PROFILE
            )
        df = self._expand_tone_scores(df)
        df["fraktion"] = df["fraktion"].map(
            lambda f: _canonicalise_faction(f) if f else f
        )
        result = (
            df.groupby("fraktion")[_TONE_LABELS_PROFILE]
            .mean()
            .round(4)
            .reset_index()
        )
        return result
