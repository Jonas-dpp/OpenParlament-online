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
SQLAlchemy ORM models for OpenParlament.

Domain objects
──────────────
Sitzung      – A plenary session (Plenarsitzung) of the Bundestag.
Redner       – A speaker (Abgeordneter / Minister).
Rede         – A single speech held during a Sitzung.
Zwischenruf  – A verbal or non-verbal interjection during a Rede.

Relationships
─────────────
Sitzung  1 ── n  Rede
Redner   1 ── n  Rede
Rede     1 ── n  Zwischenruf
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ─────────────────────────────────────────────────────────────────────────────
# Sitzung
# ─────────────────────────────────────────────────────────────────────────────

class Sitzung(Base):
    """Represents a single plenary session of the Bundestag."""

    __tablename__ = "sitzungen"

    sitzungs_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wahlperiode: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sitzungsnr: Mapped[int] = mapped_column(Integer, nullable=False)
    datum: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    # German weekday name derived from datum: "Montag" … "Sonntag".
    wochentag: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    titel: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    gesamtwortzahl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    reden: Mapped[List["Rede"]] = relationship(
        "Rede", back_populates="sitzung", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("wahlperiode", "sitzungsnr", name="uq_sitzung_wp_nr"),
    )

    def __repr__(self) -> str:
        return (
            f"<Sitzung wahlperiode={self.wahlperiode} "
            f"nr={self.sitzungsnr} datum={self.datum} wochentag={self.wochentag}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Redner
# ─────────────────────────────────────────────────────────────────────────────

class Redner(Base):
    """Represents a speaker (MP, minister, etc.)."""

    __tablename__ = "redner"

    redner_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bundestag_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    vorname: Mapped[str] = mapped_column(String(128), nullable=False)
    nachname: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    titel: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    fraktion: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    partei: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    reden: Mapped[List["Rede"]] = relationship(
        "Rede", back_populates="redner", cascade="all, delete-orphan"
    )

    @property
    def vollname(self) -> str:
        """Return the full display name including optional title."""
        parts = []
        if self.titel:
            parts.append(self.titel)
        parts.append(self.vorname)
        parts.append(self.nachname)
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"<Redner {self.vollname} ({self.fraktion})>"


# ─────────────────────────────────────────────────────────────────────────────
# Rede
# ─────────────────────────────────────────────────────────────────────────────

class Rede(Base):
    """Represents a single speech during a plenary session."""

    __tablename__ = "reden"

    rede_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bundestag_rede_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    sitzung_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sitzungen.sitzungs_id"), nullable=False, index=True
    )
    redner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("redner.redner_id"), nullable=False, index=True
    )
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tagesordnungspunkt: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    wortanzahl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- NLP Felder ---
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ton_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tone_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    adressaten: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    sitzung: Mapped["Sitzung"] = relationship("Sitzung", back_populates="reden")
    redner: Mapped["Redner"] = relationship("Redner", back_populates="reden")
    zwischenrufe: Mapped[List["Zwischenruf"]] = relationship(
        "Zwischenruf", back_populates="rede", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Rede id={self.rede_id} redner_id={self.redner_id} "
            f"wörter={self.wortanzahl}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Zwischenruf
# ─────────────────────────────────────────────────────────────────────────────

class Zwischenruf(Base):
    """Represents a verbal or non-verbal interjection during a speech."""

    __tablename__ = "zwischenrufe"

    ruf_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rede_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reden.rede_id"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Faction that produced the interjection (extracted from the raw text).
    fraktion: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    # Sentiment score assigned by the NLP engine: -1.0 (very negative) to +1.0 (positive).
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Human-readable category: "Beifall", "Lachen", "Widerspruch", "Zwischenruf", "Unruhe", "Zustimmung", "Sonstiges".
    kategorie: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Rhetorical tone label assigned by ToneClassifier: "Aggression", "Sarkasmus", "Humor", "Neutral".
    ton_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Detailed probabilities for each tone.
    tone_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Comma-separated list of detected addressees (factions / persons) from AddresseeDetector.
    adressaten: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    rede: Mapped["Rede"] = relationship("Rede", back_populates="zwischenrufe")

    def __repr__(self) -> str:
        return (
            f"<Zwischenruf id={self.ruf_id} fraktion={self.fraktion} "
            f"score={self.sentiment_score}>"
        )
