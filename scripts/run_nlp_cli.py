#!/usr/bin/env python3
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
scripts/run_nlp_cli.py

CLI for running NLP scoring on existing data.

Uses NLPSession (scripts/nlp_session.py) which can prompt for CUDA usage
and initializes the NLP engines with the chosen device.

Features:
- target: zwischenrufe | reden | all
- batch-size, device, use-ner
- --cuda flag with optional --device-index (interactive GPU prompt when tty)
- dry-run (don't write) or persist into DB
- configurable target field names for persistence (defaults match existing project)
- limit for quick tests
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm import tqdm

# Project root import helper (so the script can be run from repo root)
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_session
from src.models import Zwischenruf, Rede
from src.nlp import _TONE_LABELS
from sqlalchemy import inspect as sa_inspect, select, update
from sqlalchemy.types import JSON

from scripts.nlp_session import NLPSession  # new helper
from src import ringtones

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_nlp_cli")


def chunked(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def is_column_attr(cls, name: str) -> bool:
    """Return True only when *name* is an actual mapped column attribute on *cls*.

    Uses SQLAlchemy mapper inspection rather than ``hasattr``, which would also
    match relationships, hybrid properties, and other non-column attributes.
    """
    return name in sa_inspect(cls).mapper.column_attrs.keys()


def _pk_attr(cls) -> str:
    """Return the single primary-key attribute name for a SQLAlchemy mapped class.

    Raises ``ValueError`` when the class has a composite primary key, because
    the bulk-update logic requires each mapping dict to contain exactly one PK
    value to identify the row.
    """
    pks = sa_inspect(cls).mapper.primary_key
    if len(pks) != 1:
        raise ValueError(
            f"{cls.__name__} has a composite primary key ({[c.key for c in pks]}); "
            "bulk UPDATE via a single PK dict is not supported."
        )
    return pks[0].key


def _serialize_for_column(cls, field: str, value, *, is_json: bool | None = None) -> object:
    """Serialize *value* for storage in *field* of *cls*.

    - If the column type is JSON/JSONB the value is stored as-is (the dialect
      handles serialization).
    - Otherwise (TEXT, VARCHAR, …) the value is serialized to a JSON string via
      ``json.dumps`` so it can be round-tripped reliably.  ``str()`` is *not*
      used because it produces Python repr syntax (e.g. single-quoted strings)
      which is not valid JSON.

    Pass ``is_json=True/False`` when the column type is already known (e.g. from
    a pre-computed ``_cls_is_json`` dict) to avoid the ``sa_inspect`` overhead.
    """
    if is_json is None:
        col = sa_inspect(cls).mapper.column_attrs[field].columns[0]
        is_json = isinstance(col.type, JSON)
    if is_json:
        return value
    return json.dumps(value, ensure_ascii=False)


def _is_json_column(cls, field: str) -> bool:
    """Return True when *field* on *cls* is a JSON/JSONB column type."""
    col = sa_inspect(cls).mapper.column_attrs[field].columns[0]
    return isinstance(col.type, JSON)


def gather_targets(session, target: str, limit: int | None, only_unscored: bool, sentiment_field: str):
    rows = []
    if target in ("zwischenrufe", "all"):
        q = select(Zwischenruf)
        if only_unscored and is_column_attr(Zwischenruf, sentiment_field):
            q = q.where(getattr(Zwischenruf, sentiment_field).is_(None))
        if limit:
            q = q.limit(limit)
        for r in session.execute(q).scalars():
            rows.append(("zwischenruf", r))
    if target in ("reden", "all"):
        q = select(Rede)
        if only_unscored and is_column_attr(Rede, sentiment_field):
            q = q.where(getattr(Rede, sentiment_field).is_(None))
        if limit:
            q = q.limit(limit)
        for r in session.execute(q).scalars():
            rows.append(("rede", r))
    if limit:
        rows = rows[:limit]
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the NLP CLI.

    Exposed as a public function so tests can inspect argument defaults without
    parsing a real command-line (e.g. to assert that ``--tone-label-field``
    defaults to the correct model column name).
    """
    p = argparse.ArgumentParser(description="Run NLP (sentiment, tone, addressee) on existing DB rows.")
    p.add_argument("--target", choices=("zwischenrufe", "reden", "all"), default="zwischenrufe",
                   help="Which table(s) to process (default: zwischenrufe).")
    p.add_argument("--batch-size", type=int, default=100, help="Batch size for neural scoring (default: 100).")
    p.add_argument("--device", type=int, default=-1, help="Device for transformers pipelines (default -1 -> CPU). Ignored when --cuda is set.")
    p.add_argument("--cuda", action="store_true",
                   help="Attempt to use CUDA. If interactive and no --device-index provided you'll be prompted.")
    p.add_argument("--device-index", type=int, default=None,
                   help="Explicit CUDA device index when using --cuda.")
    p.add_argument("--use-ner", action="store_true", help="Enable NER in AddresseeDetector (requires transformers).")
    p.add_argument("--fp16", action="store_true",
                   help="Load neural models in half-precision (float16) for faster GPU inference. Ignored on CPU.")
    p.add_argument("--dry-run", action="store_true", help="Compute results but do not write to DB.")
    p.add_argument("--only-unscored", action="store_true", help="Only process rows where sentiment field is NULL/unset.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of rows (quick tests). 0 = no limit.")
    # persistence field names (match your models; defaults follow project's convention)
    p.add_argument("--sentiment-field", default="sentiment_score", help="Column name to write sentiment scores to.")
    p.add_argument("--tone-label-field", default="ton_label", help="Column name to write tone label to.")
    p.add_argument("--tone-scores-field", default="tone_scores", help="Column name to write tone scores to (dict/JSON).")
    p.add_argument("--addressee-field", default="adressaten", help="Column name to write addressee list to.")
    p.add_argument("--no-sentiment", action="store_true", help="Skip sentiment scoring.")
    p.add_argument("--no-tone", action="store_true", help="Skip tone classification.")
    p.add_argument("--no-addressee", action="store_true", help="Skip addressee detection.")
    p.add_argument("--make-noise", action="store_true",
                   help="Enable audible ringtones for hearable notifications.")
    return p


def main():
    args = build_parser().parse_args()
    AUDIO_ON = args.make_noise

    if args.no_sentiment and args.no_tone and args.no_addressee:
        logger.error("Nothing to do: all scoring types disabled (--no-sentiment --no-tone --no-addressee).")
        return

    if AUDIO_ON:
        ringtones.setup_audio_logger(enabled=True)

    limit = args.limit if args.limit and args.limit > 0 else None

    logger.info("Starting NLP CLI: target=%s batch=%d cuda=%s device_index=%s fp16=%s use_ner=%s dry_run=%s only_unscored=%s limit=%s",
                args.target, args.batch_size, args.cuda, args.device_index, args.fp16, args.use_ner, args.dry_run, args.only_unscored, limit)

    # Use NLPSession which handles CUDA prompting & engine initialization.
    # When --cuda is not set, pass --device directly so engines are created
    # with the correct device from the start (no post-init mutation needed).
    with ringtones.monitor_process(enabled=AUDIO_ON):
        with NLPSession(
            use_cuda=args.cuda,
            device_index=args.device_index,
            device=None if args.cuda else args.device,
            batch_size=args.batch_size,
            use_ner=args.use_ner,
            fp16=args.fp16,
            audio_on=AUDIO_ON,
        ) as agent:
            sentiment_engine = agent.sentiment_engine if not args.no_sentiment else None
            tone_classifier = agent.tone_classifier if not args.no_tone else None
            addressee_detector = agent.addressee_detector if not args.no_addressee else None

            with get_session() as session:
                rows = gather_targets(session, args.target, limit, args.only_unscored,
                                      args.sentiment_field)
                total = len(rows)
                if total == 0:
                    logger.info("No rows selected for processing.")
                    return

                logger.info("Processing %d rows in %d batches of up to %d", total, (total + args.batch_size - 1) // args.batch_size, args.batch_size)

                # Prepare lists per object type for easier batching by text
                obj_list: List[Tuple[str, Union[Zwischenruf, Rede]]] = rows

                logger.info("---Start of NLP---")

                # Pre-compute per-class metadata once to avoid repeated SQLAlchemy
                # introspection (sa_inspect) inside the hot per-row loop.
                _target_classes: List[type] = []
                if args.target in ("zwischenrufe", "all"):
                    _target_classes.append(Zwischenruf)
                if args.target in ("reden", "all"):
                    _target_classes.append(Rede)

                _write_fields = [f for f, active in [
                    (args.sentiment_field, sentiment_engine is not None),
                    (args.tone_label_field, tone_classifier is not None),
                    (args.tone_scores_field, tone_classifier is not None),
                    (args.addressee_field, addressee_detector is not None),
                ] if active]

                _cls_pk: Dict[type, str] = {cls: _pk_attr(cls) for cls in _target_classes}
                _cls_has: Dict[type, Dict[str, bool]] = {
                    cls: {f: is_column_attr(cls, f) for f in _write_fields}
                    for cls in _target_classes
                }
                _cls_is_json: Dict[type, Dict[str, bool]] = {
                    cls: {f: _is_json_column(cls, f) for f in _write_fields if is_column_attr(cls, f)}
                    for cls in _target_classes
                }

                # Warn once for any configured field that doesn't exist on a target class.
                for cls, field_map in _cls_has.items():
                    for field, exists in field_map.items():
                        if not exists:
                            logger.warning(
                                "Field '%s' does not exist on %s — scores for this field will be skipped.",
                                field, cls.__name__,
                            )

                total_batches = (total + args.batch_size - 1) // args.batch_size
                for batch_idx, batch in tqdm(
                    enumerate(chunked(obj_list, args.batch_size), start=1),
                    total=total_batches,
                    desc="NLP batches",
                    unit="batch",
                ):
                    # Non-blocking heartbeat every 100 batches so the user knows
                    # the process is still alive without freezing the progress bar.
                    if batch_idx % 100 == 0:
                        ringtones.play_in_background(ringtones.alert_heartbeat, enabled=AUDIO_ON)

                    texts = [getattr(x[1], "text", "") or "" for x in batch]

                    # Sentiment
                    sentiments = []
                    if sentiment_engine:
                        try:
                            sentiments = sentiment_engine.score_batch(texts)
                        except Exception as exc:
                            logger.error("Sentiment engine failed on batch %d: %s", batch_idx, exc)
                            sentiments = [0.0] * len(batch)

                    # Tone
                    tones: List[Tuple[str, dict]] = []
                    if tone_classifier:
                        try:
                            tones = tone_classifier.classify_batch(texts)
                        except Exception as exc:
                            logger.error("Tone classifier failed on batch %d: %s", batch_idx, exc)
                            tones = [("Neutral", {lbl: 0.0 for lbl in _TONE_LABELS}) for _ in batch]

                    # Addressees
                    addressees = []
                    if addressee_detector:
                        try:
                            addressees = addressee_detector.detect_batch(texts)
                        except Exception as exc:
                            logger.error("Addressee detection failed on batch %d: %s", batch_idx, exc)
                            addressees = [[] for _ in batch]

                    # Apply results — collect bulk-update mappings grouped by ORM class.
                    bulk_maps: Dict[type, List[dict]] = defaultdict(list)
                    for i, (obj_type, obj) in enumerate(batch):
                        cls = obj.__class__
                        pk_name = _cls_pk[cls]
                        row_map: dict = {pk_name: getattr(obj, pk_name)}
                        has = _cls_has[cls]
                        is_json = _cls_is_json[cls]

                        if sentiment_engine and sentiments:
                            if has.get(args.sentiment_field):
                                row_map[args.sentiment_field] = float(sentiments[i])

                        if tone_classifier and tones:
                            label, scores = tones[i]
                            if has.get(args.tone_label_field):
                                row_map[args.tone_label_field] = label
                            if has.get(args.tone_scores_field):
                                row_map[args.tone_scores_field] = _serialize_for_column(
                                    cls, args.tone_scores_field, scores,
                                    is_json=is_json.get(args.tone_scores_field),
                                )

                        if addressee_detector and addressees:
                            vals = addressees[i]
                            if has.get(args.addressee_field):
                                row_map[args.addressee_field] = _serialize_for_column(
                                    cls, args.addressee_field, vals,
                                    is_json=is_json.get(args.addressee_field),
                                )

                        has_updates = len(row_map) > 1  # row_map contains the PK plus at least one field to update
                        if has_updates:
                            bulk_maps[cls].append(row_map)

                    # Commit if not dry-run — one bulk UPDATE per class, one commit per batch.
                    # Skip commit entirely when there is nothing to write (avoids a no-op roundtrip).
                    if not args.dry_run:
                        if bulk_maps:
                            try:
                                for cls, mappings in bulk_maps.items():
                                    if mappings:
                                        session.execute(
                                            update(cls),
                                            mappings,
                                            execution_options={"synchronize_session": False},
                                        )
                                session.commit()

                                # Expire all ORM instances so any subsequent reads reflect the
                                # values written by the bulk UPDATE (which bypasses ORM tracking).
                                session.expire_all()

                                if batch_idx == 1: # log the first batch commit to confirm persistence is working, then every 1000 batches to avoid spamming the console
                                    logger.info("Committed first batch %d (%d items).", batch_idx, len(batch))
                                    # Non-blocking advancement tone: data is flowing into the DB.
                                    ringtones.play_in_background(ringtones.alert_advancement, enabled=AUDIO_ON)
                                if batch_idx % 1000 == 0: # log every 1000 batches to avoid spamming the console | MIGHT NEED TO DELETE THIS IF TOO SPAMMY/DOESN'T WORK WELL WITH OTHER LOGGING
                                    logger.info("Committed batch %d (%d items).", batch_idx, len(batch))
                            except Exception as exc:
                                session.rollback()
                                logger.error("Failed to commit batch %d: %s", batch_idx, exc)
                        else:
                            logger.debug("Batch %d: no mapped fields to update, skipping commit.", batch_idx)
                    else:
                        if batch_idx % 100 == 0: # log every 100 batches in dry-run mode to avoid spamming the console
                            logger.info("Dry-run: computed batch %d (%d items) — not committed.", batch_idx, len(batch))

        logger.info("NLP run finished. dry_run=%s", args.dry_run)
        if AUDIO_ON:
            ringtones.play_in_background(ringtones.alert_finish, enabled=True)

if __name__ == "__main__":
    main()