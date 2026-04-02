"""
run_scraper.py – Download Bundestag plenary protocols and import them into the DB.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_session, init_db
from src.models import Redner, Rede, Sitzung, Zwischenruf
from src.parser import BundestagXMLParser
from src.scraper import BundestagScraper
from src import ringtones
from sqlalchemy import select, update

# NLP helpers shared with run_nlp_cli.py
from scripts.run_nlp_cli import is_column_attr, _pk_attr, _serialize_for_column
from scripts.nlp_session import NLPSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Importer
# ─────────────────────────────────────────────────────────────────────────────

class ProtocolImporter:
    def __init__(self) -> None:
        self._parser = BundestagXMLParser()
        self._redner_cache: dict[str, Redner] = {}

    @staticmethod
    def _redner_key(r) -> str | None:
        if r.bundestag_id:
            return str(r.bundestag_id)
        return r.vollname.lower().strip()

    def _ensure_redner_cache(self, session):
        """Populate cache once per run to prevent N+1 queries."""
        if not self._redner_cache:
            rows = session.execute(
                select(Redner).where(Redner.bundestag_id.is_not(None))
            ).scalars().all()
            self._redner_cache = {str(r.bundestag_id): r for r in rows}

    def import_file(self, path: Path) -> bool:
        fname = path.stem
        result = None
        wp, snr = None, None

        # 1. Determine IDs
        if fname.isdigit() and len(fname) >= 3:
            wp, snr = int(fname[:2]), int(fname[2:])
        else:
            result = None

        with get_session() as session:
            try:
                # 1. Existenz-Check (Sitzung)
                existing = session.execute(
                    select(Sitzung).where(Sitzung.wahlperiode == wp, Sitzung.sitzungsnr == snr)
                ).scalar_one_or_none()

                if existing:
                    logger.info("Skipping already-imported Sitzung %d/%d.", wp, snr)
                    return False

                # 2. Parsen (Hier passiert die Magie durch dein Parser-Update!)
                # Der Parser nutzt jetzt session.get(), um Duplikate in der Session zu vermeiden.
                if result is None:
                    result = self._parser.parse_file(path, session=session)
                    wp, snr = result.sitzung.wahlperiode, result.sitzung.sitzungsnr

                # 3. Cache & Session synchronisieren
                with session.no_autoflush:
                    self._ensure_redner_cache(session)
                
                redner_map: dict[str, Redner] = {}
                for r in result.redner:
                    key = self._redner_key(r)
                    if not key: continue
                    
                    # WICHTIG: Prüfen, ob SQLAlchemy das Objekt schon kennt (durch den Parser)
                    # oder ob es in unserem Importer-Cache liegt.
                    db_redner = self._redner_cache.get(key)
                    
                    if db_redner:
                        redner_map[key] = db_redner
                    else:
                        # Wenn der Redner neu ist, fügen wir ihn dem Cache hinzu
                        # Er wurde vom Parser bereits mit session.add() markiert
                        redner_map[key] = r
                        if r.bundestag_id:
                            self._redner_cache[str(r.bundestag_id)] = r

                # 4. Persistence (Sitzung & Reden verknüpfen)
                session.add(result.sitzung)
                session.flush() # ID für Sitzung generieren

                for rede in result.reden:
                    key = self._redner_key(rede.redner)
                    # Nutze den Redner aus unserer Map (verhindert doppelte Instanzen)
                    rede.redner = redner_map.get(key, rede.redner)
                    rede.sitzung = result.sitzung
                    session.add(rede)

                for zwr in result.zwischenrufe:
                    session.add(zwr)

                session.commit()
                logger.info("Imported Sitzung %d/%d.", wp, snr)
                return True
            except Exception as e:
                session.rollback()
                logger.error(
                    "Import failed for %s/%s (%s): %s\n%s",
                    wp, snr, path.name, e, traceback.format_exc(),
                )
                raise

# ─────────────────────────────────────────────────────────────────────────────
# NLP scoring
# ─────────────────────────────────────────────────────────────────────────────

def run_nlp_scoring(
    batch_size: int = 100,
    *,
    use_cuda: bool = False,
    device_index: int | None = None,
    use_ner: bool = False,
    fp16: bool = False,
    commit_interval: int = 10,
) -> None:
    """Run sentiment, tone, and addressee scoring on all unscored Zwischenrufe.

    Uses :class:`NLPSession` so that all three NLP engines are initialised in
    the same way as in ``run_nlp_cli.py`` (fp16 support, CUDA device selection,
    lazy transformers import).  Results are written back via a single bulk
    UPDATE per ORM class per *commit_interval* batches (or on the final batch)
    rather than after every single batch, reducing DB round-trips.

    :param commit_interval: Number of batches to accumulate before issuing a
        ``COMMIT``.  Lower values are safer (less re-work on failure) but
        produce more DB round-trips.  Default: 10.
    """
    if commit_interval < 1:
        raise ValueError(f"commit_interval must be >= 1, got {commit_interval}")
    from tqdm import tqdm

    with NLPSession(
        use_cuda=use_cuda,
        device_index=device_index,
        batch_size=batch_size,
        use_ner=use_ner,
        fp16=fp16,
    ) as agent:
        sentiment_engine = agent.sentiment_engine
        tone_classifier = agent.tone_classifier
        addressee_detector = agent.addressee_detector

        with get_session() as session:
            rows = session.execute(
                select(Zwischenruf).where(Zwischenruf.sentiment_score.is_(None))
            ).scalars().all()

            if not rows:
                logger.info("All interjections already scored.")
                return

            num_batches = (len(rows) + batch_size - 1) // batch_size
            # Accumulated bulk-update mappings across batches within the current
            # commit window, keyed by ORM class.
            pending_rows: Dict[type, List[dict]] = defaultdict(list)
            # Track the range of rows covered by the pending window for logging.
            pending_start: int = 1

            batch_iter = tqdm(
                range(0, len(rows), batch_size),
                total=num_batches,
                desc="NLP Scoring",
                unit="batch",
                mininterval=0.5,
            )
            for batch_num, batch_start in enumerate(batch_iter, 1):
                batch_rows: List[Zwischenruf] = rows[batch_start:batch_start + batch_size]
                texts = [z.text or "" for z in batch_rows]

                sentiments = []
                if sentiment_engine:
                    try:
                        sentiments = sentiment_engine.score_batch(texts)
                    except Exception as exc:
                        logger.error("Sentiment engine failed: %s", exc)
                        sentiments = [0.0] * len(batch_rows)

                tones = []
                if tone_classifier:
                    try:
                        tones = tone_classifier.classify_batch(texts)
                    except Exception as exc:
                        logger.error("Tone classifier failed: %s", exc)
                        tones = [("Neutral", {}) for _ in batch_rows]

                addressees = []
                if addressee_detector:
                    try:
                        addressees = addressee_detector.detect_batch(texts)
                    except Exception as exc:
                        logger.error("Addressee detection failed: %s", exc)
                        addressees = [[] for _ in batch_rows]

                # Accumulate bulk-update mappings for this batch.
                for i, obj in enumerate(batch_rows):
                    cls = obj.__class__
                    pk_name = _pk_attr(cls)
                    row_map: dict = {pk_name: getattr(obj, pk_name)}

                    if sentiments and is_column_attr(cls, "sentiment_score"):
                        row_map["sentiment_score"] = float(sentiments[i])

                    if tones and is_column_attr(cls, "ton_label"):
                        label, scores = tones[i]
                        row_map["ton_label"] = label
                        if is_column_attr(cls, "ton_scores"):
                            row_map["ton_scores"] = _serialize_for_column(cls, "ton_scores", scores)

                    if addressees and is_column_attr(cls, "adressaten"):
                        row_map["adressaten"] = _serialize_for_column(cls, "adressaten", addressees[i])

                    if len(row_map) > 1:
                        pending_rows[cls].append(row_map)

                is_last_batch = batch_num == num_batches
                should_commit = (batch_num % commit_interval == 0) or is_last_batch

                if should_commit and pending_rows:
                    try:
                        for cls, mappings in pending_rows.items():
                            session.execute(
                                update(cls),
                                mappings,
                                execution_options={"synchronize_session": False},
                            )
                        session.commit()
                        session.expire_all()
                        logger.info(
                            "Committed rows %d–%d (batches %d–%d).",
                            pending_start,
                            batch_start + len(batch_rows),
                            (pending_start - 1) // batch_size + 1,
                            batch_num,
                        )
                        pending_rows.clear()
                        pending_start = batch_start + len(batch_rows) + 1
                    except Exception as exc:
                        session.rollback()
                        logger.error(
                            "Failed to commit NLP rows %d–%d: %s  "
                            "— You can safely re-run this command; already committed rows will not be affected.",
                            pending_start,
                            batch_start + len(batch_rows),
                            exc,
                        )
                        raise

# ─────────────────────────────────────────────────────────────────────────────
# CLI / Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenParlament Scraper")
    p.add_argument("--wahlperiode", type=int, default=20)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--redownload", action="store_true", 
                   help="Re-download files even if they already exist locally.")
    
    # NLP Arguments
    p.add_argument("--nlp", action="store_true",
                   help="Run NLP scoring (sentiment, tone, addressee) after import.")
    p.add_argument("--nlp-batch-size", type=int, default=100,
                   help="Batch size for NLP scoring.")
    p.add_argument("--nlp-cuda", action="store_true",
                   help="Use CUDA for NLP models.")
    p.add_argument("--nlp-device-index", type=int, default=None,
                   help="Explicit CUDA device index.")
    p.add_argument("--nlp-use-ner", action="store_true",
                   help="Enable NER in AddresseeDetector.")
    p.add_argument("--nlp-fp16", action="store_true",
                   help="Load NLP models in half-precision (GPU only).")
    p.add_argument("--commit-interval", type=int, default=10,
                   metavar="N",
                   help="Commit NLP scoring results to DB every N batches (default: 10). Must be >= 1.")
    p.add_argument("--make-noise", action="store_true",
                   help="Enable audible ringtones for hearable notifications.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    AUDIO_ON = args.make_noise

    if args.commit_interval < 1:
        logger.error("--commit-interval must be >= 1, got %d", args.commit_interval)
        sys.exit(1)

    if AUDIO_ON:
        ringtones.setup_audio_logger(enabled=True)

    init_db()

    download_dir = _PROJECT_ROOT / "data" / "xml"
    scraper = BundestagScraper(
        download_dir=download_dir,
        request_delay=args.delay,
        skip_existing=not args.redownload,
    )
    importer = ProtocolImporter()

    with ringtones.monitor_process(enabled=AUDIO_ON):
        urls = scraper.fetch_protocol_urls(wahlperiode=args.wahlperiode, max_pages=args.max_pages)
        paths = scraper.download_all(urls)

        successes = 0
        failures = 0
        for p in paths:
            try:
                if importer.import_file(p): successes += 1
            except Exception as exc:
                failures += 1
                logger.error("Failed to import protocol file %s: %s", p, exc)

        if successes > 0:
            # Non-blocking: plays in background so the loop/progress doesn't stall.
            ringtones.play_in_background(ringtones.alert_advancement, enabled=AUDIO_ON)
        if failures > 0:
            logger.warning("Imported %d new session(s); %d file(s) failed.", successes, failures)
        else:
            logger.info("Imported %d new session(s).", successes)

        if args.nlp:
            # Signal that the next heavy phase (NLP scoring) is starting.
            ringtones.play_in_background(ringtones.alert_advancement, enabled=AUDIO_ON)
            run_nlp_scoring(
                batch_size=args.nlp_batch_size,
                use_cuda=args.nlp_cuda,
                device_index=args.nlp_device_index,
                use_ner=args.nlp_use_ner,
                fp16=args.nlp_fp16,
                commit_interval=args.commit_interval,
            )
            
    if AUDIO_ON:
        ringtones.play_in_background(ringtones.alert_finish, enabled=True)

if __name__ == "__main__":
    main()