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
NLP engine for OpenParlament.
Provides Sentiment analysis, Tone classification, and Addressee detection.
"""

from __future__ import annotations

import logging
import re
import sys
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/Torch nicht gefunden. Die neuralen Funktionen werden fehlschlagen.")

# Standard models
_DEFAULT_SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
_DEFAULT_TONE_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
_DEFAULT_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"

_LABEL_TO_SCORE = {
    "positive": 1.0, "neutral": 0.0, "negative": -1.0,
    "Positive": 1.0, "Neutral": 0.0, "Negative": -1.0,
}

# ---------------------------------------------------------------------------
# Zentrale Normalisierung & Patterns (Fraktionen)
# ---------------------------------------------------------------------------

_FRAKTION_CLEANUP_MAP = {
    "LINKE": "Die Linke", "DIE LINKE": "Die Linke", "LINKEN": "Die Linke",
    "GRÜNEN": "Bündnis 90/Die Grünen", "GRÜNE": "Bündnis 90/Die Grünen",
    "BÜNDNIS 90/DIE GRÜNEN": "Bündnis 90/Die Grünen", "BÜNDNIS 90": "Bündnis 90/Die Grünen",
    "CDU": "CDU/CSU", "CSU": "CDU/CSU", "CDU/CSU": "CDU/CSU", "UNION": "CDU/CSU",
    "SPD": "SPD", "SPDSP": "SPD", "SPDSPD": "SPD",
    "FDP": "FDP", "AFD": "AfD", "BSW": "BSW", "SSW": "SSW",
    "PDS": "Die Linke", "PDS/LINKE LISTE": "Die Linke",
    "FRAKTIONSLOS": "fraktionslos"
}

_VARIANTS_SORTED = sorted(_FRAKTION_CLEANUP_MAP.keys(), key=len, reverse=True)
_FRAKTION_REGEX_PART = "|".join(re.escape(f) for f in _VARIANTS_SORTED)

_FRAKTION_CONTEXT_PATTERN = re.compile(
    r"(?:bei(?:\s+Abgeordneten)?(?:\s+der)?|von der|der|des)\s+"
    r"(?:Abgeordneten?\s+)?"
    r"(" + _FRAKTION_REGEX_PART + r")", re.IGNORECASE,
)

_FRAKTION_DIRECT_PATTERN = re.compile(
    r"\b(" + _FRAKTION_REGEX_PART + r")\b", re.IGNORECASE,
)

def _canonicalise_fraktion(name: str) -> str:
    return _FRAKTION_CLEANUP_MAP.get(name.upper(), name) if name else name

# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------

_POSITIVE_KEYWORDS = frozenset(["beifall", "zustimmung", "sehr gut", "bravo", "richtig", "gut so", "heiterkeit", "lachen"])
_NEGATIVE_KEYWORDS = frozenset(["widerspruch", "nein", "pfui", "buh", "unruhe", "tumult", "entrüstung", "empörung", "schämen"])

def _rule_based_sentiment_score(text: str) -> Optional[float]:
    t = text.lower()
    if any(kw in t for kw in _POSITIVE_KEYWORDS): return 0.8
    if any(kw in t for kw in _NEGATIVE_KEYWORDS): return -0.8
    return None

class SentimentEngine:
    def __init__(self, model_name: str = _DEFAULT_SENTIMENT_MODEL, batch_size: int = 32, device: int = -1, torch_dtype: Any = None, **kwargs):
        # Accept extra kwargs for forward compatibility (NLPSession may pass torch_dtype, fp16, etc.)
        if kwargs:
            logger.debug("SentimentEngine.__init__ ignoring extra kwargs: %s", kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipe = None

    def score_one(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts: List[str]) -> List[float]:
        scores: List[Optional[float]] = [_rule_based_sentiment_score(t) for t in texts]
        unresolved_indices = [i for i, s in enumerate(scores) if s is None]

        if unresolved_indices:
            neural = self._neural_score_batch([texts[i] for i in unresolved_indices])
            for idx, s in zip(unresolved_indices, neural):
                scores[idx] = s
        return [s if s is not None else 0.0 for s in scores]

    def _get_pipeline(self):
        if self._pipe is None:
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers/Torch required.")
            logger.info("Lade Sentiment-Modell '%s'...", self.model_name)
            kwargs: Dict[str, Any] = dict(truncation=True, max_length=128)
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            self._pipe = pipeline("sentiment-analysis", model=self.model_name, device=self.device, **kwargs)
        return self._pipe

    def _neural_score_batch(self, texts: List[str]) -> List[float]:
        try:
            stream = tqdm(
                self._get_pipeline()(texts, batch_size=self.batch_size),
                total=len(texts),
                desc="Sentiment",
                leave=False,
                disable=not (sys.stderr is not None and getattr(sys.stderr, "isatty", lambda: False)()),
            )
            return [_LABEL_TO_SCORE.get(r["label"], 0.0) * r["score"] for r in stream]
        except Exception as exc:
            logger.error("Sentiment-Analyse fehlgeschlagen: %s", exc)
            return [0.0] * len(texts)

# ---------------------------------------------------------------------------
# ToneClassifier
# ---------------------------------------------------------------------------

_TONE_LABELS = ["Aggression", "Sarkasmus", "Humor", "Neutral"]
_TONE_RULES: List[Tuple[str, str]] = [
    ("beifall", "Neutral"), ("heiterkeit", "Humor"), ("lachen", "Humor"), ("witz", "Humor"),
    ("ironie", "Sarkasmus"), ("sarkasmus", "Sarkasmus"), ("widerspruch", "Aggression"),
    ("empörung", "Aggression"), ("entrüstung", "Aggression"), ("unruhe", "Aggression"),
    ("tumult", "Aggression"), ("schämen", "Aggression"), ("pfui", "Aggression"),
]

def _tone_rule_based(text: str) -> Optional[Tuple[str, Dict[str, float]]]:
    t = text.lower()
    for kw, label in _TONE_RULES:
        if kw in t:
            scores = {lbl: (0.85 if lbl == label else 0.05) for lbl in _TONE_LABELS}
            return label, scores
    return None

class ToneClassifier:
    def __init__(self, model_name: str = _DEFAULT_TONE_MODEL, device: int = -1, torch_dtype: Any = None, batch_size: int = 32, **kwargs):
        if kwargs:
            logger.debug("ToneClassifier.__init__ ignoring extra kwargs: %s", kwargs)
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self._pipe = None

    def classify(self, text: str) -> Tuple[str, Dict[str, float]]:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        results: List[Optional[Tuple[str, Dict[str, float]]]] = [_tone_rule_based(t) for t in texts]
        unresolved = [i for i, r in enumerate(results) if r is None]

        if unresolved:
            neural = self._neural_classify_batch([texts[i] for i in unresolved])
            for idx, res in zip(unresolved, neural):
                results[idx] = res
        return [r if r is not None else ("Neutral", {lbl: 0.0 for lbl in _TONE_LABELS}) for r in results]

    def _get_pipeline(self):
        if self._pipe is None:
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers/Torch required.")
            logger.info("Lade Tone-Modell '%s'...", self.model_name)
            kwargs: Dict[str, Any] = dict(truncation=True, max_length=128)
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            self._pipe = pipeline("zero-shot-classification", model=self.model_name, device=self.device, **kwargs)
        return self._pipe

    def _neural_classify_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        try:
            stream = tqdm(
                self._get_pipeline()(texts, candidate_labels=_TONE_LABELS, batch_size=self.batch_size),
                total=len(texts),
                desc="Tone",
                leave=False,
                disable=not (sys.stderr is not None and getattr(sys.stderr, "isatty", lambda: False)()),
            )
            return [(item["labels"][0], dict(zip(item["labels"], item["scores"]))) for item in stream]
        except Exception as exc:
            logger.error("Tone-Analyse fehlgeschlagen: %s", exc)
            return [("Neutral", {lbl: 0.0 for lbl in _TONE_LABELS}) for _ in texts]

# ---------------------------------------------------------------------------
# AddresseeDetector
# ---------------------------------------------------------------------------
# Module-level NER pipeline cache (shared across all AddresseeDetector instances).
# Protected by a double-checked lock so the pipeline is loaded only once even
# under concurrent access.  _NER_FAILED prevents repeated load attempts after
# a permanent failure (e.g. transformers/torch not installed).

_NER_PIPELINE = None
_NER_LOCK = Lock()
_NER_FAILED = False  # If True, skip all future attempts to load NER.


class AddresseeDetector:
    def __init__(
        self,
        use_ner: bool = False,
        ner_model_name: str = _DEFAULT_NER_MODEL,
        device: int = -1,
        torch_dtype: Any = None,
        batch_size: int = 32,
        **kwargs,
    ):
        if kwargs:
            logger.debug("AddresseeDetector.__init__ ignoring extra kwargs: %s", kwargs)

        self.use_ner = use_ner
        self.ner_model_name = ner_model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size

    def detect(self, text: str) -> List[str]:
        found, seen = [], set()

        # 1. & 2. Fraktionen
        for pattern in (_FRAKTION_CONTEXT_PATTERN, _FRAKTION_DIRECT_PATTERN):
            if found and pattern == _FRAKTION_DIRECT_PATTERN:
                break
            for match in pattern.finditer(text):
                canonical = _canonicalise_fraktion(match.group(1))
                if canonical not in seen:
                    seen.add(canonical)
                    found.append(canonical)

        # 3. Personen (NER)
        if self.use_ner:
            for person in self._ner_detect_persons(text):
                if person not in seen:
                    seen.add(person)
                    found.append(person)

        return found

    def detect_batch(self, texts: List[str]) -> List[List[str]]:
        if not texts:
            return []
        if not self.use_ner:
            return [self.detect(t) for t in texts]

        # Run NER over the entire batch, then merge with rule-based results per text.
        ner_batched = self._neural_detect_batch(texts)
        results = []
        for text, ents in zip(texts, ner_batched):
            found, seen = [], set()

            # Fraktionen
            for pattern in (_FRAKTION_CONTEXT_PATTERN, _FRAKTION_DIRECT_PATTERN):
                if found and pattern == _FRAKTION_DIRECT_PATTERN:
                    break
                for match in pattern.finditer(text):
                    canonical = _canonicalise_fraktion(match.group(1))
                    if canonical not in seen:
                        seen.add(canonical)
                        found.append(canonical)

            # Personen
            for e in ents:
                if e.get("entity_group") in ("PER", "PERSON"):
                    person = e["word"].strip()
                    if person not in seen:
                        seen.add(person)
                        found.append(person)

            results.append(found)

        return results

    def _neural_detect_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        try:
            stream = tqdm(
                self._get_ner_pipeline()(texts, batch_size=self.batch_size),
                total=len(texts),
                desc="NER",
                leave=False,
                disable=not (sys.stderr is not None and getattr(sys.stderr, "isatty", lambda: False)()),
            )
            return list(stream)
        except Exception as exc:
            logger.error("NER-Analyse fehlgeschlagen: %s", exc)
            return [[] for _ in texts]

    def _get_ner_pipeline(self):
        global _NER_PIPELINE, _NER_FAILED

        with _NER_LOCK:
            if _NER_FAILED:
                raise RuntimeError("NER pipeline already failed – skipping reload")

            if _NER_PIPELINE is None:
                try:
                    if not _TRANSFORMERS_AVAILABLE:
                        raise RuntimeError("Transformers/Torch required.")

                    logger.info("Lade NER-Modell '%s'...", self.ner_model_name)

                    kwargs: Dict[str, Any] = dict(
                        aggregation_strategy="simple",
                    )
                    if self.torch_dtype is not None:
                        kwargs["torch_dtype"] = self.torch_dtype

                    _NER_PIPELINE = pipeline(
                        "ner",
                        model=self.ner_model_name,
                        device=self.device,
                        **kwargs,
                    )

                except Exception as e:
                    _NER_FAILED = True
                    logger.exception("NER pipeline failed permanently")
                    raise

        return _NER_PIPELINE

    def _ner_detect_persons(self, text: str) -> List[str]:
        try:
            pipe = self._get_ner_pipeline()
            ents = pipe(text)

            return [
                e["word"].strip()
                for e in ents
                if e.get("entity_group") in ("PER", "PERSON")
            ]

        except Exception:
            logger.exception("NER inference failed")
            return []