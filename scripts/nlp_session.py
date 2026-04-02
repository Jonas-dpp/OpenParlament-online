#!/usr/bin/env python3
"""
scripts/nlp_session.py

NLPSession helper

Prompts for CUDA device when requested (interactive).
Initializes SentimentEngine, ToneClassifier, AddresseeDetector with the chosen device.
Acts as a simple context manager for the NLP engine lifetime.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# Import project NLP classes lazily (they import transformers if available)
from src.nlp import SentimentEngine, ToneClassifier, AddresseeDetector


def _choose_cuda_device_interactive(max_devices: int, audio_on: bool) -> int:
    from src import ringtones
    prompt = (
        f"{max_devices} CUDA device(s) detected. Choose device index [0-{max_devices - 1}] "
        "(enter for 0): "
    )
    try:
        if audio_on:
            ringtones.play_in_background(ringtones.alert_input_required, enabled=True)
        resp = input(prompt).strip()
        if resp == "":
            return 0
        idx = int(resp)
        if 0 <= idx < max_devices:
            return idx
    except Exception:
        pass
    logger.info("Invalid choice, defaulting to device 0")
    return 0


class NLPSession:
    """
    NLPSession(initializes Sentiment/Tone/Addressee engines)

    Parameters
    ----------
    use_cuda : bool
        Request CUDA (if available).
    device_index : int, optional
        Explicit device index (overrides interactive prompt).
    device : int, optional
        Direct device override; bypasses CUDA detection when set.
    batch_size : int
        Passed to SentimentEngine.
    use_ner : bool
        Passed to AddresseeDetector.
    fp16 : bool
        Load neural models in half-precision (float16). Only applied when the
        resolved device is a GPU (device >= 0). Requires torch.
    """

    def __init__(
        self,
        *,
        use_cuda: bool = False,
        device_index: Optional[int] = None,
        device: Optional[int] = None,
        batch_size: int = 100,
        use_ner: bool = False,
        fp16: bool = False,
        audio_on: bool = False,
    ):
        self.use_cuda = use_cuda
        self.device_index = device_index
        self._explicit_device = device  # direct override; bypasses CUDA logic when set
        self.batch_size = batch_size
        self.use_ner = use_ner
        self.fp16 = fp16
        self.audio_on = audio_on

        self.device: int = -1  # -1 = CPU, >=0 GPU index
        self.sentiment_engine: Optional[SentimentEngine] = None
        self.tone_classifier: Optional[ToneClassifier] = None
        self.addressee_detector: Optional[AddresseeDetector] = None

    def _resolve_device(self) -> int:
        # An explicit device override bypasses all CUDA detection logic.
        if self._explicit_device is not None:
            return self._explicit_device

        if not self.use_cuda:
            return -1

        if not _TORCH_AVAILABLE:
            logger.warning("torch not available: falling back to CPU")
            return -1

        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available: falling back to CPU")
                return -1
            count = torch.cuda.device_count()
            if count == 0:
                logger.warning("No CUDA devices found: falling back to CPU")
                return -1

            if self.device_index is None:
                # interactive prompt if terminal available
                if sys.stdin is not None and sys.stdin.isatty():
                    idx = _choose_cuda_device_interactive(count, self.audio_on)
                else:
                    idx = 0
                    logger.info("No device index specified and non-interactive; using device 0")
            else:
                idx = int(self.device_index)
                if idx < 0 or idx >= count:
                    logger.warning(
                        "Requested device index %s out of range [0-%d); using 0", idx, count
                    )
                    idx = 0
            logger.info("Using CUDA device %d", idx)
            return idx
        except Exception as exc:
            logger.warning("Error while resolving CUDA device: %s. Falling back to CPU.", exc)
            return -1

    def __enter__(self) -> "NLPSession":
        # Decide device
        self.device = self._resolve_device()

        # Determine torch dtype for fp16 support (GPU only).
        torch_dtype = None
        if self.fp16 and self.device >= 0:
            if _TORCH_AVAILABLE:
                torch_dtype = torch.float16
                logger.info("fp16 enabled: loading models with torch.float16")
            else:
                logger.warning("--fp16 requested but torch is not available; ignoring.")

        # Create engines. The project's classes accept device: int (-1 CPU, >=0 GPU)
        try:
            self.sentiment_engine = SentimentEngine(batch_size=self.batch_size, device=self.device, torch_dtype=torch_dtype)
        except Exception as exc:
            logger.error("Failed to initialize SentimentEngine: %s", exc)
            self.sentiment_engine = None

        try:
            self.tone_classifier = ToneClassifier(device=self.device, torch_dtype=torch_dtype, batch_size=self.batch_size)
        except Exception as exc:
            logger.error("Failed to initialize ToneClassifier: %s", exc)
            self.tone_classifier = None

        try:
            self.addressee_detector = AddresseeDetector(use_ner=self.use_ner, device=self.device, torch_dtype=torch_dtype, batch_size=self.batch_size)
        except Exception as exc:
            logger.error("Failed to initialize AddresseeDetector: %s", exc)
            self.addressee_detector = None

        logger.info(
            "NLPSession ready. Device=%s",
            "cpu" if self.device < 0 else f"cuda:{self.device}",
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        # No explicit teardown required for transformers pipelines; keep placeholder.
        # In future we could .close() or free GPU memory here.
        self.sentiment_engine = None
        self.tone_classifier = None
        self.addressee_detector = None
        logger.info("NLPSession closed")
        return False
