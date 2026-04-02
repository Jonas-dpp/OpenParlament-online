"""
Bundestag Open Data scraper for OpenParlament.

Downloads plenary protocol XML files from the official Bundestag open-data
portal and stores them locally.

Usage
─────
    scraper = BundestagScraper(download_dir=Path("data/xml"))
    urls = scraper.fetch_protocol_urls(wahlperiode=20)
    scraper.download_all(urls)

URL discovery strategy (two sources)
─────────────────────────────────────
1. **dserver (primary / complete)**
   ``dserver.bundestag.de`` serves XML files at a fully predictable path::

       https://dserver.bundestag.de/btp/{wp}/{wp}{sn:03d}.xml

   Because the URL contains no opaque blob-ID, we can probe every session
   number systematically.  Each candidate is probed with a streaming GET
   request (HEAD is avoided because some servers return 405 or incorrect
   status codes for HEAD).  Probing stops automatically after
   ``_MAX_CONSECUTIVE_MISSES`` consecutive non-200 responses.  This is the
   *only* reliable way to enumerate all sessions – the filterlist API cannot
   page past its internal TYPO3 offset limit.

2. **Filterlist API (secondary / seed)**
   ``bundestag.de/ajax/filterlist/…`` returns recently published protocols.
   It is kept as a supplementary source to catch any file not yet propagated
   to the dserver mirror, but pagination is guarded by a **stall detector**:
   if a page contains no new URLs (all already seen from the dserver pass)
   the loop terminates immediately instead of burning through pointless
   requests.

To enumerate multiple Wahlperioden at once use ``fetch_all_wahlperioden``::

    scraper = BundestagScraper(download_dir=Path("data/xml"))
    urls = scraper.fetch_all_wahlperioden([18, 19, 20])

The scraper is deliberately conservative: it respects HTTP rate limits
(configurable delay between requests) and skips files that have already been
downloaded (idempotent).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from lxml import etree

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://www.bundestag.de"

# The Bundestag open-data listing page (filter endpoint with XML protocol links).
# The page number is controlled by the `offset` query parameter.
# NOTE: No query parameters here – they are passed explicitly in _fetch_page()
# to avoid duplicate parameters in the final request URL.
#
# The path segment "866354-866354" is the CMS node ID of the Bundestag
# Plenarprotokolle open-data page.  It is stable across requests; only the
# known-good value "866354" is accepted by the server – shorter, longer, or
# otherwise different values return HTTP 400.
_OPENDATA_URL = (
    "https://www.bundestag.de/ajax/filterlist/de/services/opendata/"
    "866354-866354"
)

# Root XML element that identifies a genuine Bundestag plenary-protocol file.
# Any downloaded XML whose root tag differs from this value is rejected.
_EXPECTED_ROOT_TAG = "dbtplenarprotokoll"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "OpenParlament/1.1.0 (Automatisierte Analyse v. BT-Protokollen zur Untersuchung v. Kommunikation und Dynamiken; "
        "academic research)"
    ),
    "Accept-Language": "de-DE,de;q=0.9",
}

_XML_LINK_PATTERN = re.compile(r"\.xml$", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# dserver constants
# ─────────────────────────────────────────────────────────────────────────────

# dserver.bundestag.de hosts plenary protocol XMLs at a predictable path:
#   /btp/{wahlperiode}/{wahlperiode}{sitzungsnr:03d}.xml
# Unlike the blob-storage URLs, no opaque CMS ID is required.
_DSERVER_BASE = "https://dserver.bundestag.de/btp"

# Stop probing dserver after this many consecutive non-200 responses.  A run
# of misses at the high end of a Wahlperiode signals that no further sessions
# exist.  5 is sufficient for current/recent Wahlperioden; older Wahlperioden
# with known session-number gaps may benefit from a higher value (e.g. 10).
_MAX_CONSECUTIVE_MISSES = 5

# Number of times to retry a dserver probe on a 5xx (server-side) error
# before giving up and counting the attempt as a miss.
_MAX_RETRIES = 3

# Connection timeout (seconds) for dserver probe requests.  Probes only need
# the HTTP status line – the response body is never read – so 5 seconds is
# more than sufficient.  Downloads use a separate, longer timeout (30 s).
_PROBE_TIMEOUT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _url_filename(url: str) -> str:
    """Return the lowercase bare filename of a URL for deduplication.

    Both ``https://dserver.bundestag.de/btp/20/20001.xml`` and
    ``https://www.bundestag.de/resource/blob/XXX/20001.xml`` resolve to the
    same session; comparing bare filenames prevents the same session from
    appearing twice when it is discovered via both sources.

    Uses :func:`urllib.parse.urlparse` to correctly handle query strings
    (``?download=true``), URL fragments (``#section``), encoded characters,
    and other edge cases that chained ``.split()`` calls would mishandle.

    :param url: Absolute URL of an XML file.
    :return: Lowercase filename with no path, query-string, or fragment.
    """
    return Path(urlparse(url).path).name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# BundestagScraper
# ─────────────────────────────────────────────────────────────────────────────

class BundestagScraper:
    """Download plenary protocol XML files from the Bundestag open-data portal.

    Parameters
    ──────────
    download_dir : Path
        Directory where XML files are saved.  Created automatically.
    request_delay : float
        Seconds to wait between filterlist-API requests and between file
        downloads (be a good citizen).  Default: ``1.0``.
    probe_delay : float
        Seconds to wait between individual dserver probe requests.  This can
        be much shorter than *request_delay* because probes only read the HTTP
        status line (stream=True, body never transferred).  Default: ``0.05``
        (20 req/s), which is respectful without being prohibitively slow.
    skip_existing : bool
        When ``True`` (the default) files that already exist in *download_dir*
        are not re-downloaded.  Set to ``False`` to force re-downloading every
        file even if it was previously downloaded.
    session : requests.Session, optional
        Inject a custom session (useful for testing / mocking).
    """

    def __init__(
        self,
        download_dir: Path = Path("data/xml"),
        request_delay: float = 1.0,
        probe_delay: float = 0.05,
        skip_existing: bool = True,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.probe_delay = probe_delay
        self.skip_existing = skip_existing
        self._session = session or self._build_session()

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch_protocol_urls(self, wahlperiode: int, max_pages: int = 100) -> List[str]:
        """Return a deduplicated list of absolute XML download URLs for *wahlperiode*.

        Uses two complementary sources:

        1. **dserver** (primary) – probes the deterministic URL pattern on
           ``dserver.bundestag.de`` to enumerate *all* sessions regardless of
           any backend pagination limit.
        2. **Filterlist API** (secondary) – supplements with any recently
           published files not yet present on dserver.  Pagination stops as
           soon as a full page contains only already-seen URLs (stall
           detection), so it never burns requests beyond the server's window.

        Deduplication is **filename-based**: the same session number discovered
        via dserver (``…/btp/20/20001.xml``) and via the filterlist
        (``…/resource/blob/XXX/20001.xml``) shares the same bare filename
        ``20001.xml`` and will therefore be collected only once.

        :param wahlperiode: The Wahlperiode to scrape (e.g. 20).
        :param max_pages: Safety cap on filterlist pagination iterations.
        :return: Deduplicated list of absolute URLs pointing to XML files.
        """
        # seen_filenames tracks bare filenames (e.g. "20001.xml") so that a
        # session arriving from both dserver and the filterlist is stored once.
        seen_filenames: set = set()
        urls: List[str] = []

        # ── Primary: dserver (complete, deterministic) ───────────────────────
        logger.info(
            "Probing dserver for Wahlperiode %d XML files…", wahlperiode
        )
        for url in self._fetch_via_dserver(wahlperiode):
            fname = _url_filename(url)
            if fname not in seen_filenames:
                seen_filenames.add(fname)
                urls.append(url)

        # ── Secondary: filterlist API (seed / recent uploads) ────────────────
        logger.info(
            "Supplementing from filterlist API for Wahlperiode %d…", wahlperiode
        )
        offset = 0
        limit = 10
        for page in range(max_pages):
            page_urls = self._fetch_page(offset, limit, wahlperiode)
            if not page_urls:
                logger.info(
                    "Filterlist returned no URLs at offset=%d; stopping.", offset
                )
                break
            new_found = 0
            for url in page_urls:
                fname = _url_filename(url)
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    urls.append(url)
                    new_found += 1
            if new_found == 0:
                # Every URL on this page refers to a session already discovered
                # via dserver (or a previous filterlist page).  The backend is
                # either at its offset limit or repeating results – either way
                # there is nothing new to collect.
                logger.info(
                    "Filterlist stall detected at offset=%d (page %d); "
                    "all results already seen.",
                    offset,
                    page,
                )
                break
            offset += limit
            time.sleep(self.request_delay)

        logger.info(
            "Found %d total protocol URLs for Wahlperiode %d.", len(urls), wahlperiode
        )
        return urls

    def fetch_all_wahlperioden(
        self,
        wahlperioden: List[int],
        max_pages: int = 100,
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """Return deduplicated XML URLs for all given Wahlperioden.

        When *max_workers* is ``None`` (the default) the degree of parallelism
        is chosen automatically as ``min(8, len(wahlperioden))``.  For IO-bound
        HTTP work this provides a significant wall-clock speedup without
        saturating the server.  Pass ``max_workers=1`` to force strictly
        sequential execution (useful in tests or when a single WP is given).

        Each parallel worker thread gets its own :class:`BundestagScraper`
        instance (and therefore its own ``requests.Session``) because
        ``requests.Session`` is not thread-safe.

        :param wahlperioden: Ordered list of Wahlperioden to scrape
            (e.g. ``[18, 19, 20]``).
        :param max_pages: Safety cap forwarded to each
            :meth:`fetch_protocol_urls` call.
        :param max_workers: Number of threads for parallel Wahlperioden
            discovery.  ``None`` (default) auto-selects
            ``min(8, len(wahlperioden))``.
        :return: Flat, deduplicated list of XML download URLs.
        """
        effective_workers = (
            min(8, len(wahlperioden))
            if max_workers is None
            else max_workers
        )

        if effective_workers > 1:
            def _fetch_one_wahlperiode(wp: int) -> List[str]:
                # Each thread gets its own BundestagScraper (and therefore
                # its own requests.Session) to avoid session sharing across
                # threads, which is not safe.
                worker = BundestagScraper(
                    download_dir=self.download_dir,
                    request_delay=self.request_delay,
                    probe_delay=self.probe_delay,
                    skip_existing=self.skip_existing,
                )
                return worker.fetch_protocol_urls(wp, max_pages=max_pages)

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                wp_results = list(executor.map(_fetch_one_wahlperiode, wahlperioden))
        else:
            wp_results = [
                self.fetch_protocol_urls(wp, max_pages=max_pages)
                for wp in wahlperioden
            ]

        seen_filenames: set = set()
        all_urls: List[str] = []

        for wp_urls in wp_results:
            for url in wp_urls:
                fname = _url_filename(url)
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    all_urls.append(url)

        logger.info(
            "fetch_all_wahlperioden: %d total URLs across %d Wahlperioden.",
            len(all_urls),
            len(wahlperioden),
        )
        return all_urls

    def download_all(self, urls: List[str]) -> List[Path]:
        """Download all *urls* and return a list of local file paths.

        Already-downloaded files are skipped without any delay.  A summary of
        newly downloaded vs already-present files is printed at INFO level so
        the operator can see what happened without trawling through debug logs.

        :param urls: List of absolute XML download URLs.
        :return: Paths to the local XML files (both newly downloaded and
            pre-existing).
        """
        paths: List[Path] = []
        total = len(urls)
        downloaded = 0
        skipped = 0
        for i, url in enumerate(urls, 1):
            filename = _url_filename(url)
            dest = self.download_dir / (filename if filename.endswith(".xml") else filename + ".xml")
            # For statistics, treat a file as "already on disk" when it exists and
            # skip_existing is enabled. The actual XML-root validation and decision
            # to reuse or redownload the file is delegated to download_one().
            pre_existed = dest.exists() and self.skip_existing
            path = self.download_one(url)
            if path:
                if pre_existed:
                    skipped += 1
                else:
                    downloaded += 1
                paths.append(path)
            logger.debug("[%d/%d] %s", i, total, url)
        logger.info(
            "Finished: %d/%d files ready  (%d newly downloaded, %d already on disk, %d failed).",
            len(paths), total, downloaded, skipped, total - len(paths),
        )
        return paths

    def download_one(self, url: str) -> Optional[Path]:
        """Download a single XML file.  Returns local path or None on failure.

        After downloading, the file content is validated: only files whose XML
        root element is ``dbtplenarprotokoll`` are kept.  Any other XML (or
        non-XML response) is discarded and ``None`` is returned.
        """
        filename = _url_filename(url)
        if not filename.endswith(".xml"):
            filename = filename + ".xml"
        dest = self.download_dir / filename

        # Skip only if the existing file is readable and valid.
        if dest.exists() and self.skip_existing:
            try:
                existing = dest.read_bytes()
            except OSError as exc:
                logger.warning(
                    "Could not read existing file %s (%s); re-downloading.", dest, exc
                )
            else:
                if _is_dbtplenarprotokoll(existing):
                    logger.info("Skipping %s (already downloaded and valid)", url)
                    return dest
                logger.warning(
                    "Existing file %s is invalid (unexpected XML root); re-downloading.",
                    dest,
                )

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.content
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return None

        if not _is_dbtplenarprotokoll(content):
            logger.warning(
                "Skipping %s: root element is not <%s>", url, _EXPECTED_ROOT_TAG
            )
            return None

        dest.write_bytes(content)
        logger.info("Downloaded: %s → %s", url, dest)
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return dest

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_page(
        self, offset: int, limit: int, wahlperiode: int
    ) -> List[str]:
        """Scrape one paginated results page and return XML URLs."""
        params = {
            "limit": limit,
            "offset": offset,
            "noFilterSet": "true",
        }
        try:
            resp = self._session.get(_OPENDATA_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Error fetching listing page (offset=%d): %s", offset, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls: List[str] = []

        # Compile a precise pattern that matches "<wahlperiode><digits>.xml" only,
        # e.g. "20001.xml" for Wahlperiode 20.  This is more robust than a plain
        # startswith() check, which would accept arbitrary XML files whose names
        # happen to begin with the wahlperiode digits.
        wp_filename_pattern = re.compile(
            rf"^{re.escape(str(wahlperiode))}\d+\.xml$", re.IGNORECASE
        )

        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            # Keep only XML links that match the target Wahlperiode.
            if _XML_LINK_PATTERN.search(href):
                abs_url = href if href.startswith("http") else urljoin(_BASE_URL, href)
                filename = _url_filename(abs_url)
                if wp_filename_pattern.match(filename):
                    urls.append(abs_url)

        return urls

    def _fetch_via_dserver(
        self, wahlperiode: int, max_sessions: int = 300
    ) -> List[str]:
        """Probe the deterministic dserver URL space for *wahlperiode*.

        Sends a streaming GET request (not HEAD – HEAD can return 405 or
        unreliable status codes on some servers) for each candidate session
        number (``1`` to *max_sessions*).  Only the HTTP status code is
        inspected; the body is never read.  A URL is included in the result
        when the server responds with HTTP 200.  Probing stops early once
        ``_MAX_CONSECUTIVE_MISSES`` consecutive non-200 responses have been
        received, which reliably signals the end of a Wahlperiode's session
        list.  Transient 5xx errors are retried up to ``_MAX_RETRIES`` times
        before being counted as a miss.

        :param wahlperiode: The Wahlperiode to scan (e.g. 20).
        :param max_sessions: Hard upper bound on session numbers to probe.
        :return: List of dserver XML URLs that returned HTTP 200.
        """
        urls: List[str] = []
        consecutive_misses = 0

        for sn in range(1, max_sessions + 1):
            url = self._dserver_xml_url(wahlperiode, sn)
            status = self._probe_url(url)
            if status == 200:
                urls.append(url)
                consecutive_misses = 0
            else:
                logger.debug("Miss: %s", url)
                consecutive_misses += 1
                if consecutive_misses >= _MAX_CONSECUTIVE_MISSES:
                    logger.info(
                        "Stopping dserver probe after %d consecutive misses "
                        "(last sitzungsnr=%d).",
                        _MAX_CONSECUTIVE_MISSES,
                        sn,
                    )
                    break
            time.sleep(self.probe_delay)

        logger.info(
            "dserver probe found %d URLs for Wahlperiode %d.", len(urls), wahlperiode
        )
        return urls

    def _probe_url(self, url: str) -> int:
        """Probe *url* with a streaming GET and return ``200`` on success or ``0``.

        Uses ``stream=True`` so the response body is never transferred, making
        the request as lightweight as a HEAD while remaining compatible with
        servers that reject HEAD (405).  The connection timeout is
        ``_PROBE_TIMEOUT`` seconds – sufficient to receive the HTTP status line
        without waiting for a body that will never be read.

        Only HTTP 200 is treated as "found".  Any other non-5xx response (e.g.
        404) returns ``0`` immediately without retry, since these are definitive
        client-side misses.  Transient 5xx responses are retried up to
        ``_MAX_RETRIES`` times before returning ``0``.  Network-level exceptions
        are also retried; ``0`` is returned if all attempts fail.  The wait
        between retries uses :attr:`probe_delay` (not the heavier
        :attr:`request_delay` that governs downloads).

        :param url: URL to probe.
        :return: ``200`` if the server confirmed the resource exists; ``0`` for
            any other outcome (missing resource, server error, or network
            exception).
        """
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(
                    url, timeout=_PROBE_TIMEOUT, allow_redirects=True, stream=True
                )
                resp.close()
                if resp.status_code == 200:
                    return 200
                if resp.status_code < 500:
                    # Definitive client-side response (e.g. 404, 403).  No
                    # point retrying – the resource is simply not there.
                    return 0
                logger.warning(
                    "dserver returned %d for %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt + 1,
                    _MAX_RETRIES,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "dserver GET failed for %s: %s (attempt %d/%d)",
                    url,
                    exc,
                    attempt + 1,
                    _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(self.probe_delay)
        return 0

    @staticmethod
    def _dserver_xml_url(wahlperiode: int, sitzungsnr: int) -> str:
        """Return the deterministic dserver URL for a single session.

        Pattern: ``/btp/{wp}/{wp}{sn:03d}.xml``

        Examples::

            _dserver_xml_url(20, 1)   → "https://dserver.bundestag.de/btp/20/20001.xml"
            _dserver_xml_url(20, 214) → "https://dserver.bundestag.de/btp/20/20214.xml"
            _dserver_xml_url(19, 12)  → "https://dserver.bundestag.de/btp/19/19012.xml"
        """
        return f"{_DSERVER_BASE}/{wahlperiode}/{wahlperiode}{sitzungsnr:03d}.xml"

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)
        return session


def _is_dbtplenarprotokoll(content: bytes) -> bool:
    """Return ``True`` if *content* is a Bundestag plenary-protocol XML.

    Performs a fast root-element check using lxml without loading the entire
    document into memory.  Any parse error or unexpected root tag causes the
    function to return ``False`` so that callers can safely discard the file.

    The parser is configured with ``resolve_entities=False`` and
    ``no_network=True`` to guard against XML entity-expansion attacks (XML
    bombs) and server-side request forgery via external entity references.

    :param content: Raw bytes of the downloaded XML file.
    :return: ``True`` iff the root element tag equals ``"dbtplenarprotokoll"``.
    """
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(content, parser=parser)
        return root.tag == _EXPECTED_ROOT_TAG
    except etree.XMLSyntaxError:
        return False
