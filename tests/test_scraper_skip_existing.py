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

import time
from pathlib import Path

from src.scraper import BundestagScraper


_VALID_XML = b'<?xml version="1.0" encoding="UTF-8"?><dbtplenarprotokoll></dbtplenarprotokoll>'
_INVALID_XML = b'<?xml version="1.0" encoding="UTF-8"?><not_the_right_root></not_the_right_root>'


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.get_calls: list[tuple[str, float]] = []

    def get(self, url: str, timeout: float = 30, **kwargs):
        self.get_calls.append((url, timeout))
        return self._response


def test_skip_existing_only_if_valid(tmp_path: Path) -> None:
    url = "https://example.test/20001.xml"
    dest = tmp_path / "20001.xml"

    # Case 1: existing file valid => skip without HTTP call
    dest.write_bytes(_VALID_XML)
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(download_dir=tmp_path, skip_existing=True, session=session)

    out = scraper.download_one(url)
    assert out == dest
    assert session.get_calls == []
    assert dest.read_bytes() == _VALID_XML

    # Case 2: existing file invalid => re-download (HTTP call) and overwrite
    dest.write_bytes(_INVALID_XML)
    session2 = _FakeSession(_FakeResponse(_VALID_XML))
    # request_delay=0 to avoid actual sleeping in tests
    scraper2 = BundestagScraper(download_dir=tmp_path, skip_existing=True, session=session2, request_delay=0)

    out2 = scraper2.download_one(url)
    assert out2 == dest
    assert len(session2.get_calls) == 1
    assert dest.read_bytes() == _VALID_XML


def test_no_sleep_when_skipping(tmp_path: Path) -> None:
    """download_one must not sleep when the file is already cached."""
    url = "https://example.test/20001.xml"
    dest = tmp_path / "20001.xml"
    dest.write_bytes(_VALID_XML)

    session = _FakeSession(_FakeResponse(_VALID_XML))
    # Use a non-zero delay so we'd notice if it fired
    scraper = BundestagScraper(download_dir=tmp_path, skip_existing=True, session=session, request_delay=5.0)

    start = time.monotonic()
    out = scraper.download_one(url)
    elapsed = time.monotonic() - start

    assert out == dest
    assert session.get_calls == []       # no HTTP request
    assert elapsed < 1.0                 # no sleep


def test_sleep_only_on_actual_download(tmp_path: Path) -> None:
    """download_all must not sleep for already-cached files."""
    urls = [
        "https://example.test/20001.xml",
        "https://example.test/20002.xml",
    ]
    for name in ("20001.xml", "20002.xml"):
        (tmp_path / name).write_bytes(_VALID_XML)

    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(download_dir=tmp_path, skip_existing=True, session=session, request_delay=5.0)

    start = time.monotonic()
    paths = scraper.download_all(urls)
    elapsed = time.monotonic() - start

    assert len(paths) == 2
    assert session.get_calls == []  # all cached, no HTTP
    assert elapsed < 1.0            # no sleep at all
