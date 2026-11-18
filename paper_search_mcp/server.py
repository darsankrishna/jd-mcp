# paper_search_mcp/server.py
import argparse
import asyncio
import io
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections.abc import Awaitable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import Field
from urllib.parse import unquote

import httpx
import requests
from mcp.server.fastmcp import FastMCP

from .academic_platforms.arxiv import ArxivSearcher
from .academic_platforms.base_search import BASESearcher
from .academic_platforms.biorxiv import BioRxivSearcher
from .academic_platforms.citeseerx import CiteSeerXSearcher
from .academic_platforms.core import CORESearcher
from .academic_platforms.crossref import CrossRefSearcher
from .academic_platforms.dblp import DBLPSearcher
from .academic_platforms.doaj import DOAJSearcher
from .academic_platforms.europepmc import EuropePMCSearcher
from .academic_platforms.google_scholar import GoogleScholarSearcher
from .academic_platforms.hal import HALSearcher
from .academic_platforms.iacr import IACRSearcher
from .academic_platforms.medrxiv import MedRxivSearcher
from .academic_platforms.openaire import OpenAiresearcher
from .academic_platforms.openalex import OpenAlexSearcher
from .academic_platforms.pmc import PMCSearcher
from .academic_platforms.pubmed import PubMedSearcher
from .academic_platforms.sci_hub import SciHubFetcher
from .academic_platforms.semantic import SemanticSearcher
from .academic_platforms.ssrn import SSRNSearcher
from .academic_platforms.openreview import OpenReviewSearcher
from .academic_platforms.unpaywall import UnpaywallResolver, UnpaywallSearcher
from .academic_platforms.zenodo import ZenodoSearcher
from .config import get_env, load_env_file
from .identity import IdentityStore, normalize_doi, resolve
from .search_cache import SearchCache, search_key

# Initialize MCP server
mcp = FastMCP("paper_search_server")
logger = logging.getLogger(__name__)
GOOGLE_SCHOLAR_TOOL_TIMEOUT_SECONDS = 20.0
SEARCH_PAPERS_SOURCE_TIMEOUT_SECONDS = 45.0
SEARCH_EXECUTOR_MAX_WORKERS = 32


class SearchExecutorSaturatedError(RuntimeError):
    """Raised instead of queuing unbounded work behind blocked providers."""


class _BoundedSearchExecutor:
    """Keep timed-out blocking searches isolated in a fixed-size worker pool."""

    def __init__(self, max_workers: int):
        self._max_workers = max_workers
        self._slots = BoundedSemaphore(max_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="paper-search",
        )

    def submit(self, function, /, *args, **kwargs) -> Future:
        if not self._slots.acquire(blocking=False):
            raise SearchExecutorSaturatedError(
                "search worker capacity is exhausted by unfinished provider calls"
            )

        try:
            future = self._executor.submit(function, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise

        future.add_done_callback(lambda _future: self._slots.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


_SEARCH_EXECUTOR = _BoundedSearchExecutor(SEARCH_EXECUTOR_MAX_WORKERS)
_SECTION_EXECUTOR = _BoundedSearchExecutor(2)
_SEARCH_CACHE = SearchCache.from_env()
_IDENTITY_STORE = IdentityStore(
    get_env("IDENTITY_DB_PATH", "~/.cache/paper-search-mcp/identity.sqlite3")
)
MAX_PDF_BYTES = int(get_env("MAX_PDF_BYTES", str(50 * 1024 * 1024)))

# Instances of searchers
arxiv_searcher = ArxivSearcher()
pubmed_searcher = PubMedSearcher()
biorxiv_searcher = BioRxivSearcher()
medrxiv_searcher = MedRxivSearcher()
google_scholar_searcher = GoogleScholarSearcher()
iacr_searcher = IACRSearcher()
semantic_searcher = SemanticSearcher()
crossref_searcher = CrossRefSearcher()
openalex_searcher = OpenAlexSearcher()
pmc_searcher = PMCSearcher()
core_searcher = CORESearcher()
europepmc_searcher = EuropePMCSearcher()
dblp_searcher = DBLPSearcher()
openaire_searcher = OpenAiresearcher()
citeseerx_searcher = CiteSeerXSearcher()
doaj_searcher = DOAJSearcher()
base_searcher = BASESearcher()
unpaywall_resolver = UnpaywallResolver()
unpaywall_searcher = UnpaywallSearcher(resolver=unpaywall_resolver)
zenodo_searcher = ZenodoSearcher()
hal_searcher = HALSearcher()
ssrn_searcher = SSRNSearcher()
openreview_searcher = OpenReviewSearcher()
# scihub_searcher = SciHubSearcher()


# Asynchronous helper to adapt synchronous searchers
# Runs blocking requests-based calls in a thread pool to avoid blocking the event loop.
async def async_search(searcher, query: str, max_results: int, **kwargs) -> List[Dict]:
    cache = _SEARCH_CACHE
    key = search_key(searcher, query, max_results, kwargs) if cache.settings.enabled else None
    generation = None
    if key is not None:
        cached, generation = await asyncio.wrap_future(
            _SEARCH_EXECUTOR.submit(cache.lookup, key)
        )
        if cached is not None:
            return cached
    search_future = _SEARCH_EXECUTOR.submit(
        searcher.search,
        query,
        max_results=max_results,
        **kwargs,
    )
    papers = await asyncio.wrap_future(search_future)
    result = [paper.to_dict() for paper in papers]
    # A cancelled or failed provider call never reaches the cache write. Clear
    # increments a shared generation so in-flight searches cannot repopulate it.
    if key is not None and generation is not None and result:
        try:
            await asyncio.wrap_future(_SEARCH_EXECUTOR.submit(cache.put, key, result, generation))
        except SearchExecutorSaturatedError:
            pass  # An optional cache must not turn a successful search into failure.
    return result


async def _run_search_with_timeout(
    source_name: str,
    search_task: Awaitable[List[Dict]],
    timeout_seconds: Optional[float] = None,
) -> List[Dict]:
    """Run one source without allowing it to stall a multi-source request."""
    if timeout_seconds is None:
        timeout_seconds = SEARCH_PAPERS_SOURCE_TIMEOUT_SECONDS

    try:
        return await asyncio.wait_for(search_task, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"search for source '{source_name}' timed out after {timeout_seconds:g} seconds"
        ) from exc


ALL_SOURCES = [
    "arxiv",
    "pubmed",
    "biorxiv",
    "medrxiv",
    "iacr",
    "semantic",
    "crossref",
    "openalex",
    "pmc",
    "core",
    "europepmc",
    "dblp",
    "openaire",
    "citeseerx",
    "doaj",
    "base",
    "zenodo",
    "hal",
    "ssrn",
    "openreview",
    "unpaywall",
]


# ---------------------------------------------------------------------------
# Optional paid-platform connectors (disabled by default)
# Set PAPER_SEARCH_MCP_IEEE_API_KEY to activate IEEE Xplore
# (legacy IEEE_API_KEY is also supported).
# ---------------------------------------------------------------------------
_ieee_api_key = get_env("IEEE_API_KEY", "")

if _ieee_api_key:
    from .academic_platforms.ieee import IEEESearcher
    ieee_searcher = IEEESearcher()
    ALL_SOURCES.append("ieee")
    logger.info("IEEE Xplore enabled via configured environment key.")
else:
    ieee_searcher = None

# ACM Digital Library has been open access since 2026-01-01 and has no public
# search API, so it is served from Crossref (no key required) and enabled by
# default.
from .academic_platforms.acm import ACMSearcher
acm_searcher = ACMSearcher()
ALL_SOURCES.append("acm")


# Explicit-only institutional sources never consume quota via sources="all".
# Google Scholar is scraped (throttled, against its terms), so it is never a default.
EXPLICIT_ONLY_SOURCES = ["wos", "scopus", "google_scholar"]


def _parse_sources(sources: str) -> List[str]:
    if not sources or sources.strip().lower() == "all":
        return ALL_SOURCES

    normalized = [part.strip().lower() for part in sources.split(",") if part.strip()]
    return [source for source in normalized if source in ALL_SOURCES + EXPLICIT_ONLY_SOURCES]


def _invalid_sources(sources: str) -> List[str]:
    """Return distinct requested source names that are unknown or unavailable."""
    if not sources or sources.strip().lower() == "all":
        return []

    normalized = [part.strip().lower() for part in sources.split(",") if part.strip()]
    return list(dict.fromkeys(source for source in normalized if source not in ALL_SOURCES + EXPLICIT_ONLY_SOURCES))


def _safe_filename(filename_hint: str, default: str = "paper") -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename_hint).strip("._")
    if not safe:
        return default
    return safe[:120]


async def _download_from_url(
    pdf_url: str,
    save_path: str,
    filename_hint: str = "paper",
    expected_title: str = "",
    expected_doi: str = "",
) -> Optional[str]:
    """Download and, when identity hints are present, verify a fallback PDF."""
    if not pdf_url:
        return None

    output_name = f"{_safe_filename(filename_hint)}.pdf"
    output_path = os.path.join(save_path, output_name)
    temporary_path = ""

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            async with client.stream("GET", pdf_url) as response:
                if response.status_code >= 400:
                    return None
                if int(response.headers.get("content-length") or 0) > MAX_PDF_BYTES:
                    logger.warning("PDF at %s exceeds the %d-byte limit", pdf_url, MAX_PDF_BYTES)
                    return None
                chunks: List[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        logger.warning("PDF at %s exceeds the %d-byte limit", pdf_url, MAX_PDF_BYTES)
                        return None
                    chunks.append(chunk)
                content_type = (response.headers.get("content-type") or "").lower()

        content = b"".join(chunks)
        if not content:
            return None
        if not _looks_like_pdf(content):
            logger.warning(
                "Resolved URL did not return PDF bytes: %s (content-type=%s)",
                pdf_url,
                content_type,
            )
            return None

        if expected_title or expected_doi:
            matches = await asyncio.to_thread(
                _pdf_matches_expected,
                content,
                expected_title,
                expected_doi,
            )
            if not matches:
                logger.warning(
                    "Downloaded PDF from %s could not be verified as title=%r DOI=%r",
                    pdf_url,
                    expected_title[:120],
                    expected_doi,
                )
                return None

        os.makedirs(save_path, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_name}.",
            suffix=".part",
            dir=save_path,
            delete=False,
        ) as file_obj:
            temporary_path = file_obj.name
            file_obj.write(content)
        os.replace(temporary_path, output_path)
        temporary_path = ""
        return output_path
    except Exception as exc:
        logger.warning("Direct URL download failed for %s: %s", pdf_url, exc)
        return None
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _looks_like_pdf(content: bytes) -> bool:
    """Check the PDF header instead of trusting a URL suffix or content type."""
    return bool(content) and b"%PDF-" in content[:1024]


_TITLE_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "among",
        "based",
        "between",
        "from",
        "into",
        "study",
        "that",
        "their",
        "through",
        "using",
        "with",
    }
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _title_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", _normalize_text(value), flags=re.UNICODE)
        if len(token) >= 4 and token not in _TITLE_STOPWORDS
    }


def _normalize_doi(value: str) -> str:
    normalized = unquote((value or "").strip()).casefold()
    normalized = re.sub(
        r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        normalized,
    )
    return re.sub(r"\s+", "", normalized).rstrip(".,;)")


def _title_similarity(a: str, b: str) -> float:
    """Return token F1 similarity for two titles, in the range [0, 1]."""
    if not a or not b:
        return 0.0
    tokens_a = _title_tokens(a)
    tokens_b = _title_tokens(b)
    if not tokens_a or not tokens_b:
        return 1.0 if _normalize_text(a) == _normalize_text(b) else 0.0
    return 2 * len(tokens_a & tokens_b) / (len(tokens_a) + len(tokens_b))


def _pdf_matches_expected(
    pdf_source: str | os.PathLike[str] | bytes,
    expected_title: str,
    expected_doi: str = "",
) -> bool:
    """Verify fallback PDF identity from its first three extractable pages.

    An unreadable or image-only document is unverifiable and therefore rejected
    by the fallback chain. Direct source downloads without identity hints are not
    affected by this conservative rule.
    """
    if not expected_title and not expected_doi:
        return True

    try:
        from pypdf import PdfReader

        reader_source = io.BytesIO(pdf_source) if isinstance(pdf_source, bytes) else pdf_source
        reader = PdfReader(reader_source)
    except Exception as exc:
        logger.debug("Could not parse fallback PDF for identity verification: %s", exc)
        return False

    try:
        text_parts = []
        for page in reader.pages[:3]:
            text_parts.append(page.extract_text() or "")
    except Exception as exc:
        logger.debug("Fallback PDF text extraction failed: %s", exc)
        return False

    text = "\n".join(text_parts)
    if not text.strip():
        return False

    normalized_text = _normalize_text(text)
    normalized_doi = _normalize_doi(expected_doi)
    if normalized_doi:
        text_without_whitespace = re.sub(r"\s+", "", normalized_text)
        if normalized_doi in text_without_whitespace:
            return True
        doi_alphanumeric = re.sub(r"[^a-z0-9]", "", normalized_doi)
        text_alphanumeric = re.sub(r"[^a-z0-9]", "", normalized_text)
        if doi_alphanumeric and doi_alphanumeric in text_alphanumeric:
            return True

    expected_tokens = _title_tokens(expected_title)
    if not expected_tokens:
        return False
    document_tokens = _title_tokens(text)
    matching_tokens = expected_tokens & document_tokens
    return len(matching_tokens) / len(expected_tokens) >= 0.6


async def _try_repository_fallback(
    doi: str,
    title: str,
    save_path: str,
    expected_title: str | None = None,
) -> tuple[Optional[str], str]:
    """Search OA repositories for a paper matching the DOI or title, then download.

    When `expected_title` is provided, candidate papers whose titles are too
    dissimilar are skipped — this prevents the fallback from returning a
    plausible-but-wrong PDF. The same `expected_title` is propagated to
    `_download_from_url` so the downloaded PDF's content is also verified.
    """
    repository_searchers = [
        ("arxiv", arxiv_searcher),
        ("openaire", openaire_searcher),
        ("core", core_searcher),
        ("europepmc", europepmc_searcher),
        ("pmc", pmc_searcher),
    ]

    validation_title = title if expected_title is None else expected_title
    normalized_expected_doi = _normalize_doi(doi)
    query_candidates = list(
        dict.fromkeys(
            candidate
            for candidate in ((doi or "").strip(), (title or "").strip())
            if candidate
        )
    )
    if not query_candidates:
        return None, "no DOI/title provided for repository fallback"

    repository_errors: List[str] = []
    rejected_candidates = 0
    attempted_urls: set[str] = set()

    for repo_name, searcher in repository_searchers:
        for query in query_candidates:
            try:
                papers = await asyncio.to_thread(searcher.search, query, max_results=3)
            except Exception as exc:
                repository_errors.append(f"{repo_name}:{exc}")
                continue

            if not papers:
                continue

            for paper in papers:
                pdf_url = str(getattr(paper, "pdf_url", "") or "").strip()
                if not pdf_url or pdf_url in attempted_urls:
                    continue

                candidate_doi = _normalize_doi(str(getattr(paper, "doi", "") or ""))
                doi_matches = bool(
                    normalized_expected_doi
                    and candidate_doi
                    and normalized_expected_doi == candidate_doi
                )
                if validation_title and not doi_matches:
                    candidate_title = str(getattr(paper, "title", "") or "")
                    if candidate_title:
                        sim = _title_similarity(candidate_title, validation_title)
                        if sim < 0.6:
                            rejected_candidates += 1
                            logger.warning(
                                "Repository %s fallback title mismatch (sim=%.2f): "
                                "'%s' vs expected '%s' — skipping",
                                repo_name,
                                sim,
                                candidate_title[:80],
                                validation_title[:80],
                            )
                            continue

                attempted_urls.add(pdf_url)
                raw_paper_id = getattr(paper, "paper_id", "")
                paper_id = str(raw_paper_id or query).strip()
                downloaded = await _download_from_url(
                    pdf_url,
                    save_path,
                    f"{repo_name}_{paper_id}",
                    expected_title=validation_title,
                    expected_doi=doi,
                )
                if downloaded:
                    return downloaded, ""
                rejected_candidates += 1

    if rejected_candidates:
        repository_errors.append(
            f"{rejected_candidates} candidate PDF(s) did not match the requested paper"
        )
    if not repository_errors:
        repository_errors.append("no repository PDF candidate found")
    return None, "; ".join(repository_errors)


# ---------------------------------------------------------------------------
# Source registry (replaces the per-source if/elif dispatch and wrapper tools)
# ---------------------------------------------------------------------------
SEARCHERS: Dict[str, Any] = {
    "arxiv": arxiv_searcher,
    "pubmed": pubmed_searcher,
    "biorxiv": biorxiv_searcher,
    "medrxiv": medrxiv_searcher,
    "google_scholar": google_scholar_searcher,
    "iacr": iacr_searcher,
    "semantic": semantic_searcher,
    "crossref": crossref_searcher,
    "openalex": openalex_searcher,
    "pmc": pmc_searcher,
    "core": core_searcher,
    "europepmc": europepmc_searcher,
    "dblp": dblp_searcher,
    "openaire": openaire_searcher,
    "citeseerx": citeseerx_searcher,
    "doaj": doaj_searcher,
    "base": base_searcher,
    "zenodo": zenodo_searcher,
    "hal": hal_searcher,
    "ssrn": ssrn_searcher,
    "openreview": openreview_searcher,
    "unpaywall": unpaywall_searcher,
    "acm": acm_searcher,
}
if ieee_searcher is not None:
    SEARCHERS["ieee"] = ieee_searcher


def _searcher(source: str) -> Any:
    # Institutional connectors read their keys at call time.
    if source == "wos":
        from .academic_platforms.wos import WebOfScienceSearcher
        return WebOfScienceSearcher()
    if source == "scopus":
        from .academic_platforms.scopus import ScopusSearcher
        return ScopusSearcher()
    if source not in SEARCHERS:
        raise ValueError(f"Unknown or unavailable source '{source}'.")
    return SEARCHERS[source]


async def _search_source(source: str, query: str, max_results: int, year: str = "") -> List[Dict]:
    kwargs: Dict[str, Any] = {}
    if source == "semantic" and year:
        kwargs["year"] = year
    elif source == "iacr":
        kwargs["fetch_details"] = False
    elif source == "google_scholar":
        kwargs["timeout_seconds"] = GOOGLE_SCHOLAR_TOOL_TIMEOUT_SECONDS - 1.0
    return await async_search(_searcher(source), query, max_results, **kwargs)


def _provider_error(exc: BaseException) -> Dict[str, str]:
    """Typed per-source failure, so an outage is never mistaken for zero results."""
    if isinstance(exc, TimeoutError):
        kind = "timeout"
    elif isinstance(exc, SearchExecutorSaturatedError):
        kind = "capacity"
    elif isinstance(exc, (httpx.HTTPError, requests.RequestException)):
        kind = "http"
    elif isinstance(exc, ValueError):
        kind = "invalid_request"
    else:
        kind = "provider_error"
    return {"kind": kind, "type": type(exc).__name__, "message": str(exc) or type(exc).__name__}


def _resolve_works(results: Dict[str, List[Dict]], errors: Dict[str, Any]) -> List[Dict]:
    try:
        return resolve(results, _IDENTITY_STORE)
    except (sqlite3.Error, OSError) as exc:
        # A broken identity store must not lose the search; UPIDs are then per-query.
        errors["identity_store"] = _provider_error(exc)
        return resolve(results, None)


# Read-only towards providers; like the search cache, it records identities locally.
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def search_papers(
    query: str,
    max_results_per_source: int = 5,
    sources: str = "all",
    year: str = "",
) -> Dict[str, Any]:
    """Search academic sources in parallel and return deduplicated, fused works.

    Records describing the same work (preprint and published version included)
    are merged under a persistent UPID, with merge evidence and per-field
    provenance. Works are ranked by reciprocal rank fusion across sources.

    Args:
        query: Search query string.
        max_results_per_source: Max results to fetch from each selected source.
        sources: Comma-separated source names or 'all'. 'all' covers:
            arxiv,pubmed,biorxiv,medrxiv,iacr,semantic,crossref,openalex,pmc,core,europepmc,dblp,openaire,citeseerx,doaj,base,zenodo,hal,ssrn,openreview,unpaywall,acm
            (plus ieee when keyed). Explicit only: wos, scopus, google_scholar.
        year: Optional year filter for Semantic Scholar only.
    Returns:
        Works ranked by RRF, per-source counts, and typed per-source errors.
    """
    selected = list(dict.fromkeys(_parse_sources(sources)))
    errors: Dict[str, Any] = {
        source: {"kind": "invalid_request", "type": "ValueError", "message": "Unknown or unavailable source."}
        for source in _invalid_sources(sources)
    }
    if not selected:
        errors["sources"] = {"kind": "invalid_request", "type": "ValueError", "message": "No valid sources selected."}
        return {"query": query, "sources_requested": sources, "sources_used": [],
                "source_results": {}, "errors": errors, "papers": [], "total": 0, "raw_total": 0}

    outputs = await asyncio.gather(
        *(
            _run_search_with_timeout(
                source,
                _search_source(source, query, max_results_per_source, year),
                GOOGLE_SCHOLAR_TOOL_TIMEOUT_SECONDS if source == "google_scholar" else None,
            )
            for source in selected
        ),
        return_exceptions=True,
    )

    results: Dict[str, List[Dict]] = {}
    source_results: Dict[str, int] = {}
    for source, output in zip(selected, outputs):
        if isinstance(output, Exception):
            errors[source] = _provider_error(output)
            source_results[source] = 0
            continue
        results[source] = output
        source_results[source] = len(output)

    works = await asyncio.to_thread(_resolve_works, results, errors)
    return {
        "query": query,
        "sources_requested": sources,
        "sources_used": selected,
        "source_results": source_results,
        "errors": errors,
        "papers": works,
        "total": len(works),
        "raw_total": sum(source_results.values()),
    }


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_paper(identifier: str) -> Dict[str, Any]:
    """Look up a work by DOI, arXiv id, PMID, OpenAlex/S2 id or UPID.

    Returns the persistent identity seen in earlier searches (UPID, crosswalk
    identifiers, redirects, quarantined conflicting ids) and Crossref metadata
    when a DOI is known.
    """
    identity = await asyncio.to_thread(_IDENTITY_STORE.describe, identifier)
    doi = normalize_doi(identifier) or next(
        (i[len("doi:"):] for i in identity.get("identifiers", []) if i.startswith("doi:")), ""
    )
    paper = await asyncio.to_thread(crossref_searcher.get_paper_by_doi, doi) if doi else None
    return {"identity": identity, "metadata": paper.to_dict() if paper else {}}


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
async def explain_match(identifier_a: str, identifier_b: str) -> Dict[str, Any]:
    """Explain whether two identifiers resolve to the same work, and why.

    Accepts DOIs, arXiv ids, 'pmid:..', 'openalex:..', 's2:..', UPIDs or record
    keys ('src:<source>:<paper_id>') from search results. Returns merged,
    confidence, the stored reason, and the evidence chain or veto.
    """
    return await asyncio.to_thread(_IDENTITY_STORE.explain, identifier_a, identifier_b)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True})
async def download_paper(
    source: str,
    paper_id: str,
    doi: str = "",
    title: str = "",
    save_path: str = "./downloads",
    use_scihub: bool = False,
) -> str:
    """Download a PDF: source-native, then OA repositories, then Unpaywall.

    PDFs larger than PAPER_SEARCH_MCP_MAX_PDF_BYTES (default 50 MiB) are refused.
    Sci-Hub is never used unless use_scihub=True is passed explicitly.

    Args:
        source: Source name the paper_id comes from (e.g. arxiv, semantic, crossref).
        paper_id: Source-native paper identifier.
        doi: Optional DOI for OA fallback and identity verification.
        title: Optional title for OA fallback and identity verification.
        save_path: Directory to save downloaded files.
        use_scihub: Explicit opt-in to try Sci-Hub after OA attempts fail.
    Returns:
        Download path on success or explanatory error message.
    """
    source_name = source.strip().lower()
    attempt_errors: List[str] = []
    primary_error = ""
    downloader = getattr(SEARCHERS.get(source_name), "download_pdf", None)
    if downloader is not None:
        try:
            primary_result = await asyncio.to_thread(downloader, paper_id, save_path)
            if isinstance(primary_result, str) and os.path.isfile(primary_result):
                # shortcut: connector downloads are size-checked after writing, not streamed;
                # move each connector onto _download_from_url to bound memory as well.
                if os.path.getsize(primary_result) <= MAX_PDF_BYTES:
                    return primary_result
                os.remove(primary_result)
                primary_error = f"PDF exceeds the {MAX_PDF_BYTES}-byte download limit"
            elif isinstance(primary_result, str) and primary_result:
                primary_error = primary_result
        except Exception as exc:
            primary_error = str(exc)
            logger.warning("Primary download failed for %s/%s: %s", source_name, paper_id, exc)
    else:
        primary_error = f"Unsupported source '{source_name}' for primary download."
    if primary_error:
        attempt_errors.append(f"primary: {primary_error}")

    repository_result, repository_error = await _try_repository_fallback(
        doi, title, save_path, expected_title=title,
    )
    if repository_result:
        return repository_result
    if repository_error:
        attempt_errors.append(f"repositories: {repository_error}")

    normalized_doi = (doi or "").strip()
    if normalized_doi:
        unpaywall_url = await asyncio.to_thread(unpaywall_resolver.resolve_best_pdf_url, normalized_doi)
        if unpaywall_url:
            unpaywall_result = await _download_from_url(
                unpaywall_url, save_path, f"unpaywall_{normalized_doi}",
                expected_title=title, expected_doi=normalized_doi,
            )
            if unpaywall_result:
                return unpaywall_result
            attempt_errors.append("unpaywall: resolved OA URL but download failed (or content did not match expected title)")
        else:
            attempt_errors.append("unpaywall: no OA URL found (or PAPER_SEARCH_MCP_UNPAYWALL_EMAIL/UNPAYWALL_EMAIL missing)")
    else:
        attempt_errors.append("unpaywall: DOI not provided")

    if not use_scihub:
        return "Download failed after OA fallback chain. Details: " + " | ".join(attempt_errors)

    fallback_identifier = (doi or "").strip() or (title or "").strip() or paper_id
    fetcher = SciHubFetcher(output_dir=save_path)
    fallback_result = await asyncio.to_thread(fetcher.download_pdf, fallback_identifier)
    if fallback_result and os.path.isfile(fallback_result):
        verified = os.path.getsize(fallback_result) <= MAX_PDF_BYTES and await asyncio.to_thread(
            _pdf_matches_expected, fallback_result, title, doi,
        )
        if verified:
            return fallback_result
        attempt_errors.append("scihub: downloaded PDF was too large or did not match the requested paper")
        try:
            os.remove(fallback_result)
        except OSError as exc:
            logger.warning("Could not remove unverified Sci-Hub download %s: %s", fallback_result, exc)

    return "Download failed after OA fallback chain and Sci-Hub fallback. Details: " + " | ".join(attempt_errors)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True})
async def read_paper(source: str, paper_id: str, save_path: str = "./downloads") -> str:
    """Download (if needed) and extract the text of a paper from one source.

    Args:
        source: Source name the paper_id comes from (e.g. arxiv, biorxiv, semantic).
        paper_id: Source-native paper identifier.
        save_path: Directory where the PDF is/will be saved.
    Returns:
        Extracted text, or the connector's explanation when reading is unsupported.
    """
    searcher = _searcher(source.strip().lower())
    if not hasattr(searcher, "read_paper") or source.strip().lower() == "scopus":
        raise ValueError(f"Source '{source}' does not support reading papers.")
    return await asyncio.to_thread(searcher.read_paper, paper_id, save_path)


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def extract_sections(
    pdf_path: str,
    max_pages: Annotated[int, Field(strict=True, ge=1, le=100)] = 30,
    max_chars: Annotated[int, Field(strict=True, ge=1, le=200_000)] = 60_000,
    max_sections: Annotated[int, Field(strict=True, ge=1, le=100)] = 50,
    timeout_seconds: Annotated[float, Field(strict=True, gt=0, le=60, allow_inf_nan=False)] = 30.0,
) -> Dict:
    """Split an already-downloaded PDF into heuristic heading-based sections.

    pdf_path must be relative to PAPER_SEARCH_MCP_SECTION_PDF_ROOT (default
    ./downloads); absolute paths, traversal and symlinks are rejected. Reads at
    most 20 MiB, 100 pages, 200000 characters and 100 sections, with a 60-second
    maximum call budget. Parsing uses a resource-limited process (POSIX required)
    and separate two-job capacity. Defaults are smaller. Preserves ordering/page spans,
    repeats and unclassified text, and marks truncation. English standalone
    headings only; no OCR, semantic fields, medical inference or new downloads.
    """
    from .sections import extract_pdf_sections, validate_limits
    validate_limits(max_pages, max_chars, max_sections, timeout_seconds)
    cancellation = threading.Event()
    try:
        future = _SECTION_EXECUTOR.submit(
            extract_pdf_sections, pdf_path, max_pages, max_chars, max_sections, timeout_seconds,
            _cancel_event=cancellation,
        )
    except SearchExecutorSaturatedError as exc:
        raise RuntimeError("PDF parser capacity is exhausted; try again later") from exc
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("PDF section extraction exceeded its time budget") from exc
    finally:
        # Cancellation/timeout stops and reaps the child. Only this dedicated
        # two-slot supervisor pool stays occupied during bounded cleanup.
        cancellation.set()



async def _openalex_relationship(method, identifier, max_results, filter,
                                 max_pages, max_requests, timeout_seconds):
    from .academic_platforms.openalex_relations import validate_options
    validate_options(max_results, max_pages, max_requests, timeout_seconds)
    future = _SEARCH_EXECUTOR.submit(
        method, identifier, max_results=max_results, filter=filter,
        max_pages=max_pages, max_requests=max_requests,
        timeout_seconds=timeout_seconds,
    )
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("OpenAlex relationship lookup exceeded its time budget") from exc


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_related_papers(
    identifier: str,
    direction: Literal["citing", "references"] = "citing",
    max_results: int = 10,
    filter: str = "",
    max_pages: int = 5,
    max_requests: int = 8,
    timeout_seconds: float = 30.0,
) -> Dict:
    """One hop of papers citing, or referenced by, a DOI or OpenAlex work ID.

    Never resolves titles. Most-cited first for citing. Limits: 500 results,
    5 pages, 8 HTTP requests, 60 seconds; the seed lookup and redirects consume
    requests. API failures raise an error rather than an empty success. Missing
    OpenAlex reference coverage may yield an empty list for 'references'.
    """
    method = openalex_searcher.get_citations if direction == "citing" else openalex_searcher.get_references
    return await _openalex_relationship(
        method, identifier, max_results, filter, max_pages, max_requests, timeout_seconds,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False})
async def manage_search_cache(action: Literal["status", "clear"] = "status") -> Dict:
    """Show or clear the opt-in local search cache (never shows query contents).

    status: settings and entry count. clear: delete all cached search results;
    works while caching is disabled and never deletes PDFs or the identity store.
    Enable with PAPER_SEARCH_MCP_SEARCH_CACHE_ENABLED and restart the server.
    """
    operation = _SEARCH_CACHE.status if action == "status" else _SEARCH_CACHE.clear
    return await asyncio.wrap_future(_SEARCH_EXECUTOR.submit(operation))


def _wait_for_windows_process_exit(process_id: int) -> bool:
    """Wait for a Windows process handle to become signalled."""
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    wait_object_0 = 0x00000000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(synchronize, False, process_id)
    if not handle:
        logger.warning(
            "Could not watch MCP client process %s (Windows error %s)",
            process_id,
            ctypes.get_last_error(),
        )
        return False

    try:
        result = kernel32.WaitForSingleObject(handle, infinite)
    finally:
        kernel32.CloseHandle(handle)

    if result != wait_object_0:
        logger.warning(
            "Waiting for MCP client process %s failed (result %#x)",
            process_id,
            result,
        )
        return False
    return True


def _exit_when_orphaned(poll_seconds: float = 5.0) -> None:
    """Exit if the MCP client that spawned this stdio server goes away.

    A stdio server is owned by exactly one client. When that client dies without
    closing the pipe cleanly, ``mcp.run`` keeps blocking on stdin and the process
    survives indefinitely, re-adopted by init. They accumulate: nine of these had
    piled up on one developer machine, the oldest running for over a day.

    Watch for reparenting and leave.
    """
    original_ppid = os.getppid()
    if os.name == "nt":
        # Windows keeps reporting the original parent PID after that process has
        # exited, so polling getppid() cannot detect orphaning. A process handle
        # becomes signalled at termination and works without an extra dependency.
        if not _wait_for_windows_process_exit(original_ppid):
            return
        logger.info("MCP client process %s exited; shutting down", original_ppid)
        os._exit(0)
        return

    while True:
        time.sleep(poll_seconds)
        ppid = os.getppid()
        if ppid == 1 or ppid != original_ppid:
            logger.info(
                "MCP client gone (ppid %s -> %s); shutting down", original_ppid, ppid
            )
            os._exit(0)
            return


def _server_env(name: str, default: str) -> str:
    """Return a server setting, preferring the repository-wide env prefix."""
    load_env_file()
    for key in (f"PAPER_SEARCH_MCP_{name}", f"PAPER_SEARCH_{name}"):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _valid_port(raw: str) -> int:
    try:
        port = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "port must be an integer between 1 and 65535"
        ) from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer between 1 and 65535")
    return port


def _valid_http_path(raw: str) -> str:
    path = raw.strip()
    if not path.startswith("/"):
        raise argparse.ArgumentTypeError("path must start with '/'")
    return path


def _build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper Search MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=_server_env("TRANSPORT", "stdio"),
        help="MCP transport (default: stdio; env: PAPER_SEARCH_MCP_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=_server_env("HOST", "127.0.0.1"),
        help="Network bind host (env: PAPER_SEARCH_MCP_HOST)",
    )
    parser.add_argument(
        "--port",
        type=_valid_port,
        default=_server_env("PORT", "8000"),
        help="Network bind port (env: PAPER_SEARCH_MCP_PORT)",
    )
    parser.add_argument(
        "--path",
        type=_valid_http_path,
        default=_server_env("PATH", "/mcp"),
        help="Streamable HTTP endpoint path (env: PAPER_SEARCH_MCP_PATH)",
    )
    parser.add_argument(
        "--auth", choices=("none", "oauth"), default=None,
        help="HTTP auth mode (env: PAPER_SEARCH_MCP_AUTH; default: none; ignored for stdio)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run over stdio or a shared network transport.

    Command-line arguments override environment settings. ``stdio`` stays the
    default, so existing MCP client configurations remain compatible. The
    shorter ``PAPER_SEARCH_*`` names introduced by PR #114 remain accepted as
    aliases for the preferred ``PAPER_SEARCH_MCP_*`` names.
    """
    parser = _build_server_parser()
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        # Only meaningful for stdio: an http server has no owning client to outlive.
        threading.Thread(
            target=_exit_when_orphaned, name="orphan-watchdog", daemon=True
        ).start()
        mcp.run(transport="stdio")
        return

    from .http_auth import AuthConfigurationError, load_http_auth_config, run_protected_http

    try:
        auth_config = load_http_auth_config(
            args.auth, args.path if args.transport == "streamable-http" else "/sse"
        )
    except AuthConfigurationError as exc:
        parser.error(str(exc))

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if args.transport == "streamable-http":
        mcp.settings.streamable_http_path = args.path
    logger.info(
        "serving %s on %s:%s", args.transport, mcp.settings.host, mcp.settings.port
    )
    if auth_config is not None:
        run_protected_http(mcp, args.transport, auth_config)
    else:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
