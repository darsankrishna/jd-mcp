"""Run each system's merge logic on one case; return a cluster label per record.

- jd-mcp: ``paper_search_mcp.identity.resolve`` (this repo), no persistent store.
- paper-search-mcp: ``_paper_unique_key`` / ``_dedupe_papers`` copied verbatim
  from openags/paper-search-mcp@cab1b5c ``paper_search_mcp/server.py`` (MIT).
- scholarmcp: a line-for-line Python port of the merge loop in
  lstudlo/scholarmcp@0d34ff5 ``packages/scholar-mcp/src/research/literature-service.ts``
  (``resolveTargetKey`` and helpers, MIT). It is not importable on its own: it
  lives inside a service method that also calls the providers.
- Liyux3/scholar-mcp: imports ``scholar_mcp.records.deduplicate`` from a checkout
  you supply. Nothing is copied: it is licensed PolyForm Noncommercial with
  additional terms, so check those terms allow your use before running it.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from typing import Any, Callable, Dict, List

from paper_search_mcp.identity import resolve


def jd_mcp(records: List[Dict[str, Any]]) -> List[int]:
    keys = [(rec["source"], rec["paper_id"]) for rec in records]
    assert len(set(keys)) == len(keys), "benchmark records must be unique per (source, paper_id)"
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_source.setdefault(rec["source"], []).append(dict(rec))
    label = {}
    for number, work in enumerate(resolve(by_source)):
        for member in work["records"]:
            label[(member["source"], member["paper_id"])] = number
    return [label[key] for key in keys]


# --- openags/paper-search-mcp@cab1b5c, verbatim --------------------------------

def _paper_unique_key(paper: Dict[str, Any]) -> str:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"

    title = (paper.get("title") or "").strip().lower()
    authors = (paper.get("authors") or "").strip().lower()
    if title:
        return f"title:{title}|authors:{authors}"

    paper_id = (paper.get("paper_id") or "").strip().lower()
    return f"id:{paper_id}"


def paper_search_mcp(records: List[Dict[str, Any]]) -> List[int]:
    # _dedupe_papers keeps the first record per key and drops the rest.
    keys: Dict[str, int] = {}
    return [keys.setdefault(_paper_unique_key(rec), len(keys)) for rec in records]


# --- lstudlo/scholarmcp@0d34ff5, ported ----------------------------------------

SCHOLARMCP_FUZZY_TITLE_THRESHOLD = 0.84  # config.ts default


def _sm_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sm_strip_non_letters(text: str) -> str:
    # JS: .replace(/[^\p{L}\p{N}\s]/gu, '')
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN" or c.isspace())


def _sm_title_key(title: str) -> str:
    return _sm_strip_non_letters(_sm_whitespace(title).lower())


def _sm_tokens(title: str) -> set:
    # tokenizeForRanking: ASCII-only, as in the original.
    text = re.sub(r"[^a-z0-9\s]", " ", _sm_whitespace(title).lower())
    return {token for token in text.split(" ") if len(token) >= 3}


def _sm_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return overlap / (len(a) + len(b) - overlap)


def _sm_doi(doi: str) -> str | None:
    if not doi:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.IGNORECASE).lower()


def _sm_year(value: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _sm_years_compatible(a, b) -> bool:
    return not a or not b or abs(a - b) <= 2


def _sm_shares_author(left: List[str], right: List[str]) -> bool:
    if not left or not right:
        return True
    names = {_sm_strip_non_letters(_sm_whitespace(n).lower()) for n in left}
    names.discard("")
    return any(_sm_strip_non_letters(_sm_whitespace(n).lower()) in names for n in right)


def scholarmcp(records: List[Dict[str, Any]]) -> List[int]:
    merged: Dict[str, Dict[str, Any]] = {}
    doi_to_key: Dict[str, str] = {}
    title_to_keys: Dict[str, List[str]] = {}
    labels: List[int] = []
    order: Dict[str, int] = {}

    def authors_of(rec):
        return [a.strip() for a in str(rec.get("authors") or "").split(";") if a.strip()]

    def resolve_target(work) -> str | None:
        doi = _sm_doi(work["doi"])
        if doi and doi in doi_to_key:
            return doi_to_key[doi]
        for key in title_to_keys.get(_sm_title_key(work["title"]), []):
            candidate = merged[key]
            if doi and candidate["doi"] and doi != candidate["doi"]:
                continue
            if _sm_years_compatible(candidate["year"], work["year"]) and _sm_shares_author(
                candidate["authors"], work["authors"]
            ):
                return key
        incoming = _sm_tokens(work["title"])
        best_key, best = None, 0.0
        for key, candidate in merged.items():
            if doi and candidate["doi"] and doi != candidate["doi"]:
                continue
            if not _sm_years_compatible(candidate["year"], work["year"]):
                continue
            if not _sm_shares_author(candidate["authors"], work["authors"]):
                continue
            similarity = _sm_jaccard(_sm_tokens(candidate["title"]), incoming)
            if similarity > best:
                best, best_key = similarity, key
        if best_key and best >= SCHOLARMCP_FUZZY_TITLE_THRESHOLD:
            return best_key
        return None

    for rec in records:
        work = {"title": rec.get("title") or "", "doi": rec.get("doi") or "",
                "year": _sm_year(rec.get("published_date")), "authors": authors_of(rec)}
        target = resolve_target(work)
        doi = _sm_doi(work["doi"])
        title_key = _sm_title_key(work["title"])
        if target is None:
            key = doi or f"{rec['source']}:{rec['paper_id']}:{len(merged)}"
            merged[key] = {"title": work["title"], "doi": doi, "year": work["year"], "authors": work["authors"]}
            order[key] = len(order)
            if doi:
                doi_to_key[doi] = key
            title_to_keys.setdefault(title_key, []).append(key)
            labels.append(order[key])
            continue
        existing = merged[target]
        existing["year"] = existing["year"] or work["year"]
        existing["doi"] = existing["doi"] or doi
        existing["authors"] = existing["authors"] or work["authors"]
        if existing["doi"]:
            doi_to_key[existing["doi"]] = target
        if target not in title_to_keys.setdefault(title_key, []):
            title_to_keys[title_key].append(target)
        labels.append(order[target])
    return labels


# --- Liyux3/scholar-mcp, imported from a user-supplied checkout ------------------

def liyux3_loader(checkout: str) -> Callable[[List[Dict[str, Any]]], List[int]]:
    # Appended, not prepended: files in the checkout must not shadow the stdlib.
    sys.path.append(checkout)
    from scholar_mcp.records import deduplicate  # type: ignore

    relation_names = ("is-preprint-of", "has-preprint")

    def run(records: List[Dict[str, Any]]) -> List[int]:
        papers = []
        for index, rec in enumerate(records):
            extra = rec.get("extra") or {}
            ids = {}
            if rec.get("doi"):
                ids["DOI"] = rec["doi"]
            native = {"arxiv": "ArXiv", "pubmed": "PubMed", "openalex": "OpenAlex"}.get(rec["source"])
            if native:
                ids[native] = rec["paper_id"]
            for arxiv_id in extra.get("arxiv_ids") or []:
                ids.setdefault("ArXiv", arxiv_id)
            relations = [
                {"type": name, "id_type": target.get("id-type"), "id": target.get("id")}
                for name in relation_names for target in (extra.get("relation") or {}).get(name, [])
            ]
            year = _sm_year(rec.get("published_date"))
            papers.append({
                "title": rec.get("title") or "", "source": rec["source"],
                "authors": [a.strip() for a in str(rec.get("authors") or "").split(";") if a.strip()],
                "external_ids": ids, "year": year, "relations": relations,
                "publication_types": [extra["crossref_type"]] if extra.get("crossref_type") else [],
                # merge_records unions these keys, which recovers cluster membership.
                "_retrieval_ranks": {f"bench:{index}": 0},
            })
        labels = [0] * len(records)
        for number, paper in enumerate(deduplicate(papers)):
            for key in paper.get("_retrieval_ranks") or {}:
                labels[int(key.split(":", 1)[1])] = number
        return labels

    return run


SYSTEMS: Dict[str, Callable[[List[Dict[str, Any]]], List[int]]] = {
    "jd-mcp": jd_mcp,
    "paper-search-mcp@cab1b5c": paper_search_mcp,
    "scholarmcp@0d34ff5 (port)": scholarmcp,
}
