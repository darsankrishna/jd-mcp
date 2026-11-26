"""Work-level paper identity: evidence-based matching, vetoes, persistent UPIDs.

Records are ``Paper.to_dict()`` dicts. Identifiers are read only from explicit
fields (DOI, source IDs, arXiv ``doi``/``journal-ref``, Crossref relations,
OpenAlex ids/locations), never guessed from abstract text.

Pipeline: identifiers + pairwise evidence -> edges -> union-find with
cannot-link vetoes -> works -> persistent UPIDs -> reciprocal rank fusion.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import unicodedata
import uuid
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

RRF_K = 60
MAX_YEAR_GAP = 2
MODEL_PATH = Path(__file__).with_name("match_model.json")

PREPRINT_SOURCES = frozenset({"arxiv", "biorxiv", "medrxiv", "chemrxiv", "ssrn", "iacr"})
# Single-valued identifier types; more than one value in a work is a conflict.
SINGLE_VALUED = ("doi", "preprint_doi", "arxiv", "pmid", "openalex", "s2")
ERRATUM_WORDS = frozenset({
    "erratum", "errata", "corrigendum", "corrigenda", "correction", "corrections",
    "retraction", "retracted", "addendum",
})
GENERIC_TITLES = frozenset({
    "introduction", "editorial", "preface", "foreword", "index", "contents",
    "conclusion", "conclusions", "discussion", "reply", "abstract", "frontmatter",
    "backmatter", "editorialboard", "bookreviews", "correspondence",
})
_NUMBER_MARKERS = frozenset({"part", "vol", "volume", "chapter", "book", "section", "no", "number"})
_ROMAN = re.compile(r"^[ivxlcdm]+$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


# ---------------------------------------------------------------------------
# Normalisation (Unicode-safe: letters, marks and numbers of every script)
# ---------------------------------------------------------------------------

def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold()
    # Keep categories L, M and N: Indic vowel signs and viramas are marks (M),
    # and dropping them, as `\w` tokenizers do, splits words apart.
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in text)
    return " ".join(kept.split())


def title_tokens(value: str) -> List[str]:
    return normalize_title(value).split()


def title_similarity(a: str, b: str) -> float:
    """Dice coefficient over character bigrams; works for unspaced scripts (CJK)."""
    x, y = normalize_title(a).replace(" ", ""), normalize_title(b).replace(" ", "")
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    if len(x) < 2 or len(y) < 2:
        return 0.0
    grams_x = _bigrams(x)
    grams_y = dict(_bigrams(y))
    overlap = 0
    for gram, count in grams_x.items():
        overlap += min(count, grams_y.get(gram, 0))
    return 2 * overlap / (len(x) - 1 + len(y) - 1)


def _bigrams(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in range(len(text) - 1):
        counts[text[i:i + 2]] = counts.get(text[i:i + 2], 0) + 1
    return counts


def normalize_doi(value: str) -> str:
    doi = unquote((value or "").strip()).casefold()
    doi = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", doi)
    doi = re.sub(r"\s+", "", doi).rstrip(".,;)")
    return doi if doi.startswith("10.") and "/" in doi else ""


def normalize_arxiv(value: str) -> str:
    arxiv_id = (value or "").strip().casefold()
    arxiv_id = re.sub(r"^(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)", "", arxiv_id)
    arxiv_id = re.sub(r"\.pdf$", "", arxiv_id)
    return re.sub(r"v\d+$", "", arxiv_id)


def _fold(text: str) -> str:
    # Accent folding for names only (Müller == Muller); applied to both sides.
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _initials_last(name: str) -> bool:
    tokens = unicodedata.normalize("NFKC", name).split()
    last = tokens[-1] if tokens else ""
    return (len(tokens) > 1 and "," not in name and "." not in name
            and last.isalpha() and last.isupper() and len(last) <= 3)


def author_keys(names: List[str]) -> List[Tuple[str, str]]:
    # PubMed/MEDLINE lists put initials after the surname ('Love MI; Huber W').
    # Decided per list: one name like 'Kaiming HE' is ambiguous, a whole list is not.
    initials_last = bool(names) and all(_initials_last(n) for n in names)
    return [key for key in (author_key(n, initials_last) for n in names) if key]


def author_key(name: str, initials_last: bool = False) -> Optional[Tuple[str, str]]:
    """(surname, first initial) for 'Kaiming He', 'He, Kaiming', 'K. He', and 'He K' in PubMed lists."""
    name = _fold(unicodedata.normalize("NFKC", name or "").casefold()).strip()
    if "," in name:
        surname_part, _, given_part = name.partition(",")
    else:
        parts = name.split()
        if not parts:
            return None
        if initials_last and len(parts) > 1:
            surname_part, given_part = " ".join(parts[:-1]), parts[-1]
        else:
            surname_part, given_part = parts[-1], " ".join(parts[:-1])
    surname_tokens = normalize_title(surname_part).split()
    given_tokens = normalize_title(given_part).split()
    if not surname_tokens:
        return None
    return surname_tokens[-1], given_tokens[0][0] if given_tokens else ""


def split_authors(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def author_overlap(a: List[Tuple[str, str]], b: List[Tuple[str, str]]) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)

    def same(x: Tuple[str, str], y: Tuple[str, str]) -> bool:
        return x[0] == y[0] and (not x[1] or not y[1] or x[1] == y[1])

    return sum(1 for x in small if any(same(x, y) for y in large)) / len(small)


def _roman_to_int(token: str) -> int:
    total = 0
    for i, ch in enumerate(token):
        value = _ROMAN_VALUES[ch]
        nxt = _ROMAN_VALUES[token[i + 1]] if i + 1 < len(token) else 0
        total += -value if value < nxt else value
    return total


def title_numbers(title: str) -> frozenset:
    """Numbers and Part/Volume roman numerals, e.g. 'Part II' -> {2}."""
    tokens = title_tokens(title)
    numbers = set()
    for i, token in enumerate(tokens):
        if token.isdecimal():
            numbers.add(int(token))
        elif _ROMAN.match(token) and i > 0 and tokens[i - 1] in _NUMBER_MARKERS:
            numbers.add(_roman_to_int(token))
    return frozenset(numbers)


def _year(record: Dict[str, Any]) -> Optional[int]:
    match = re.match(r"\s*(\d{4})", str(record.get("published_date") or ""))
    return int(match.group(1)) if match else None


def _extra(record: Dict[str, Any]) -> Dict[str, Any]:
    extra = record.get("extra")
    return extra if isinstance(extra, dict) else {}


# ---------------------------------------------------------------------------
# Identifiers from explicit fields
# ---------------------------------------------------------------------------

def record_key(record: Dict[str, Any]) -> str:
    return f"src:{record.get('source', '')}:{str(record.get('paper_id') or '').strip().casefold()}"


# DOI prefixes of preprint servers: arXiv, bioRxiv/medRxiv (numeric suffix; CSHL
# journals share 10.1101 with letter suffixes), SSRN, Research Square,
# Preprints.org, Authorea, ChemRxiv, OSF and PsyArXiv.
_PREPRINT_DOI = re.compile(
    r"^10\.(?:48550/|1101/\d|2139/ssrn|21203/rs\.|20944/preprints|22541/au\.|26434/chemrxiv|31219/osf|31234/)"
)


def is_preprint_doi(doi: str) -> bool:
    return bool(_PREPRINT_DOI.match(normalize_doi(doi)))


def is_preprint(record: Dict[str, Any]) -> bool:
    return (
        record.get("source") in PREPRINT_SOURCES
        or is_preprint_doi(record.get("doi", ""))
        or _extra(record).get("crossref_type") == "posted-content"
    )


def record_identifiers(record: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    """Return (key, reason, primary) tuples. Non-primary ids only link records."""
    source = record.get("source", "")
    paper_id = str(record.get("paper_id") or "").strip()
    extra = _extra(record)
    ids: List[Tuple[str, str, bool]] = [(record_key(record), "source record", False)]

    doi = normalize_doi(record.get("doi", ""))
    if doi:
        reason = "arXiv doi field" if source == "arxiv" else f"{source} doi field"
        ids.append((f"doi:{doi}", reason, True))
        if doi.startswith("10.48550/arxiv."):
            ids.append((f"arxiv:{normalize_arxiv(doi[len('10.48550/arxiv.'):])}", "arXiv DOI", True))

    native_types = {"arxiv": "arxiv", "pubmed": "pmid", "openalex": "openalex", "semantic": "s2"}
    if paper_id and source in native_types:
        id_type = native_types[source]
        value = normalize_arxiv(paper_id) if id_type == "arxiv" else paper_id.casefold()
        ids.append((f"{id_type}:{value}", f"{source} record id", True))

    for value in extra.get("arxiv_ids") or []:
        ids.append((f"arxiv:{normalize_arxiv(value)}", f"{source} arXiv id field", True))
    if extra.get("pmid"):
        ids.append((f"pmid:{str(extra['pmid']).strip().casefold()}", f"{source} pmid field", True))

    relations = extra.get("relation") or {}
    for relation in ("is-preprint-of", "has-preprint", "is-version-of", "has-version"):
        for target in relations.get(relation) or []:
            if not isinstance(target, dict):
                continue
            id_type = str(target.get("id-type", "")).casefold()
            if id_type == "doi" and normalize_doi(target.get("id", "")):
                ids.append((f"doi:{normalize_doi(target['id'])}", f"crossref {relation}", False))
            elif id_type == "arxiv":
                ids.append((f"arxiv:{normalize_arxiv(target.get('id', ''))}", f"crossref {relation}", False))
    return list(dict.fromkeys(ids))


def _primary_values(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(type, value) pairs a record claims about itself, for conflict checks."""
    values = []
    for key, _reason, primary in record_identifiers(record):
        if not primary:
            continue
        id_type, value = key.split(":", 1)
        if id_type == "doi" and is_preprint(record):
            id_type = "preprint_doi"
        values.append((id_type, value))
    return values


# ---------------------------------------------------------------------------
# Pairwise evidence and vetoes
# ---------------------------------------------------------------------------

class _Features:
    __slots__ = ("title", "numbers", "erratum", "generic", "authors", "year", "journal_ref", "venue",
                 "published_doi")

    def __init__(self, record: Dict[str, Any]):
        self.title = record.get("title") or ""
        tokens = set(title_tokens(self.title))
        self.numbers = title_numbers(self.title)
        self.erratum = bool(tokens & ERRATUM_WORDS)
        self.generic = normalize_title(self.title).replace(" ", "") in GENERIC_TITLES
        self.authors = author_keys(split_authors(record.get("authors")))
        self.year = _year(record)
        extra = _extra(record)
        self.journal_ref = str(extra.get("journal_ref") or "")
        self.venue = str(extra.get("container_title") or extra.get("venue") or "")
        doi = normalize_doi(record.get("doi", ""))
        self.published_doi = doi if doi and not is_preprint(record) else ""


def veto(a: _Features, b: _Features) -> Optional[str]:
    """Hard reasons two records can never be the same work."""
    if a.numbers != b.numbers:
        return f"title numbers/parts differ ({sorted(a.numbers)} vs {sorted(b.numbers)})"
    if a.erratum != b.erratum:
        return "one record is an erratum/corrigendum/retraction notice"
    if a.authors and b.authors and author_overlap(a.authors, b.authors) == 0:
        return "author lists are disjoint"
    if a.year and b.year and abs(a.year - b.year) > MAX_YEAR_GAP:
        return f"year gap {abs(a.year - b.year)} exceeds {MAX_YEAR_GAP}"
    # Two records each claiming their own published DOI are two works. Preprint
    # DOIs never trigger this, so a preprint still links to its published version.
    if a.published_doi and b.published_doi and a.published_doi != b.published_doi:
        return f"different published DOIs ({a.published_doi} vs {b.published_doi})"
    return None


def _journal_ref_matches(preprint: _Features, other: _Features) -> bool:
    if not preprint.journal_ref:
        return False
    if other.year and str(other.year) in preprint.journal_ref:
        return True
    ref_tokens = {t for t in title_tokens(preprint.journal_ref) if len(t) >= 3}
    return bool(ref_tokens & {t for t in title_tokens(other.venue) if len(t) >= 3})


FEATURES = (
    "title_dice", "title_token_jaccard", "author_overlap", "first_author_match",
    "year_gap", "journal_ref_match", "venue_overlap", "title_length_ratio",
)


def pair_features(a: _Features, b: _Features) -> List[float]:
    """Pairwise features for the match classifier, each in [0, 1], ordered as FEATURES."""
    tokens_a, tokens_b = set(title_tokens(a.title)), set(title_tokens(b.title))
    compact_a = len(normalize_title(a.title).replace(" ", ""))
    compact_b = len(normalize_title(b.title).replace(" ", ""))
    venue_a = {t for t in title_tokens(a.venue or a.journal_ref) if len(t) >= 3}
    venue_b = {t for t in title_tokens(b.venue or b.journal_ref) if len(t) >= 3}
    return [
        title_similarity(a.title, b.title),
        len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a | tokens_b else 0.0,
        author_overlap(a.authors, b.authors),
        float(bool(a.authors and b.authors and author_overlap(a.authors[:1], b.authors[:1]) == 1.0)),
        min(abs(a.year - b.year), MAX_YEAR_GAP) / MAX_YEAR_GAP if a.year and b.year else 0.0,
        float(_journal_ref_matches(a, b) or _journal_ref_matches(b, a)),
        float(bool(venue_a & venue_b)),
        min(compact_a, compact_b) / max(compact_a, compact_b, 1),
    ]


def _load_model() -> Optional[Dict[str, Any]]:
    try:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None  # only while benchmarks/dedup/train.py builds the first model
    if model.get("features") != list(FEATURES):
        # Loads as missing so train.py can rebuild it; scoring raises until then.
        logging.getLogger(__name__).warning("%s was trained on other features; retrain it", MODEL_PATH.name)
        return None
    return model


MODEL = _load_model()
MATCH_THRESHOLD = MODEL["threshold"] if MODEL else 1.0


def match_probability(features: List[float], model: Optional[Dict[str, Any]] = None) -> float:
    model = model or MODEL
    if model is None:
        raise RuntimeError(f"{MODEL_PATH.name} is missing; run python -m benchmarks.dedup.train")
    z = model["bias"] + sum(w * x for w, x in zip(model["weights"], features))
    return 1.0 / (1.0 + math.exp(-z))


def score_pair(a: _Features, b: _Features) -> Tuple[float, str]:
    """Return (match probability, reason) for records without a shared identifier."""
    if a.generic or b.generic:
        return 0.0, "generic title; identifier evidence required"
    if not a.authors or not b.authors:
        return 0.0, "author list missing; identifier evidence required"
    x = pair_features(a, b)
    probability = match_probability(x)
    reason = f"classifier p={probability:.2f}: title similarity {x[0]:.2f}, author overlap {x[2]:.2f}"
    if x[5]:
        reason += ", arXiv journal-ref matches venue/year"
    return probability, reason


# ---------------------------------------------------------------------------
# Union-find with cannot-link constraints
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.members = {i: [i] for i in range(size)}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if len(self.members[ri]) < len(self.members[rj]):
            ri, rj = rj, ri
        self.parent[rj] = ri
        self.members[ri].extend(self.members.pop(rj))


def resolve(
    results: Dict[str, List[Dict[str, Any]]],
    store: Optional["IdentityStore"] = None,
) -> List[Dict[str, Any]]:
    """Merge per-source result lists into works, ranked by reciprocal rank fusion.

    ``results`` maps source name to that source's records in its own rank order.
    """
    records: List[Dict[str, Any]] = []
    ranks: List[Tuple[str, int]] = []
    for source, papers in results.items():
        for rank, paper in enumerate(papers, start=1):
            paper = dict(paper)
            if not paper.get("source"):
                paper["source"] = source
            records.append(paper)
            ranks.append((source, rank))
    if not records:
        return []

    keys = [record_key(r) for r in records]
    identifiers = [record_identifiers(r) for r in records]
    stored = store.lookup({k for ids in identifiers for k, _, _ in ids}) if store else {}
    for ids in identifiers:
        for upid in sorted({stored[k] for k, _, _ in ids if k in stored}):
            ids.append((upid, "persisted crosswalk", False))

    # Edge kinds: 0 shared identifier (never vetoed), 1 persisted crosswalk UPID
    # (vetoed, so a stored wrong merge splits), 2 pairwise evidence (vetoed).
    edges: List[Tuple[int, float, str, str, int, int, str]] = []
    holders: Dict[str, List[Tuple[int, str]]] = {}
    for i, ids in enumerate(identifiers):
        for key, reason, _primary in ids:
            persisted = key.startswith("upid:")
            # A vetoed persisted link to one holder must not hide a valid link to another.
            for j, other_reason in holders.get(key, [])[: None if persisted else 1]:
                if j != i:
                    edges.append((1 if persisted else 0, -1.0, *sorted((keys[i], keys[j])), i, j,
                                  f"shared identifier {key} ({other_reason} / {reason})"))
            holders.setdefault(key, []).append((i, reason))

    # Evidence edges: all pairs. shortcut: O(n^2) pairs, fine for ~100 records per query;
    # add title-bigram blocking if per-query volume grows to thousands.
    features = [_Features(r) for r in records]
    rejected: List[Tuple[str, str, float, str]] = []
    vetoes: Dict[Tuple[int, int], Optional[str]] = {}
    # Vetoes stored by earlier searches stay in force: a pair split once on fuller
    # metadata is not re-merged when a later source omits that metadata.
    stored_vetoes = store.vetoes(keys) if store else {}

    def pair_veto(i: int, j: int) -> Optional[str]:
        pair = (min(i, j), max(i, j))
        if pair not in vetoes:
            stored_reason = stored_vetoes.get(tuple(sorted((keys[i], keys[j]))))
            vetoes[pair] = f"stored veto: {stored_reason}" if stored_reason else veto(features[i], features[j])
        return vetoes[pair]

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            confidence, reason = score_pair(features[i], features[j])
            reason_veto = pair_veto(i, j)
            if confidence >= MATCH_THRESHOLD and not reason_veto:
                edges.append((2, -confidence, *sorted((keys[i], keys[j])), i, j, reason))
            elif reason_veto and title_similarity(features[i].title, features[j].title) >= 0.6:
                rejected.append((*sorted((keys[i], keys[j])), confidence, f"vetoed: {reason_veto}"))

    # Identifier edges first, then evidence by confidence; ties break on record
    # keys, so the result does not depend on the order sources returned records.
    edges.sort(key=lambda e: e[:4])
    uf = _UnionFind(len(records))
    accepted: List[Tuple[int, int, float, str]] = []
    for kind, neg_conf, key_a, key_b, i, j, reason in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if kind > 0:
            blocking = next(
                ((x, y) for x in uf.members[ri] for y in uf.members[rj] if pair_veto(x, y)), None,
            )
            if blocking:
                x, y = blocking
                # Record the pair that actually conflicts, not the edge's endpoints.
                rejected.append((*sorted((keys[x], keys[y])), -neg_conf, f"vetoed: {pair_veto(x, y)}"))
                continue
        uf.union(i, j)
        accepted.append((i, j, -neg_conf if kind == 2 else 1.0, reason))

    clusters: Dict[int, List[int]] = {}
    for i in range(len(records)):
        clusters.setdefault(uf.find(i), []).append(i)
    cluster_edges: Dict[int, List[Tuple[int, int, float, str]]] = {}
    for edge in accepted:
        cluster_edges.setdefault(uf.find(edge[0]), []).append(edge)

    works = []
    for root, members in clusters.items():
        members.sort(key=lambda i: keys[i])
        work = _build_work(
            [records[i] for i in members],
            [ranks[i] for i in members],
            [
                {"a": keys[i], "b": keys[j], "confidence": round(c, 4), "reason": r}
                for i, j, c, r in cluster_edges.get(root, [])
            ],
        )
        work["_identifiers"] = sorted(
            {(k, keys[i]) for i in members for k, _, _ in identifiers[i] if not k.startswith("upid:")}
        )
        work["_stored_upids"] = sorted({k for i in members for k, _, _ in identifiers[i] if k.startswith("upid:")})
        works.append(work)

    if store:
        store.commit(works, rejected)
    else:
        for work in works:
            work["upid"] = work["_stored_upids"][0] if work["_stored_upids"] else f"upid:{uuid.uuid4().hex}"
    # RRF ties break on best single-source rank, then record key: deterministic.
    works.sort(key=lambda w: (
        -w["rrf_score"], -len(w["sources"]), min(w["source_ranks"].values()), w["_identifiers"][0][1],
    ))
    for work in works:
        work.pop("_identifiers")
        work.pop("_stored_upids")
    return works


def _build_work(
    members: List[Dict[str, Any]],
    ranks: List[Tuple[str, int]],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Published versions supply bibliographic fields; preprints follow.
    order = sorted(range(len(members)), key=lambda i: (is_preprint(members[i]), ranks[i][1]))
    provenance: Dict[str, str] = {}

    def pick(field: str, choose=None) -> Any:
        candidates = [i for i in order if members[i].get(field)]
        if not candidates:
            return "" if field != "citations" else 0
        best = choose(candidates) if choose else candidates[0]
        provenance[field] = members[best]["source"]
        return members[best][field]

    claims: Dict[str, Dict[str, List[str]]] = {}
    for record in members:
        for id_type, value in _primary_values(record):
            claims.setdefault(id_type, {}).setdefault(value, [])
            if record["source"] not in claims[id_type][value]:
                claims[id_type][value].append(record["source"])
    ids: Dict[str, Any] = {}
    quarantined: Dict[str, List[Dict[str, Any]]] = {}
    for id_type, values in claims.items():
        if id_type in SINGLE_VALUED and len(values) > 1:
            quarantined[id_type] = [{"value": v, "sources": s} for v, s in sorted(values.items())]
        else:
            ids[id_type] = sorted(values)[0] if len(values) == 1 else sorted(values)

    best_rank: Dict[str, int] = {}
    for source, rank in ranks:
        best_rank[source] = min(rank, best_rank.get(source, rank))

    primary = members[order[0]]
    work = {
        "upid": "",
        # The record that leads the work, kept for callers of the old flat format.
        "source": primary["source"],
        "paper_id": primary.get("paper_id", ""),
        "title": pick("title"),
        "authors": pick("authors", lambda c: max(c, key=lambda i: len(split_authors(members[i]["authors"])))),
        "abstract": pick("abstract", lambda c: max(c, key=lambda i: len(members[i]["abstract"]))),
        "doi": ids.get("doi") if isinstance(ids.get("doi"), str) else "",
        "published_date": pick("published_date"),
        "url": pick("url"),
        "pdf_url": pick("pdf_url"),
        "citations": pick("citations", lambda c: max(c, key=lambda i: _int(members[i]["citations"]))),
        "ids": ids,
        "quarantined_ids": quarantined,
        "sources": sorted(best_rank),
        "source_ranks": best_rank,
        "rrf_score": round(sum(1.0 / (RRF_K + r) for r in best_rank.values()), 6),
        "provenance": provenance,
        "pdf_urls": [{"url": r["pdf_url"], "source": r["source"]} for r in members if r.get("pdf_url")],
        "records": [
            {"source": r["source"], "paper_id": r.get("paper_id", ""), "url": r.get("url", ""),
             "is_preprint": is_preprint(r)}
            for r in members
        ],
        "merge_evidence": evidence,
    }
    if work["doi"]:
        provenance["doi"] = ",".join(claims["doi"][work["doi"]])
    return work


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Persistent crosswalk
# ---------------------------------------------------------------------------

class IdentityStore:
    """SQLite crosswalk: identifier -> UPID, redirects, merge evidence, quarantine."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS crosswalk (
        identifier TEXT NOT NULL, record TEXT NOT NULL, upid TEXT NOT NULL,
        PRIMARY KEY (identifier, record));
    CREATE TABLE IF NOT EXISTS redirects (old_upid TEXT PRIMARY KEY, new_upid TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS splits (
        old_upid TEXT NOT NULL, new_upid TEXT NOT NULL, PRIMARY KEY (old_upid, new_upid));
    CREATE TABLE IF NOT EXISTS edges (
        a TEXT NOT NULL, b TEXT NOT NULL, upid TEXT NOT NULL, merged INTEGER NOT NULL,
        confidence REAL NOT NULL, reason TEXT NOT NULL, PRIMARY KEY (a, b));
    CREATE TABLE IF NOT EXISTS quarantine (
        upid TEXT NOT NULL, id_type TEXT NOT NULL, value TEXT NOT NULL, sources TEXT NOT NULL,
        PRIMARY KEY (upid, id_type, value));
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.executescript(self.SCHEMA)
        return connection

    @staticmethod
    def _follow(connection: sqlite3.Connection, upid: str) -> str:
        seen = set()
        while upid not in seen:
            seen.add(upid)
            row = connection.execute("SELECT new_upid FROM redirects WHERE old_upid=?", (upid,)).fetchone()
            if not row:
                return upid
            upid = row[0]
        return upid

    def lookup(self, identifiers: Iterable[str]) -> Dict[str, str]:
        with closing(self._connect()) as connection:
            return self._lookup(connection, list(identifiers))

    def _lookup(self, connection: sqlite3.Connection, identifiers: List[str]) -> Dict[str, str]:
        found = {}
        for start in range(0, len(identifiers), 500):
            chunk = identifiers[start:start + 500]
            rows = connection.execute(
                f"SELECT identifier, upid FROM crosswalk WHERE identifier IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            found.update({ident: self._follow(connection, upid) for ident, upid in rows})
        return found

    def commit(self, works: List[Dict[str, Any]], rejected: List[Tuple[str, str, float, str]]) -> None:
        with closing(self._connect()) as connection, connection:
            # Re-read under the write lock: a concurrent search may have minted a
            # UPID for these identifiers since resolve() looked them up.
            connection.execute("BEGIN IMMEDIATE")
            claims: List[Dict[str, int]] = []
            for work in works:
                identifiers = sorted({ident for ident, _ in work["_identifiers"]})
                current = self._lookup(connection, identifiers)
                counts = {self._follow(connection, u): 0 for u in work["_stored_upids"]}
                for ident in identifiers:
                    if ident in current:
                        counts[current[ident]] = counts.get(current[ident], 0) + 1
                claims.append(counts)
            # A stored UPID claimed by several works was split by a veto: the work
            # holding most of its identifiers keeps it (ties: smallest record key).
            owner: Dict[str, int] = {}
            for index, counts in enumerate(claims):
                for upid, count in counts.items():
                    best = owner.get(upid)
                    if best is None or (-count, works[index]["_identifiers"][0][1]) < (
                        -claims[best][upid], works[best]["_identifiers"][0][1]
                    ):
                        owner[upid] = index

            record_upid: Dict[str, str] = {}
            for index, work in enumerate(works):
                mine = sorted(u for u in claims[index] if owner[u] == index)
                upid = mine[0] if mine else f"upid:{uuid.uuid4().hex}"
                for old in mine[1:]:
                    connection.execute("INSERT OR REPLACE INTO redirects VALUES (?, ?)", (old, upid))
                    connection.execute("UPDATE crosswalk SET upid=? WHERE upid=?", (upid, old))
                    connection.execute("UPDATE edges SET upid=? WHERE upid=?", (upid, old))
                    connection.execute("UPDATE OR IGNORE quarantine SET upid=? WHERE upid=?", (upid, old))
                for old in sorted(u for u in claims[index] if owner[u] != index):
                    connection.execute("INSERT OR IGNORE INTO splits VALUES (?, ?)", (old, upid))
                work["upid"] = upid
                record_upid.update({record: upid for _, record in work["_identifiers"]})
                connection.executemany(
                    "INSERT OR REPLACE INTO crosswalk VALUES (?, ?, ?)",
                    [(ident, record, upid) for ident, record in work["_identifiers"]],
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, 1, ?, ?)",
                    [(e["a"], e["b"], upid, e["confidence"], e["reason"]) for e in work["merge_evidence"]],
                )
                for id_type, entries in work["quarantined_ids"].items():
                    connection.executemany(
                        "INSERT OR REPLACE INTO quarantine VALUES (?, ?, ?, ?)",
                        [(upid, id_type, e["value"], json.dumps(e["sources"])) for e in entries],
                    )

            # Stored merges between records that now resolve to different works are
            # demoted, keeping the veto as the reason; the rest follow their work.
            veto_reason = {(a, b): r for a, b, _c, r in rejected}
            touched = sorted({u for counts in claims for u in counts})
            stored_edges = connection.execute(
                f"SELECT a, b, upid, reason FROM edges WHERE merged=1 AND upid IN ({','.join('?' * len(touched))})",
                touched,
            ).fetchall() if touched else []
            for a, b, upid, reason in stored_edges:
                if a not in record_upid or b not in record_upid:
                    continue
                if record_upid[a] == record_upid[b]:
                    connection.execute("UPDATE edges SET upid=? WHERE a=? AND b=?", (record_upid[a], a, b))
                else:
                    why = veto_reason.get((a, b), "records now resolve to different works")
                    connection.execute(
                        "UPDATE edges SET merged=0, upid='', reason=? WHERE a=? AND b=?",
                        (f"split ({why}); previously merged: {reason}", a, b),
                    )
            # A veto never overwrites a merge already recorded for the pair.
            connection.executemany(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, '', 0, ?, ?)",
                [(a, b, round(c, 4), r) for a, b, c, r in rejected],
            )

    def vetoes(self, record_keys: Iterable[str]) -> Dict[Tuple[str, str], str]:
        """Stored pairwise vetoes (rejected or split pairs) among these records."""
        wanted = set(record_keys)
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT a, b, reason FROM edges WHERE merged=0").fetchall()
        # shortcut: scans all stored vetoes per search; index by record if the store grows large.
        return {(a, b): reason for a, b, reason in rows if a in wanted and b in wanted}

    def _upid_of(self, connection: sqlite3.Connection, identifier: str) -> Tuple[str, List[str]]:
        """Resolve a DOI, arXiv id, prefixed identifier or UPID to (upid, record keys)."""
        identifier = canonical_identifier(identifier)
        if identifier.startswith("upid:"):
            upid = self._follow(connection, identifier)
            rows = connection.execute("SELECT DISTINCT record FROM crosswalk WHERE upid=?", (upid,)).fetchall()
            return (upid, [r[0] for r in rows]) if rows else ("", [])
        rows = connection.execute(
            "SELECT record, upid FROM crosswalk WHERE identifier=?", (identifier,)
        ).fetchall()
        if not rows:
            return "", []
        return self._follow(connection, rows[0][1]), sorted({r[0] for r in rows})

    def describe(self, identifier: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            upid, _records = self._upid_of(connection, identifier)
            if not upid:
                return {"identifier": identifier, "known": False}
            identifiers = [r[0] for r in connection.execute(
                "SELECT DISTINCT identifier FROM crosswalk WHERE upid=? ORDER BY identifier", (upid,))]
            redirected_from = [r[0] for r in connection.execute(
                "SELECT old_upid FROM redirects WHERE new_upid=?", (upid,))]
            split_from = [r[0] for r in connection.execute(
                "SELECT old_upid FROM splits WHERE new_upid=?", (upid,))]
            split_into = [r[0] for r in connection.execute(
                "SELECT new_upid FROM splits WHERE old_upid=?", (upid,))]
            quarantine = [
                {"id_type": t, "value": v, "sources": json.loads(s)}
                for t, v, s in connection.execute(
                    "SELECT id_type, value, sources FROM quarantine WHERE upid=?", (upid,))
            ]
            return {"identifier": identifier, "known": True, "upid": upid,
                    "identifiers": identifiers, "redirected_from": redirected_from,
                    "split_from": split_from, "split_into": split_into,
                    "quarantined_ids": quarantine}

    def explain(self, identifier_a: str, identifier_b: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            upid_a, records_a = self._upid_of(connection, identifier_a)
            upid_b, records_b = self._upid_of(connection, identifier_b)
            result: Dict[str, Any] = {"a": identifier_a, "b": identifier_b, "upid_a": upid_a, "upid_b": upid_b}
            if not upid_a or not upid_b:
                missing = [i for i, u in ((identifier_a, upid_a), (identifier_b, upid_b)) if not u]
                return {**result, "merged": False, "reason": f"not seen in any search yet: {missing}"}
            if upid_a != upid_b:
                keys_a, keys_b = set(records_a), set(records_b)
                rejections = [
                    {"a": a, "b": b, "confidence": c, "reason": r}
                    for a, b, c, r in connection.execute(
                        "SELECT a, b, confidence, reason FROM edges WHERE merged=0")
                    if (a in keys_a and b in keys_b) or (a in keys_b and b in keys_a)
                ]
                return {**result, "merged": False, "confidence": 0.0,
                        "reason": "different works" if not rejections else rejections[0]["reason"],
                        "rejected_candidates": rejections}
            edges = connection.execute(
                "SELECT a, b, confidence, reason FROM edges WHERE upid=? AND merged=1", (upid_a,)
            ).fetchall()
            path = _path(edges, records_a, records_b)
            confidence = min((e["confidence"] for e in path), default=1.0)
            return {**result, "merged": True, "upid": upid_a, "confidence": confidence,
                    "reason": "; ".join(e["reason"] for e in path) or "same source record",
                    "evidence": path}


def _path(edges, sources: List[str], targets: List[str]) -> List[Dict[str, Any]]:
    """Shortest chain of stored merge edges linking any source record to any target."""
    graph: Dict[str, List[Tuple[str, Tuple]]] = {}
    for edge in edges:
        graph.setdefault(edge[0], []).append((edge[1], edge))
        graph.setdefault(edge[1], []).append((edge[0], edge))
    queue = deque((s, []) for s in sources)
    seen = set(sources)
    while queue:
        node, trail = queue.popleft()
        if node in targets:
            return [{"a": a, "b": b, "confidence": c, "reason": r} for a, b, c, r in trail]
        for nxt, edge in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trail + [edge]))
    return []


def canonical_identifier(value: str) -> str:
    """Accept '10.x/y', 'https://doi.org/..', 'arXiv:1512.03385v2', 'pmid:1', 'upid:..'."""
    value = (value or "").strip()
    lowered = value.casefold()
    if normalize_doi(value):
        return f"doi:{normalize_doi(value)}"
    for prefix in ("upid:", "src:", "pmid:", "openalex:", "s2:"):
        if lowered.startswith(prefix):
            return lowered
    if lowered.startswith("arxiv:") or re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", lowered):
        return f"arxiv:{normalize_arxiv(lowered)}"
    if lowered.startswith("doi:"):
        return f"doi:{normalize_doi(lowered[4:])}"
    return lowered
