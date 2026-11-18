"""Capture live connector records for the seed queries and label them.

    uv run python -m benchmarks.dedup.live_capture            # capture + label + review sheet
    uv run python -m benchmarks.dedup.live_capture --relabel  # label + review sheet only
    uv run python -m benchmarks.dedup.live_capture --sources openalex  # recapture one source

Records are this repo's connectors' live ``Paper.to_dict()`` output (abstracts
trimmed). Labels come from Semantic Scholar, which is never an evaluated source:
each record's own DOI, arXiv id or PMID is looked up in the S2 batch API, and
two records are the same work when they resolve to the same S2 paperId.
Records with no resolvable id, or ids resolving to different S2 papers, stay
unlabelled. ``data/live/review.csv`` lists the pairs a person should check.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from paper_search_mcp.identity import record_identifiers, record_key, title_similarity

from .generate import DATA

# dev: used while fixing bugs found in live data. test: held out, scored once.
SETS = {
    "dev": (DATA / "live", Path(__file__).with_name("live_seeds.json")),
    "test": (DATA / "live-test", Path(__file__).with_name("live_seeds_test.json")),
}
LIVE, SEEDS = SETS["dev"]
PER_SOURCE = 3
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_PREFIX = {"doi": "DOI", "arxiv": "ARXIV", "pmid": "PMID"}
REVIEW_FIELDS = ["case_id", "a", "b", "a_source", "b_source", "a_title", "b_title", "a_year", "b_year",
                 "a_authors", "b_authors", "s2_label", "flag", "review"]


def search_with_retry(searcher, query: str):
    for attempt in range(6):
        try:
            return searcher.search(query, max_results=PER_SOURCE)
        except RuntimeError as exc:
            # OpenAlex throttles anonymous search under load and asks for a retry.
            if "HTTP 429" not in str(exc) or attempt == 5:
                raise
            time.sleep(35)


def capture(only: Optional[List[str]] = None) -> None:
    from paper_search_mcp import server  # imports every connector

    config = json.loads(SEEDS.read_text(encoding="utf-8"))
    LIVE.mkdir(parents=True, exist_ok=True)
    previous = {c["id"]: c for c in load_live()} if only else {}
    with open(LIVE / "records.jsonl.part", "w", encoding="utf-8") as handle:
        for number, seed in enumerate(config["seeds"]):
            case_id = f"live-{number:02d}"
            old = previous.get(case_id, {})
            # Recapturing some sources keeps every other source's saved records.
            records = [r for r in old.get("records", []) if r["source"] not in (only or [])]
            errors = {k: v for k, v in old.get("source_errors", {}).items() if k not in (only or [])}
            seen = {(r["source"], r["paper_id"]) for r in records}
            for source in config["sources"][seed.get("sources", "default")]:
                if only and source not in only:
                    continue
                try:
                    papers = search_with_retry(server.SEARCHERS[source], seed["query"])
                except Exception as exc:  # an outage is recorded, not hidden
                    errors[source] = f"{type(exc).__name__}: {exc}"[:300]
                    continue
                for paper in papers[:PER_SOURCE]:
                    rec = paper.to_dict()
                    rec["source"] = rec.get("source") or source
                    rec["abstract"] = (rec.get("abstract") or "")[:300]
                    if (rec["source"], rec["paper_id"]) not in seen:
                        seen.add((rec["source"], rec["paper_id"]))
                        records.append(rec)
            captured = dict(old.get("captured_at_by_source", {}))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            captured.update({s: now for s in config["sources"][seed.get("sources", "default")]
                             if not only or s in only})
            case = {"id": case_id, "query": seed["query"], "category": seed["category"],
                    "captured_at_by_source": captured, "source_errors": errors, "records": records}
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(f"{case_id} {len(records):3d} records, errors: {sorted(errors) or '-'}  {seed['query'][:60]}",
                  flush=True)
    (LIVE / "records.jsonl.part").replace(LIVE / "records.jsonl")


def load_live() -> List[Dict[str, Any]]:
    with open(LIVE / "records.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def lookup_ids(rec: Dict[str, Any]) -> List[str]:
    """The record's own explicit ids, in S2 batch syntax."""
    ids = []
    for key, _reason, primary in record_identifiers(rec):
        id_type, _, value = key.partition(":")
        if primary and id_type in S2_PREFIX:
            ids.append(f"{S2_PREFIX[id_type]}:{value}")
    return sorted(set(ids))


def s2_batch(ids: List[str]) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        for attempt in range(8):
            response = requests.post(S2_BATCH, params={"fields": "paperId"}, json={"ids": chunk}, timeout=60)
            if response.status_code != 429:
                break
            time.sleep(min(60, 5 * 2 ** attempt))  # unauthenticated S2 is rate limited
        response.raise_for_status()
        for ident, item in zip(chunk, response.json()):
            resolved[ident] = item.get("paperId") if item else None
        time.sleep(3)
    return resolved


def label(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    wanted = sorted({i for case in cases for rec in case["records"] for i in lookup_ids(rec)})
    resolved = s2_batch(wanted)
    labels = {}
    for case in cases:
        for rec in case["records"]:
            ids = lookup_ids(rec)
            papers = sorted({resolved[i] for i in ids if resolved.get(i)})
            status = "labelled" if len(papers) == 1 else ("no_id" if not ids else
                                                         "not_in_s2" if not papers else "conflict")
            labels[f"{case['id']}|{record_key(rec)}"] = {
                "s2_paper_id": papers[0] if len(papers) == 1 else None,
                "status": status, "ids": ids, "resolved": {i: resolved.get(i) for i in ids},
            }
    (LIVE / "labels.json").write_text(json.dumps({
        "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authority": "Semantic Scholar Graph API /paper/batch (paperId)",
        "records": labels,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return labels


def review_sheet(cases: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> int:
    """Pairs to check by hand, chosen without looking at any system's output."""
    rng = random.Random(11)
    rows = []
    for case in cases:
        recs = case["records"]
        for i, j in combinations(range(len(recs)), 2):
            a, b = recs[i], recs[j]
            la = labels[f"{case['id']}|{record_key(a)}"]["s2_paper_id"]
            lb = labels[f"{case['id']}|{record_key(b)}"]["s2_paper_id"]
            similarity = title_similarity(a.get("title", ""), b.get("title", ""))
            s2_label = "unlabelled" if not (la and lb) else ("same" if la == lb else "different")
            if s2_label == "unlabelled":
                flag = "unlabelled, similar titles" if similarity >= 0.5 else ""
            elif s2_label == "different" and similarity >= 0.8:
                flag = "S2 says different, titles near-identical"
            elif s2_label == "same" and similarity < 0.5:
                flag = "S2 says same, titles dissimilar"
            else:
                flag = "random 10% audit sample" if rng.random() < 0.10 else ""
            if flag:
                rows.append({
                    "case_id": case["id"], "a": record_key(a), "b": record_key(b),
                    "a_source": a["source"], "b_source": b["source"],
                    "a_title": a.get("title", ""), "b_title": b.get("title", ""),
                    "a_year": str(a.get("published_date", ""))[:4], "b_year": str(b.get("published_date", ""))[:4],
                    "a_authors": str(a.get("authors", ""))[:120], "b_authors": str(b.get("authors", ""))[:120],
                    "s2_label": s2_label, "flag": flag, "review": "",
                })
    path = LIVE / "review.csv"
    previous = {}
    if path.exists():  # never discard reviews already entered
        with open(path, encoding="utf-8", newline="") as handle:
            previous = {(r["case_id"], r["a"], r["b"]): r["review"] for r in csv.DictReader(handle) if r["review"]}
    for row in rows:
        row["review"] = previous.get((row["case_id"], row["a"], row["b"]), "")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel", action="store_true", help="Skip capture; relabel saved records")
    parser.add_argument("--sources", help="Comma-separated sources to recapture into the saved records")
    parser.add_argument("--set", choices=sorted(SETS), default="dev", help="Live set (default: dev)")
    args = parser.parse_args(argv)
    global LIVE, SEEDS
    LIVE, SEEDS = SETS[args.set]
    if args.sources:
        capture([s.strip() for s in args.sources.split(",") if s.strip()])
    elif not args.relabel:
        capture()
    cases = load_live()
    labels = label(cases)
    counts: Dict[str, int] = {}
    for value in labels.values():
        counts[value["status"]] = counts.get(value["status"], 0) + 1
    print("record labels:", counts)
    print("review rows:", review_sheet(cases, labels))


if __name__ == "__main__":
    main()
