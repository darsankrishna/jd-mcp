"""Score each system's merges against the gold clusters, pair by pair.

    uv run python -m benchmarks.dedup.run [--split test|train|all|live] [--liyux3-path PATH] [--write]

A pair is positive when both records belong to one gold cluster. Precision,
recall and F1 count record pairs within each case; cases never interact.
The live split scores only pairs whose records both carry a Semantic Scholar
label; a filled-in review.csv row overrides it (same/different) or drops it (unsure).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .adapters import SYSTEMS, liyux3_loader
from .generate import DATA

RESULTS = Path(__file__).with_name("RESULTS.md")
LIVE_RESULTS = Path(__file__).with_name("RESULTS-live.md")
LIVE_TEST_RESULTS = Path(__file__).with_name("RESULTS-live-test.md")


def load_live_cases(folder: str = "live") -> List[Dict[str, Any]]:
    from paper_search_mcp.identity import record_key

    live = DATA / folder
    labels = json.loads((live / "labels.json").read_text(encoding="utf-8"))["records"]
    reviews: Dict[tuple, str] = {}
    if (live / "review.csv").exists():
        with open(live / "review.csv", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["review"].strip():
                    reviews[(row["case_id"], row["a"], row["b"])] = row["review"].strip().lower()
    cases = []
    with open(live / "records.jsonl", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            keys = [record_key(r) for r in case["records"]]
            case["gold"] = [labels[f"{case['id']}|{k}"]["s2_paper_id"] for k in keys]
            case["review"] = {(i, j): reviews[(case["id"], keys[i], keys[j])]
                              for i, j in combinations(range(len(keys)), 2)
                              if (case["id"], keys[i], keys[j]) in reviews}
            cases.append(case)
    return cases


def load_cases(split: str = "test") -> List[Dict[str, Any]]:
    if split in ("live", "live-test"):
        return load_live_cases(split)
    cases = json.loads((DATA / "handbuilt.json").read_text(encoding="utf-8"))
    with open(DATA / "generated.jsonl", encoding="utf-8") as handle:
        cases += [json.loads(line) for line in handle]
    return [c for c in cases if split == "all" or c["split"] == split]


def gold_pairs(case: Dict[str, Any]) -> Iterable[tuple]:
    if "gold" in case:  # live: partial labels plus human review
        for i, j in combinations(range(len(case["records"])), 2):
            review = case["review"].get((i, j))
            if review in ("same", "different"):
                yield i, j, review == "same"
            elif review is None and case["gold"][i] and case["gold"][j]:
                yield i, j, case["gold"][i] == case["gold"][j]
        return
    cluster_of = {i: n for n, cluster in enumerate(case["clusters"]) for i in cluster}
    for i, j in combinations(range(len(case["records"])), 2):
        yield i, j, cluster_of[i] == cluster_of[j]


def score(system: Callable, cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for case in cases:
        labels = system(case["records"])
        for i, j, same in gold_pairs(case):
            predicted = labels[i] == labels[j]
            cell = ("tp" if same else "fp") if predicted else ("fn" if same else "tn")
            for bucket in ("all", case["category"]):
                counts[bucket][cell] += 1
    return {bucket: metrics(c) for bucket, c in counts.items()}


def metrics(c: Dict[str, int]) -> Dict[str, float]:
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**c, "precision": precision, "recall": recall, "f1": f1}


def report(results: Dict[str, Dict[str, Dict[str, float]]], split: str, n_cases: int) -> str:
    lines = [f"## Split: {split} ({n_cases} cases)", "",
             "| System | Precision | Recall | F1 | TP | FP | FN | TN |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, by_bucket in results.items():
        m = by_bucket["all"]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                     f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
    categories = sorted({b for r in results.values() for b in r} - {"all"})
    names = list(results)
    lines += ["", "Per category: correct pairs / pairs.", "",
              "| Category | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for category in categories:
        row = []
        for name in names:
            m = results[name].get(category)
            row.append(f"{m['tp'] + m['tn']}/{m['tp'] + m['tn'] + m['fp'] + m['fn']}" if m else "-")
        lines.append(f"| {category} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "train", "all", "live", "live-test"), default="test")
    parser.add_argument("--liyux3-path", help="Checkout of Liyux3/scholar-mcp; check its licence terms first")
    parser.add_argument("--write", action="store_true", help=f"Write the report to {RESULTS.name}")
    args = parser.parse_args(argv)

    systems = dict(SYSTEMS)
    if args.liyux3_path:
        systems["Liyux3/scholar-mcp"] = liyux3_loader(args.liyux3_path)
    cases = load_cases(args.split)
    text = report({name: score(system, cases) for name, system in systems.items()}, args.split, len(cases))
    print(text)
    if args.write and args.split.startswith("live"):
        reviewed = sum(len(c["review"]) for c in cases)
        path = LIVE_RESULTS if args.split == "live" else LIVE_TEST_RESULTS
        header = LIVE_HEADER if args.split == "live" else LIVE_TEST_HEADER
        path.write_text(header.format(reviewed=reviewed) + text, encoding="utf-8")
    elif args.write:
        RESULTS.write_text(HEADER + text, encoding="utf-8")


HEADER = """# Deduplication benchmark results

Generated by `uv run python -m benchmarks.dedup.run --write`. All records are
constructed (see README.md): these numbers measure the case types in the gold
set, not live provider data. jd-mcp's classifier was trained on the train split
only; this report uses the held-out test split plus the hand-built cases.

The set is saturated for jd-mcp: its generator and resolver were written around
the same failure modes, so a perfect score here is expected and is not evidence
of live-data accuracy. The baselines' per-category rows show which of those
failure modes each system has.

"""

LIVE_HEADER = """# Deduplication benchmark results: live records (dev set)

Generated by `uv run python -m benchmarks.dedup.run --split live --write`.

**This is a development set.** Its results were used to find and fix two
resolver bugs: PubMed's "Love MI" author order, and two one-word-apart papers
that each have their own published DOI. Its scores are therefore biased in
jd-mcp's favour. The unbiased estimate is RESULTS-live-test.md.

Records are this repo's connectors' live responses (arXiv, Crossref, OpenAlex,
dblp, Europe PMC, PubMed) for the queries in `live_seeds.json`, captured on the
date in `data/live/records.jsonl`. A pair is scored when both records resolve to
a Semantic Scholar paperId through their own DOI, arXiv id or PMID (same id
means same work), or when a reviewer labelled it in `data/live/review.csv`.
Reviewed pairs so far: {reviewed}. Until the flagged pairs are reviewed,
these labels are Semantic Scholar's, and its merge errors count against every
system alike. Records without a resolvable id are only scored once reviewed, so
pairs decided purely by title and author matching are under-represented.

"""

LIVE_TEST_HEADER = LIVE_HEADER.replace(
    "results: live records (dev set)", "results: live records (held-out test set)"
).replace("--split live --write", "--split live-test --write").replace(
    """**This is a development set.** Its results were used to find and fix two
resolver bugs: PubMed's "Love MI" author order, and two one-word-apart papers
that each have their own published DOI. Its scores are therefore biased in
jd-mcp's favour. The unbiased estimate is RESULTS-live-test.md.
""",
    """**This is the held-out set.** Its queries (`live_seeds_test.json`) were chosen
before capture, and the resolver was not changed after seeing these results.
""",
).replace("`live_seeds.json`", "`live_seeds_test.json`").replace("`data/live/", "`data/live-test/")


if __name__ == "__main__":
    main()
