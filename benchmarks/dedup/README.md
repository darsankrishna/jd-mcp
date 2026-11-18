# Cross-source paper deduplication benchmark

A labelled gold set of record groups, and a runner that scores each system's
merge logic pair by pair (precision, recall, F1). Results: [RESULTS.md](RESULTS.md).

## Data (`data/`)

| File | Cases | Split | What it is |
| --- | --- | --- | --- |
| `handbuilt.json` | 9 | test | The cases from the base-repo analysis (preprint vs journal, ResNet DOIs, punctuation, initials, comma names, DOI prefix, "Introduction", Part I/II, three-record chain) |
| `generated.jsonl` | 588 | train / test | 84 synthetic seed works (4 non-Latin), each with 4 positive and 3 negative perturbation cases |

Each case holds `records` (in `Paper.to_dict()` form) and gold `clusters`
(lists of record indices). Generated seeds with even numbers form the train
split and odd ones the test split, so no work is in both.

Categories. Positive (same work): `doi_prefix`, `punctuation_case`,
`author_comma`, `author_initials`, `arxiv_doi_field`, `arxiv_doi_vs_venue_doi`,
`journal_ref_reworded`, `crossref_relation`, `chain_three`, `accents_dropped`,
`hyphenation`. Negative (different works): `part_numbers`, `erratum`,
`different_authors`, `year_gap`, `generic_title`, `sibling_task`,
`one_word_method`, `numbered_sequel`.

**Limits.** Every record is constructed: titles come from word lists, DOIs use
the `10.5555` test prefix, and labels hold by construction. The same person
wrote the generator and the resolver around the same failure modes, so jd-mcp's
score here shows that those modes are handled, not how it performs on live data.
To make this a real benchmark, add labelled pairs drawn from live provider
responses (as separate files with their own split) and rerun.

## Systems (`adapters.py`)

- **jd-mcp**: `paper_search_mcp.identity.resolve`, without a persistent store.
- **paper-search-mcp@cab1b5c**: `_paper_unique_key` / `_dedupe_papers`, copied verbatim (MIT).
- **scholarmcp@0d34ff5 (port)**: line-for-line Python port of the merge loop in
  `packages/scholar-mcp/src/research/literature-service.ts` (MIT). The original
  is not importable on its own. The port reproduces the analysis results
  (Part I/II and "Introduction" wrongly merged, three-record chain 3 → 2), and
  `tests/test_dedup_benchmark.py` checks that.
- **Liyux3/scholar-mcp** (opt-in): `--liyux3-path <checkout>` imports its
  `scholar_mcp.records.deduplicate`. No code is copied. It is licensed PolyForm
  Noncommercial with additional terms, including a "no competing products"
  clause. Read `LICENSE` and `LICENSE-ADDITIONAL.md` in that repo and decide
  whether your use is permitted before running it.

## Commands

```bash
uv run python -m benchmarks.dedup.generate     # rewrite data/generated.jsonl
uv run python -m benchmarks.dedup.handbuilt    # rewrite data/handbuilt.json
uv run python -m benchmarks.dedup.train        # retrain paper_search_mcp/match_model.json
uv run python -m benchmarks.dedup.run --write  # score the test split, write RESULTS.md
```

## Classifier (`train.py`)

Logistic regression over 8 pairwise features (`identity.FEATURES`): title
bigram Dice, title token Jaccard, author overlap, first-author match, year gap,
arXiv journal-ref match, venue overlap and title length ratio. It is trained by
projected gradient descent, standard library only. Weights are sign-constrained
so more agreement can never lower the probability and a larger year gap can
never raise it, classes are balanced, and the threshold is chosen by Matthews
correlation on the train split.

It trains only on the pairs it decides in `resolve`: both author lists present,
no generic title, no hard veto, and no identifier link, direct or through a chain.
