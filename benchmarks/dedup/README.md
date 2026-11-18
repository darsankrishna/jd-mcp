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

## Live records (`data/live/` dev, `data/live-test/` held out)

Real connector responses, scored separately from the constructed set.

| Set | Seeds | Results | Status |
| --- | --- | --- | --- |
| dev | `live_seeds.json` (48 queries) | [RESULTS-live.md](RESULTS-live.md) | Used to find and fix two resolver bugs, so biased in jd-mcp's favour |
| test | `live_seeds_test.json` (43 queries) | [RESULTS-live-test.md](RESULTS-live-test.md) | Queries chosen before capture; resolver hashes frozen in `data/live-test/frozen.json` before capture; scored once |

To keep the test set unbiased, do not change the resolver because of its
results. Fix problems against the dev set, then capture a new test set.

- `live_seeds.json` holds 48 search queries (preprint vs published, biomedical,
  Part/series, errata and retractions, generic titles, non-Latin). It is queries
  only: no identifiers or metadata were typed in.
- `records.jsonl` holds the top 3 results per source from this repo's own
  connectors, with the capture time per source and any source errors.
- `labels.json` is the answer key. Semantic Scholar is never an evaluated source.
  Each record's own DOI, arXiv id or PMID is looked up in the S2 batch API, and
  records resolving to the same S2 paperId are the same work. Records with no
  resolvable id, or ids resolving to different papers, stay unlabelled.
- `review.csv` lists pairs for a person to check, chosen without looking at any
  system's output:
  - unlabelled pairs with similar titles;
  - pairs S2 calls different that have near-identical titles;
  - pairs S2 calls the same that have dissimilar titles;
  - a 10% random audit sample of the rest.

  Fill the `review` column with `same`, `different` or `unsure`. The runner uses
  your label instead of S2's, and re-running the capture keeps reviews already
  entered.

```bash
uv run python -m benchmarks.dedup.live_capture                     # capture, label, review sheet
uv run python -m benchmarks.dedup.live_capture --sources openalex  # recapture one source
uv run python -m benchmarks.dedup.live_capture --relabel           # relabel saved records
uv run python -m benchmarks.dedup.run --split live --write         # score, write RESULTS-live.md
uv run python -m benchmarks.dedup.live_capture --set test          # capture the held-out set
uv run python -m benchmarks.dedup.run --split live-test --write    # score it, write RESULTS-live-test.md
```

Known gaps:
- At capture, dblp served a bot challenge to scripted requests. It is recorded as
  a source error, not worked around.
- Anonymous OpenAlex search was throttled, so its capture waits out each 429.
  Setting `PAPER_SEARCH_MCP_OPENALEX_API_KEY` avoids that.
- S2 labels carry S2's own merge errors until the flagged pairs are reviewed.

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
