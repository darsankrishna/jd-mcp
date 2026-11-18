---
name: paper-search
description: Search, download, and read academic papers from 20+ sources (arXiv, PubMed, Semantic Scholar, CrossRef, etc). Use when the user asks to find papers, search for research, look up academic literature, download a paper PDF, or extract text from a paper.
---

# Paper Search

Search, download, and read academic papers via the `paper-search` CLI.

## CLI Usage

All commands run via:
```bash
paper-search <command> [args]
```

If `paper-search` is not available, install it with `uv tool install paper-search-mcp`. Optional API keys can be configured in `~/.config/paper-search-mcp/.env`.

### Search
```bash
paper-search search "<query>" -n <max_per_source> -s <sources> -y <year>
```
- `-n`: results per source (default: 5)
- `-s`: comma-separated sources, "fast", "fastest", or "all" (default: all)
- `--exhaustive`: compatibility no-op (broad search is already the default); explicit `-s` always takes precedence
- `-y`: year filter for Semantic Scholar (e.g. "2020", "2018-2022")

For speed, use `-s fast` (OpenAlex, Crossref, arXiv, PubMed, Europe PMC; also Semantic Scholar when its API key is nonblank) or `-s fastest` (OpenAlex and Crossref). Explicit source choices are never expanded, even for DOI queries. Use `-s unpaywall` for DOI lookup; "all" already includes it. These presets do not enable optional paid sources.

### Download PDF
```bash
paper-search download <source> <paper_id> [-o ./downloads]
```

### Read (extract text)
```bash
paper-search read <source> <paper_id> [-o ./downloads]
```

### List sources
```bash
paper-search sources
```

### MCP tools from the CLI

`paper-search tool` calls the same eight tools as the MCP server: `search_papers`,
`get_paper`, `explain_match`, `download_paper`, `read_paper`, `extract_sections`,
`get_related_papers`, and `manage_search_cache`. Required arguments are positional;
optional arguments use kebab-case flags. Boolean flags use `--flag` and
`--no-flag`; omitted options retain their MCP defaults.

```bash
paper-search tool --help
paper-search tool --list  # JSON tool list, schemas, and annotations
paper-search tool search_papers --help
paper-search tool search_papers "transformer attention" --sources arxiv,crossref --max-results-per-source 2
paper-search tool get_paper 10.1038/nature12373
paper-search tool explain_match arXiv:1512.03385 10.1109/CVPR.2016.90
```

Objects and lists return JSON; text and download paths return plain text. An
exception returns a JSON error with exit code 1; invalid CLI syntax exits with
code 2. A normal MCP result retains its original meaning, including error or
partial-result fields returned by the tool itself. `--list` shows only tools
enabled by the current configuration, so IEEE tools still require their API
key. The existing `search`, `download`, `read`, and `sources` commands are
unchanged. Tool calls keep the MCP tools' source selection and timeout behavior;
Sci-Hub fallback remains opt-in through `--use-scihub`.

## Output

`search` and `download` return JSON. Search results are merged works: records of
the same paper (preprint and published versions included) share one persistent
`upid`, with `merge_evidence`, per-field `provenance`, and any conflicting ids under
`quarantined_ids`. `explain_match` gives the confidence and reason for a merge. `read` returns plain text. Config warnings go to stderr and can be ignored.

## Sources

arxiv, pubmed, biorxiv, medrxiv, iacr, semantic, crossref, openalex, pmc, core, europepmc, dblp, openaire, citeseerx, doaj, base, zenodo, hal, ssrn, openreview, unpaywall, acm

Optional (env vars): ieee (`IEEE_API_KEY`). ACM search is keyless and uses Crossref metadata; publisher PDF access may require the OA fallback.

Web of Science (`wos`) and Scopus (`scopus`) require their runtime API keys and
explicit source selection; they are never added to `all`, `fast`, or `fastest`.
Google Scholar (`google_scholar`) is scraped and likewise explicit-only.
Use `--wos-db` or the `--scopus-view`, `--scopus-sort`, `--scopus-field`, and
`--scopus-date` options on ordinary search, or inspect their corresponding tool
help. In MCP, `wos` and `scopus` are searched with their default options through
`search_papers`. `read scopus <id>` returns structured abstract metadata; `--full-text`
explicitly attempts identity-verified article text and returns a nonzero exit
when full text remains unavailable. An abstract-only result is not full text.

`get_related_papers` (`--direction citing|references`) and `manage_search_cache`
(`--action status|clear`) are also available. Relationship lookups accept
DOIs/OpenAlex IDs, with bounded one-hop retrieval. Caching is off by default;
enabling it persists queries and public results, and requires an explicit
operator choice. Clearing the search cache does not remove downloaded PDFs.

## Workflow

1. Search with targeted sources to find papers
2. Present results as a table: title, authors, year, source, DOI/URL
3. If the user wants full text, use `read <source> <paper_id>`
4. If the user wants the PDF, use `download <source> <paper_id>` and report the saved path
