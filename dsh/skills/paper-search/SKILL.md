---
name: paper-search
description: Search, download, and read academic papers through the paper-search MCP tools (mcp__paper-search__*), covering arXiv, PubMed, bioRxiv, Semantic Scholar, Crossref, OpenAlex and 15+ other sources. Use when the user asks to find papers, search research literature, download a paper PDF, or extract text from a paper.
whenToUse: Use when the user asks to find, download, or read academic papers or literature. Prefer these MCP tools over a raw web search for paper discovery.
---

# Paper Search

Academic paper discovery and retrieval in this harness runs through the `mcp__paper-search__*` tools (the paper-search-mcp MCP server). Do not run the `paper-search` CLI or hand-write API requests — use these tools.

## Tools

- `mcp__paper-search__search_papers` — multi-source concurrent search. Records of the same work (including preprint and published versions) are merged under a persistent UPID with merge evidence and per-field provenance; works are ranked by reciprocal rank fusion. Per-source failures are reported in `errors` with a `kind` (timeout, http, capacity, provider_error, invalid_request).
- `mcp__paper-search__get_paper` — look up a DOI, arXiv id, PMID or UPID: its crosswalk identifiers, redirects, quarantined conflicting ids, and Crossref metadata.
- `mcp__paper-search__explain_match` — whether two identifiers are the same work, with confidence and the stored reason or veto.
- `mcp__paper-search__download_paper` — OA-first PDF download: source-native → repository → Unpaywall; size-limited. Sci-Hub only with explicit `use_scihub=true`.
- `mcp__paper-search__read_paper` — download and extract text from one source's record.
- `mcp__paper-search__extract_sections` — split an already-downloaded PDF into heading-based sections.
- `mcp__paper-search__get_related_papers` — bounded one-hop OpenAlex `citing` or `references` lookup from a DOI or OpenAlex ID.
- `mcp__paper-search__manage_search_cache` — `status` or `clear` the opt-in search cache; clear only when the user asks. Clearing does not remove PDFs.

Sources: arxiv, pubmed, biorxiv, medrxiv, iacr, semantic, crossref, openalex, pmc, core, europepmc, dblp, openaire, citeseerx, doaj, base, zenodo, hal, ssrn, openreview, unpaywall, acm (ieee with `PAPER_SEARCH_MCP_IEEE_API_KEY`). ACM search is available without an API key. Web of Science (`wos`), Scopus (`scopus`) and Google Scholar (`google_scholar`) are explicit-only: name them in `sources`; `all` never includes them. Do not describe an abstract or metadata record as retrieved full text.

## Workflow

1. Search: `search_papers`, narrowing `sources` when the field is clear (e.g. `arxiv,semantic,crossref`).
2. Present results as a table: title, authors, year, sources, DOI/URL.
3. Full text: `read_paper`; PDF: `download_paper` with the work's DOI and title, then report the saved path.
4. If a merge looks wrong, call `explain_match` and report its reason.

## Notes

- Many public sources work without API keys; optional keys (Semantic Scholar, CORE, Unpaywall email, …) live in `~/.config/paper-search-mcp/.env` and are loaded by the server automatically.
- Google Scholar and SSRN can be bot-blocked upstream; fall back to other sources rather than retrying.
