# jd-mcp: project context

A scholarly-search MCP server. It is a fork of
[openags/paper-search-mcp](https://github.com/openags/paper-search-mcp) at commit
`cab1b5c` (MIT), keeping all its connectors and replacing its duplicate handling,
ranking and tool surface. The research goal is reliable **work-level identity**:
knowing when two records from different sources are the same paper.

Branch `fork-identity-resolver` (not pushed; `main` is only the initial commit):

| Commit | What it added |
| --- | --- |
| `dc1b1c7` | Fork: identity resolver, 8 tools, RRF, no DOI guessing, typed errors, PDF size limit |
| `5ad1358` | Trained match classifier, constructed dedup benchmark, vetoes on persisted merges |
| `d8dd3ff` | Live-data evaluation (dev + held-out), PubMed author fix, published-DOI veto, OpenAlex/dblp raise on failure |
| `7a67d36` | `--allowed-hosts`, download paths confined over HTTP |
| `8a29771` | `deploy/` stack, non-root Docker image, `search_papers` size cap |

## Scope rules (from the original brief)

Fork paper-search-mcp; keep its connectors and MIT licence (both copyright lines
are in `LICENSE`). Do not add data sources, do not build greedy single-pass
merging, do not use ASCII-only tokenizers. Ideas borrowed from scholarmcp
(per-source provenance, provider error reporting, PDF size limits) are
reimplemented, not copied. The brief was `Base repo analysis.md` (outside the repo).

## Layout

| Path | What |
| --- | --- |
| `paper_search_mcp/server.py` | The 8 MCP tools, source registry `SEARCHERS`, typed provider errors, PDF size limit, `--allowed-hosts`, save-path confinement, `main()` |
| `paper_search_mcp/identity.py` | Identity resolution, classifier scoring, vetoes, union-find, UPIDs, SQLite `IdentityStore`, RRF |
| `paper_search_mcp/match_model.json` | Trained classifier weights; regenerate with `benchmarks.dedup.train` |
| `paper_search_mcp/http_auth.py` | OAuth protected-resource mode (upstream) |
| `paper_search_mcp/academic_platforms/` | One connector per source (upstream, with fixes below) |
| `paper_search_mcp/cli.py` | `paper-search` CLI; its `search` uses the same resolver |
| `benchmarks/dedup/` | Gold sets, live capture, classifier training, runner, results |
| `deploy/` | Compose stack for a cloud server: Caddy, Keycloak on Postgres, jd-mcp |
| `tests/` | `test_identity.py`, `test_identity_connectors.py`, `test_server.py`, `test_dedup_benchmark.py` plus upstream suites |

## The 8 tools

`search_papers`, `get_paper`, `explain_match`, `download_paper`, `read_paper`,
`extract_sections`, `get_related_papers`, `manage_search_cache` (upstream had 74).
`tool_cli.py` exposes the same tools as `paper-search tool <name>`.

- `search_papers(query, max_results_per_source=5, sources="all", year="", limit=20)`
  returns merged works ranked by RRF. `total` counts all works found, `returned`
  how many were sent. Abstracts are cut to 500 characters (`ABSTRACT_CHARS`).
  `errors[source]` is `{kind, type, message}`, with kind one of `timeout`, `http`,
  `capacity`, `provider_error`, `invalid_request`.
- `sources="all"` excludes `google_scholar`, `wos` and `scopus` (explicit only).
- Sci-Hub runs only with `download_paper(use_scihub=True)`; there is no Sci-Hub tool.
- Each work: `upid`, `source`/`paper_id` of the leading record, bibliographic
  fields, `ids`, `quarantined_ids`, `sources`, `source_ranks`, `rrf_score`,
  `provenance` (field → source), `pdf_urls`, `records`, `merge_evidence`.

## Identity resolution (`identity.py`)

1. **Identifiers from explicit fields only:** DOI (normalised), arXiv id (version
   stripped; `10.48550/arXiv.*` maps to arXiv), PMID, OpenAlex, S2, Crossref
   `is-preprint-of`/`has-preprint`/version relations, OpenAlex location arXiv ids,
   and the record key `src:<source>:<paper_id>`. `extract_doi(abstract)` fallbacks
   were deleted from every connector.
2. **Edge kinds,** processed in order: 0 shared identifier (never vetoed),
   1 persisted crosswalk UPID (vetoed), 2 classifier evidence (vetoed). Ties break
   on record keys, so the result does not depend on input order.
3. **Hard vetoes** (cannot-link, checked across whole clusters before a union):
   - different title numbers or parts (Part I vs II);
   - erratum, corrigendum or retraction vs a normal paper;
   - disjoint author lists;
   - a year gap over 2;
   - two different *published* DOIs. Preprint-server prefixes don't count:
     arXiv, bioRxiv/medRxiv `10.1101/<digit>`, SSRN, Research Square, Preprints.org,
     Authorea, ChemRxiv, OSF, PsyArXiv.

   Stored vetoes from earlier searches stay cannot-link.
4. **Classifier:** logistic regression over 8 features (`FEATURES`: title bigram
   Dice, token Jaccard, author overlap, first-author match, year gap, arXiv
   journal-ref match, venue overlap, title length ratio). It is consulted only
   when both author lists exist and the title is not generic. The threshold is
   stored in the model file (0.5).
5. **Authors:** keyed as (surname, first initial), accent-folded. PubMed's
   "Love MI" order is detected per list (`author_keys`), not per name.
6. **Titles:** NFKC + casefold, keeping Unicode letters, marks and numbers
   (Indic marks kept); similarity is character-bigram Dice, which handles CJK.
7. **Persistence:** SQLite tables `crosswalk` (identifier, record, upid),
   `redirects`, `splits`, `edges` (merged and vetoed, with confidence and reason)
   and `quarantine`. Merging UPIDs redirects the loser. A veto against a stored
   merge splits it: the work holding most identifiers keeps the UPID. Commits
   re-read inside `BEGIN IMMEDIATE`.
8. **Conflicts:** two values of a single-valued id in one work go to
   `quarantined_ids`, never overwrite each other.
9. **Fields and ranking:** published records lead over preprints; every field
   records its source in `provenance`. RRF uses k = 60.

## Connector changes

- arXiv reads `arxiv_doi` and `arxiv_journal_ref`; Crossref keeps `relation`;
  OpenAlex adds `pmid`, `arxiv_ids` and `venue`; Semantic Scholar adds `arxiv_ids`
  and `pmid`.
- `Paper.to_dict()` returns `extra` as a JSON-safe dict (upstream stringified it).
- OpenAlex and dblp **raise** on HTTP errors and dblp's bot challenge, instead of
  returning `[]`.

## Benchmark (`benchmarks/dedup/`)

| Set | Data | jd-mcp F1 | scholarmcp (port) | paper-search-mcp |
| --- | --- | --- | --- | --- |
| Constructed test split | `data/generated.jsonl`, `data/handbuilt.json` | 1.000 | 0.568 | 0.244 |
| Live dev (biased: used to fix bugs) | `data/live/` | 1.000 | 0.863 | 0.550 |
| **Live held-out (the honest number)** | `data/live-test/` | **0.938** (P 1.000, R 0.882) | 0.903 | 0.786 |

- **Constructed set:** synthetic and saturated for jd-mcp, because the generator
  and the resolver were written around the same failure modes. Never cite it as
  accuracy.
- **Live sets:** real connector responses. Labels come from Semantic Scholar's
  batch API, and S2 is never an evaluated source. `review.csv` lists pairs for a
  person to check; a `review` value (`same`/`different`/`unsure`) overrides S2.
  **No pairs have been reviewed yet.**
- **Held-out set:** resolver hashes were frozen before capture
  (`data/live-test/frozen.json`). Do not change the resolver because of its results.
- **Held-out misses:** OpenAlex author strings like "PedregosaFabian"; Planck VI
  (collaboration first author, 2-year gap); FCN conference vs journal (arguable).
- **Baselines:** upstream's dedupe is copied verbatim; scholarmcp is a line-for-line
  Python port, and tests check it reproduces the analysis findings. Liyux3/scholar-mcp
  is opt-in via `--liyux3-path`. Its licence (PolyForm Noncommercial plus a
  "no competing products" clause) needs the owner's legal call; it has never been run.
- **Gaps in the live data:** dblp blocks scripted access, and anonymous OpenAlex
  was throttled, so its coverage is partial. Positive pairs are few (34 held-out).

```bash
uv run python -m benchmarks.dedup.run --write                     # constructed
uv run python -m benchmarks.dedup.run --split live-test --write   # held-out live
uv run python -m benchmarks.dedup.live_capture --set test         # recapture (new set only)
uv run python -m benchmarks.dedup.train                           # retrain match_model.json
```

## Configuration (`PAPER_SEARCH_MCP_` prefix; legacy unprefixed names also read)

| Setting | Default | Purpose |
| --- | --- | --- |
| `TRANSPORT`, `HOST`, `PORT`, `PATH` | stdio, 127.0.0.1, 8000, /mcp | Server transport (CLI flags override) |
| `ALLOWED_HOSTS` | (empty) | Extra `Host` values besides localhost: `host`, `host:port`, `host:*` |
| `AUTH` and `OAUTH_ISSUER`, `OAUTH_JWKS_URI`, `OAUTH_RESOURCE_URL`, `OAUTH_AUDIENCE`, `OAUTH_SCOPES`, `OAUTH_ALGORITHM` | none | OAuth resource mode; issuer and JWKS **must be HTTPS**; audience == resource URL |
| `IDENTITY_DB_PATH` | `~/.cache/paper-search-mcp/identity.sqlite3` (Docker: `/data/identity.sqlite3`) | Identity store |
| `SECTION_PDF_ROOT` | `./downloads` | Root for `extract_sections`, and for downloads over HTTP |
| `MAX_PDF_BYTES` | 52428800 | PDF download size limit |
| `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `CORE_API_KEY`, `UNPAYWALL_EMAIL`, … | (empty) | Provider keys |

Over HTTP transports, `download_paper` and `read_paper` refuse `save_path` outside
`SECTION_PDF_ROOT`; stdio is unrestricted. The SDK only accepts localhost `Host`
headers unless OAuth adds the resource host or `ALLOWED_HOSTS` adds others
(otherwise HTTP 421).

## Current deployment (owner's MacBook, Docker + Tailscale Funnel)

- **Public base:** `https://darsans-macbook-air-2.tail6fe82c.ts.net`, via Tailscale
  Funnel on port 443 only. The routes are saved in Tailscale and survive reboots:
  - `/mcp` and `/.well-known/oauth-protected-resource` go to jd-mcp at `127.0.0.1:8000`;
  - `/realms/papers` and `/resources` go to Keycloak at `127.0.0.1:8080`.
  - The admin console and the `master` realm are deliberately **not** published.
- **Containers** (`restart=unless-stopped`):
  - `keycloak` (26.8.2, `start-dev`, volume `keycloak-data`): `KC_HOSTNAME` = the
    public base, `KC_HOSTNAME_ADMIN=http://localhost:8080`, `KC_PROXY_HEADERS=xforwarded`.
    The master realm's `frontendUrl` is `http://localhost:8080`.
  - `jd-mcp` (local image, volume `jd-mcp-data`): OAuth mode, issuer
    `<base>/realms/papers`, resource and audience `<base>/mcp`, scope `papers:use`.
- **Keycloak realm `papers`:**
  - client scope `papers:use`: Default, in token scope, with Audience mapper
    `mcp-audience` = `<base>/mcp`;
  - anonymous client registration with trusted hosts `localhost`, `127.0.0.1`,
    `claude.ai`, `gemini.google.com`; sending-host match off, client URIs must match;
  - one user per person (never share admin).
- **Admin console:** `http://localhost:8080/admin`, on the Mac only, **in Chrome or
  Firefox**. Safari drops Keycloak's `Secure` cookies on http://localhost, which
  shows up as `cookie_not_found`.
- **Client in use:** a friend's claude.ai custom connector at `<base>/mcp`; it works
  end to end.
- **After reboot:** containers and Funnel restore themselves once Docker Desktop and
  Tailscale run. Docker Desktop "start at login" was **off** and the Mac slept after
  1 minute idle; the owner was told to enable autostart and `sudo pmset -c sleep 0`.
  Health check: `docker ps`, then `curl -X POST <base>/mcp …` should return 401.

## Deployment lessons

- An HTTP issuer is rejected at startup; local trials need HTTPS in front of
  Keycloak (Caddy `tls internal` plus `SSL_CERT_FILE` / `NODE_EXTRA_CA_CERTS`).
- `$TS`-style shell variables are empty in a new terminal tab. Write hostnames out
  in `docker run` commands.
- Funnel publishes every path on a port; publish only the paths needed.
- claude.ai connects from Anthropic's cloud, so a tailnet-only server is unreachable
  for it; Claude Code, Cursor and VS Code connect from the user's machine.
- Docker on macOS can reach the tailnet name from inside containers (verified).
- The cloud variant is `deploy/`: `cp .env.example .env`, then
  `docker compose up -d --build`; Let's Encrypt needs public DNS and ports 80/443.

## Testing

```bash
uv run pytest -q tests -p no:cacheprovider
```

Expected: about 1187 passed. Known failures not caused by this fork:
- 16 in `tests/test_sections.py`: macOS lacks the POSIX resource limits they need.
- Live-network tests such as `test_semantic.py` and `test_crossref.py::test_search`
  fail when providers rate-limit.

`tests/conftest.py` points the identity store at a temp dir and resets the
HTTP save-path flag after each test. New test files must also be added to the
explicit lists in `.github/workflows/ci.yml` and `publish.yml`.

## Open items

- Review `benchmarks/dedup/data/live-test/review.csv`, then rescore.
- Capture a larger live set, ideally with an OpenAlex API key; fix "PedregosaFabian"
  author parsing against the dev set, not the held-out one.
- Liyux3 baseline: a licence decision is needed before running it.
- Push the branch to a remote; `main` has only the initial commit.
- Shortcuts noted in code (`shortcut:` comments): O(n²) pair scoring; connector
  downloads are size-checked after writing; stored vetoes are scanned per search.
- The save-path symlink check runs once before writing, so someone who can already
  write to the server's disk could race it; remote clients cannot.
