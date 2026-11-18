# Multi-stage build for smaller image
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY paper_search_mcp/ paper_search_mcp/

RUN pip install --no-cache-dir build \
    && python -m build --wheel \
    && pip install --no-cache-dir dist/*.whl

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/paper-search-mcp /usr/local/bin/paper-search-mcp

# API keys are passed at runtime with -e (PAPER_SEARCH_MCP_* or legacy names);
# empty ENV placeholders here would shadow the legacy names, so none are baked in.

# Unprivileged user; /data holds downloads and the identity store (mount a volume).
RUN useradd --uid 10001 --create-home app && mkdir /data && chown app /data
USER app
WORKDIR /data
ENV PAPER_SEARCH_MCP_IDENTITY_DB_PATH=/data/identity.sqlite3

# Use the entry point script
CMD ["paper-search-mcp"]
