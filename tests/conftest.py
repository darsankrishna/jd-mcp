import os
import tempfile

# Keep the persistent identity store out of the user's home during tests.
os.environ.setdefault(
    "PAPER_SEARCH_MCP_IDENTITY_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="psm-identity-"), "identity.sqlite3"),
)
