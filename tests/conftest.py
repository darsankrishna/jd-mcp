import os
import sys
import tempfile

import pytest

# Keep the persistent identity store out of the user's home during tests.
os.environ.setdefault(
    "PAPER_SEARCH_MCP_IDENTITY_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="psm-identity-"), "identity.sqlite3"),
)


@pytest.fixture(autouse=True)
def _reset_http_save_path_confinement():
    # server.main() over HTTP turns confinement on module-wide; keep it per test.
    yield
    server = sys.modules.get("paper_search_mcp.server")
    if server is not None:
        server._CONFINE_SAVE_PATHS = False
