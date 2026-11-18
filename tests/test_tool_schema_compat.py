import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, patch

from paper_search_mcp import server


def _contains_nullable_union(schema):
    if isinstance(schema, dict):
        any_of = schema.get("anyOf")
        if isinstance(any_of, list) and any(
            isinstance(entry, dict) and entry.get("type") == "null" for entry in any_of
        ):
            return True
        return any(_contains_nullable_union(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_contains_nullable_union(item) for item in schema)
    return False


def test_tool_input_schemas_do_not_expose_nullable_unions():
    tools = asyncio.run(server.mcp.list_tools())

    nullable_tools = [
        tool.name for tool in tools if _contains_nullable_union(tool.inputSchema)
    ]

    assert nullable_tools == []


def test_all_tools_declare_honest_open_world_annotations():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    local_only = {"explain_match", "manage_search_cache"}

    assert all(tool.annotations is not None for tool in tools.values())
    assert [
        name for name, tool in tools.items()
        if tool.annotations.openWorldHint is not (name not in local_only)
    ] == []


def test_search_and_lookup_tools_are_annotated_read_only():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    read_only = {"search_papers", "get_paper", "get_related_papers", "explain_match", "extract_sections"}

    assert read_only <= set(tools)
    assert [name for name in read_only if tools[name].annotations.readOnlyHint is not True] == []


def test_file_writing_tools_disclose_destructive_local_updates():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    # read_paper downloads the PDF before extracting text; the cache tool can clear.
    writers = {"download_paper", "read_paper", "manage_search_cache"}

    assert writers <= set(tools)
    assert [
        name for name in writers
        if tools[name].annotations.readOnlyHint is not False
        or tools[name].annotations.destructiveHint is not True
    ] == []


def test_optional_ieee_source_needs_no_extra_tools():
    env = os.environ.copy()
    env["PAPER_SEARCH_MCP_IEEE_API_KEY"] = "test-key"
    script = """
import asyncio
import json
from paper_search_mcp import server

tools = asyncio.run(server.mcp.list_tools())
print(json.dumps({"tools": sorted(t.name for t in tools), "ieee": "ieee" in server.SEARCHERS}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["ieee"] is True
    assert len(result["tools"]) == 8


def test_empty_string_preserves_optional_semantic_year_behavior():
    with patch.object(server, "async_search", new=AsyncMock(return_value=[])) as search:
        asyncio.run(server._search_source("semantic", "test", 3, year=""))

    search.assert_awaited_once_with(server.semantic_searcher, "test", 3)


def test_nonempty_semantic_year_is_forwarded():
    with patch.object(server, "async_search", new=AsyncMock(return_value=[])) as search:
        asyncio.run(server.search_papers("test", 3, sources="semantic", year="2020"))

    search.assert_awaited_once_with(server.semantic_searcher, "test", 3, year="2020")
