"""Offline parity between the CLI tool adapter and the live MCP registry."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import TextContent, Tool

from paper_search_mcp import cli, server, tool_cli


def run_tool(*arguments):
    args = cli.build_parser().parse_args(["tool", *arguments])
    return asyncio.run(cli.cmd_tool(args))


def test_list_is_exactly_the_mcp_registry(capsys):
    expected = [
        tool.model_dump(mode="json", exclude_none=True)
        for tool in asyncio.run(server.mcp.list_tools())
    ]
    assert run_tool("--list") == 0
    assert json.loads(capsys.readouterr().out) == {"tools": expected}


def test_help_lists_every_registered_tool(capsys):
    with pytest.raises(SystemExit) as exc:
        run_tool("--help")
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for tool in asyncio.run(server.mcp.list_tools()):
        assert tool.name in output


def test_tool_help_exposes_tool_options(capsys):
    with pytest.raises(SystemExit) as exc:
        run_tool("search_papers", "--help")
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--sources" in output
    assert "--max-results-per-source" in output
    assert "--year" in output


def test_all_tool_properties_map_to_arguments_with_mcp_defaults():
    tools = asyncio.run(server.mcp.list_tools())
    parser = tool_cli.build_tool_parser(tools)
    samples = {"string": "example", "integer": "3", "boolean": "true", "number": "1.5"}
    for tool in tools:
        properties = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])
        # Required argument order follows the schema property order, just like
        # the corresponding Python function signature.
        values = [samples[properties[name]["type"]] for name in properties if name in required]
        parsed = parser.parse_args([tool.name, *values])
        assert parsed._tool_name == tool.name
        assert parsed._tool_parameter_names == tuple(properties)
        assert set(vars(parsed)) - {
            "_tool_name", "_tool_parameter_names", "_list_tools"
        } == set(required)
        # Omitted optional values must be omitted, not replaced by CLI defaults.
        # This delegates the authoritative defaults to the MCP dispatcher.
        for name in properties.keys() - set(required):
            assert not hasattr(parsed, name)


@pytest.mark.parametrize("flag,expected", [(None, False), ("--use-scihub", True), ("--no-use-scihub", False)])
def test_boolean_false_default_does_not_enable_restricted_fallback(flag, expected):
    tools = asyncio.run(server.mcp.list_tools())
    parser = tool_cli.build_tool_parser(tools)
    args = parser.parse_args(["download_paper", "arxiv", "id", *([flag] if flag else [])])
    assert getattr(args, "use_scihub", False) is expected


def test_options_and_mcp_validation_path(monkeypatch, capsys):
    search = AsyncMock(return_value=[{"paper_id": "s2id", "source": "semantic", "title": "T"}])
    monkeypatch.setattr(server, "async_search", search)
    original_call = server.mcp.call_tool
    call = AsyncMock(wraps=original_call)
    monkeypatch.setattr(server.mcp, "call_tool", call)
    assert run_tool(
        "search_papers", "transformer attention", "--max-results-per-source", "2",
        "--sources", "semantic", "--year", "2024",
    ) == 0
    assert json.loads(capsys.readouterr().out)["papers"][0]["paper_id"] == "s2id"
    call.assert_awaited_once_with("search_papers", {
        "query": "transformer attention", "max_results_per_source": 2,
        "sources": "semantic", "year": "2024",
    })
    search.assert_awaited_once_with(server.semantic_searcher, "transformer attention", 2, year="2024")


def test_plain_text_and_save_path_are_preserved(monkeypatch, capsys):
    read = Mock(return_value="paper text")
    monkeypatch.setattr(server.arxiv_searcher, "read_paper", read)
    assert run_tool("read_paper", "arxiv", "1234.5678", "--save-path", "/a directory") == 0
    assert capsys.readouterr().out == "paper text\n"
    read.assert_called_once_with("1234.5678", "/a directory")


def test_exception_returns_json_and_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(server.arxiv_searcher, "read_paper", Mock(side_effect=RuntimeError("offline")))
    assert run_tool("read_paper", "arxiv", "1") == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "error", "message": "Error executing tool read_paper: offline",
    }
    assert "Traceback" not in output.err


@pytest.mark.parametrize("arguments", [
    [], ["does_not_exist"], ["search_papers"],
    ["search_papers", "query", "--max-results-per-source", "invalid"],
    ["search_papers", "query", "--not-an-option"], ["search_papers", "query", "--max", "3"],
    ["--list", "search_papers", "query"],
])
def test_invalid_invocations_do_not_execute(arguments, monkeypatch, capsys):
    call = AsyncMock()
    monkeypatch.setattr(server.mcp, "call_tool", call)
    with pytest.raises(SystemExit) as exc:
        run_tool(*arguments)
    assert exc.value.code == 2
    call.assert_not_called()
    assert capsys.readouterr().out == ""


def test_unified_search_retains_explicit_sources_and_error_payload(monkeypatch, capsys):
    arxiv = Mock(return_value=[])
    pubmed = Mock(side_effect=AssertionError("Unrequested source"))
    monkeypatch.setattr(server.arxiv_searcher, "search", arxiv)
    monkeypatch.setattr(server.pubmed_searcher, "search", pubmed)
    assert run_tool("search_papers", "test", "--sources", "arxiv,not-a-source") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources_used"] == ["arxiv"]
    assert payload["errors"]["not-a-source"]["message"] == "Unknown or unavailable source."
    arxiv.assert_called_once()
    pubmed.assert_not_called()


def test_unified_search_still_returns_partial_results_on_timeout(monkeypatch, capsys):
    async def fake_search(source, *args, **kwargs):
        if source == "arxiv":
            await asyncio.sleep(0.05)
            return []
        return [{"paper_id": "fast", "title": "Fast"}]

    monkeypatch.setattr(server, "SEARCH_PAPERS_SOURCE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(server, "_search_source", fake_search)
    assert run_tool("search_papers", "test", "--sources", "arxiv,pubmed") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_results"] == {"arxiv": 0, "pubmed": 1}
    assert "timed out" in payload["errors"]["arxiv"]["message"]


def clean_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PAPER_SEARCH")}
    for name in ("IEEE_API_KEY", "ACM_API_KEY", "UNPAYWALL_EMAIL", "CORE_API_KEY", "DOAJ_API_KEY"):
        env.pop(name, None)
    env["PAPER_SEARCH_MCP_ENV_FILE"] = str(tmp_path / "absent.env")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return env


@pytest.mark.parametrize("configured", [False, True])
def test_cli_tool_list_is_the_compact_set_with_or_without_ieee(configured, tmp_path):
    env = clean_env(tmp_path)
    if configured:
        env["PAPER_SEARCH_MCP_IEEE_API_KEY"] = "offline-test-key"
    result = subprocess.run(
        [sys.executable, "-m", "paper_search_mcp.cli", "tool", "--list"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20, check=True,
    )
    names = {tool["name"] for tool in json.loads(result.stdout)["tools"]}
    assert len(names) == 8 and "search_papers" in names


@pytest.mark.parametrize("arguments", [
    ["--help"], ["sources"], ["search", "--help"], ["download", "--help"], ["read", "--help"],
])
def test_legacy_commands_do_not_import_server(arguments, tmp_path):
    script = """
import sys
from paper_search_mcp import cli
try:
    cli.main()
finally:
    assert 'paper_search_mcp.server' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments], cwd=tmp_path, env=clean_env(tmp_path),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "No CORE API key provided" not in result.stderr


def test_cli_process_executes_tool_and_exits_without_starting_server(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "paper_search_mcp.cli", "tool", "search_papers", "test", "--sources", "invalid"],
        cwd=tmp_path, env=clean_env(tmp_path), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["sources_used"] == []
    assert payload["errors"]["sources"]["message"] == "No valid sources selected."


def test_json_arguments_support_future_tools_without_new_registry():
    tool = Tool(name="example", inputSchema={
        "type": "object", "properties": {
            "values": {"type": "array", "items": {"type": "integer"}},
            "settings": {"type": "object", "default": {}},
        }, "required": ["values"],
    })
    parsed = tool_cli.build_tool_parser([tool]).parse_args([
        "example", "[1,2]", "--settings", '{"enabled":true}',
    ])
    assert parsed.values == [1, 2]
    assert parsed.settings == {"enabled": True}


def test_unstructured_text_output_is_retained(capsys):
    tool_cli._print_result([TextContent(type="text", text="Paper text")])
    assert capsys.readouterr().out == "Paper text\n"
