import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from paper_search_mcp import server


class TestPaperSearchServer(unittest.TestCase):
    def test_main_uses_local_stdio_transport(self):
        with (
            patch.object(server.mcp, "run") as run,
            patch.object(server.threading, "Thread") as thread,
        ):
            server.main([])

        run.assert_called_once_with(transport="stdio")
        thread.return_value.start.assert_called_once_with()

    def test_all_sources_include_new_platforms(self):
        self.assertIn("dblp", server.ALL_SOURCES)
        self.assertIn("openaire", server.ALL_SOURCES)
        self.assertIn("citeseerx", server.ALL_SOURCES)
        self.assertIn("doaj", server.ALL_SOURCES)
        self.assertIn("base", server.ALL_SOURCES)
        self.assertIn("zenodo", server.ALL_SOURCES)
        self.assertIn("hal", server.ALL_SOURCES)
        self.assertIn("ssrn", server.ALL_SOURCES)
        self.assertIn("unpaywall", server.ALL_SOURCES)

    def test_parse_sources_with_new_platforms(self):
        parsed = server._parse_sources("dblp,doaj,base,zenodo,hal,ssrn,unpaywall,invalid")
        self.assertEqual(parsed, ["dblp", "doaj", "base", "zenodo", "hal", "ssrn", "unpaywall"])

    def test_search_papers_reports_invalid_source_in_mixed_request(self):
        paper = {
            "paper_id": "1234.5678",
            "title": "Test Paper",
            "authors": "Ada Lovelace",
            "doi": "",
            "source": "arxiv",
        }
        with patch.object(server, "_search_source", new=AsyncMock(return_value=[paper])):
            result = asyncio.run(
                server.search_papers("test", sources="arxiv,not-a-source")
            )

        self.assertEqual(result["sources_used"], ["arxiv"])
        self.assertEqual(result["source_results"], {"arxiv": 1})
        self.assertEqual(
            result["errors"]["not-a-source"]["message"],
            "Unknown or unavailable source.",
        )

    def test_search_papers_surfaces_connector_exception(self):
        with patch.object(
            server.semantic_searcher,
            "search",
            side_effect=RuntimeError("HTTP 429: Too many requests"),
        ):
            result = asyncio.run(server.search_papers("test", sources="semantic"))

        self.assertEqual(result["source_results"], {"semantic": 0})
        self.assertEqual(
            result["errors"]["semantic"],
            {"kind": "provider_error", "type": "RuntimeError", "message": "HTTP 429: Too many requests"},
        )

    def test_search_papers_reports_invalid_only_request(self):
        result = asyncio.run(
            server.search_papers("test", sources="not-a-source")
        )

        self.assertEqual(result["sources_used"], [])
        self.assertEqual(
            {source: error["message"] for source, error in result["errors"].items()},
            {
                "not-a-source": "Unknown or unavailable source.",
                "sources": "No valid sources selected.",
            },
        )

    def test_search_papers_merges_preprint_and_published_and_ranks_by_rrf(self):
        by_source = {
            "arxiv": [
                {"paper_id": "1512.03385v1", "title": "Deep Residual Learning for Image Recognition",
                 "authors": "Kaiming He; Jian Sun", "doi": "10.1109/CVPR.2016.90", "source": "arxiv"},
            ],
            "crossref": [
                {"paper_id": "10.1/other", "title": "Unrelated paper", "authors": "A. N. Other",
                 "doi": "10.1/other", "source": "crossref"},
                {"paper_id": "10.1109/CVPR.2016.90", "title": "Deep Residual Learning for Image Recognition",
                 "authors": "He, Kaiming; Sun, Jian", "doi": "10.1109/CVPR.2016.90", "source": "crossref"},
            ],
        }

        async def fake_search(source, query, max_results, year=""):
            return by_source[source]

        with patch.object(server, "_search_source", new=fake_search):
            result = asyncio.run(server.search_papers("resnet", sources="arxiv,crossref"))

        self.assertEqual(result["raw_total"], 3)
        self.assertEqual(result["total"], 2)
        top = result["papers"][0]
        self.assertEqual(top["sources"], ["arxiv", "crossref"])
        self.assertEqual(top["doi"], "10.1109/cvpr.2016.90")
        self.assertTrue(top["upid"].startswith("upid:"))
        explanation = asyncio.run(server.explain_match("arXiv:1512.03385", "10.1109/CVPR.2016.90"))
        self.assertTrue(explanation["merged"])
        self.assertEqual(explanation["confidence"], 1.0)

    def test_google_scholar_and_scihub_are_not_defaults(self):
        self.assertNotIn("google_scholar", server.ALL_SOURCES)
        self.assertIn("google_scholar", server.EXPLICIT_ONLY_SOURCES)
        self.assertEqual(server._parse_sources("google_scholar"), ["google_scholar"])
        params = asyncio.run(server.mcp.list_tools())
        download = next(tool for tool in params if tool.name == "download_paper")
        self.assertIs(download.inputSchema["properties"]["use_scihub"]["default"], False)

    def test_broken_identity_store_does_not_lose_results(self):
        import tempfile
        from paper_search_mcp.identity import IdentityStore

        paper = {"paper_id": "1", "title": "Kept", "authors": "A. B", "source": "arxiv"}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, "_IDENTITY_STORE", IdentityStore(directory)), \
             patch.object(server, "_search_source", new=AsyncMock(return_value=[paper])):
            result = asyncio.run(server.search_papers("test", sources="arxiv"))

        self.assertEqual([p["title"] for p in result["papers"]], ["Kept"])
        self.assertIn("identity_store", result["errors"])

    def test_tool_set_is_compact(self):
        names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
        self.assertEqual(len(names), 8)


if __name__ == "__main__":
    unittest.main()
