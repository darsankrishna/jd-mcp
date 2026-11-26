"""Connectors expose explicit identity evidence and never guess DOIs from text."""

import asyncio
import os
from unittest.mock import patch

import feedparser

from paper_search_mcp import server
from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher
from paper_search_mcp.academic_platforms.core import CORESearcher
from paper_search_mcp.academic_platforms.crossref import CrossRefSearcher
from paper_search_mcp.academic_platforms.openalex import OpenAlexSearcher
from paper_search_mcp.academic_platforms.semantic import SemanticSearcher
from paper_search_mcp.identity import resolve

DATASET = "We release data at https://doi.org/10.5281/zenodo.1234567."

ARXIV_FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <updated>2015-12-10T19:51:55Z</updated>
    <published>2015-12-10T19:51:55Z</published>
    <title>Deep Residual Learning for Image Recognition</title>
    <summary>{{summary}}</summary>
    <author><name>Kaiming He</name></author>
    <arxiv:doi>{{doi}}</arxiv:doi>
    <arxiv:journal_ref>{{journal_ref}}</arxiv:journal_ref>
    <link href="http://arxiv.org/pdf/1512.03385v1" rel="related" type="application/pdf"/>
    <category term="cs.CV"/>
  </entry>
</feed>"""


def arxiv_paper(summary=DATASET, doi="", journal_ref=""):
    feed = feedparser.parse(ARXIV_FEED.format(summary=summary, doi=doi, journal_ref=journal_ref))
    return ArxivSearcher._paper_from_entry(feed.entries[0]).to_dict()


def test_arxiv_never_takes_a_doi_from_the_abstract():
    assert arxiv_paper()["doi"] == ""


def test_arxiv_exposes_published_doi_and_journal_ref():
    paper = arxiv_paper(doi="10.1109/CVPR.2016.90", journal_ref="CVPR 2016")
    assert paper["doi"] == "10.1109/CVPR.2016.90"
    assert paper["extra"]["journal_ref"] == "CVPR 2016"


def test_openalex_no_abstract_doi_and_arxiv_locations():
    work = {
        "id": "https://openalex.org/W1",
        "title": "Deep Residual Learning",
        "abstract_inverted_index": {"data": [0], "10.5281/zenodo.1234567": [1]},
        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
        "locations": [{"landing_page_url": "https://arxiv.org/abs/1512.03385v1"}],
    }
    paper = OpenAlexSearcher(api_key="")._parse_work(work).to_dict()
    assert paper["doi"] == ""
    assert paper["extra"]["arxiv_ids"] == ["1512.03385v1"]
    assert paper["extra"]["pmid"] == "123"


def test_two_papers_citing_the_same_dataset_are_not_collapsed():
    item = lambda n: {"id": f"https://openalex.org/W{n}", "title": f"Paper {n}",
                      "abstract_inverted_index": {DATASET: [0]}}
    searcher = OpenAlexSearcher(api_key="")
    papers = [searcher._parse_work(item(n)).to_dict() for n in (1, 2)]
    assert len(resolve({"openalex": papers})) == 2


def test_semantic_and_core_do_not_guess_dois():
    s2 = SemanticSearcher()._parse_paper({
        "paperId": "abc", "title": "T", "abstract": DATASET, "authors": [],
        "externalIds": {"ArXiv": "1512.03385", "PubMed": "42"},
    }).to_dict()
    assert s2["doi"] == ""
    assert s2["extra"]["arxiv_ids"] == ["1512.03385"] and s2["extra"]["pmid"] == "42"
    core = CORESearcher()._parse_item({"id": 1, "title": "T", "abstract": DATASET})
    assert core.doi == ""


def test_crossref_relation_is_kept():
    relation = {"is-preprint-of": [{"id-type": "doi", "id": "10.1/pub", "asserted-by": "subject"}]}
    paper = CrossRefSearcher()._parse_crossref_item({
        "DOI": "10.1101/x", "title": ["T"], "type": "posted-content", "relation": relation,
    }).to_dict()
    assert paper["extra"]["relation"] == relation
    assert paper["extra"]["crossref_type"] == "posted-content"


class _Stream:
    def __init__(self, chunks, headers):
        self.chunks, self.headers, self.status_code = chunks, headers, 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _Client:
    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url):
        return self._stream


def _download(stream, tmp_path):
    with patch.object(server, "MAX_PDF_BYTES", 10), \
         patch.object(server.httpx, "AsyncClient", return_value=_Client(stream)):
        return asyncio.run(server._download_from_url("https://example.org/x.pdf", str(tmp_path)))


def test_pdf_download_refuses_declared_oversize(tmp_path):
    assert _download(_Stream([b"%PDF-1.7"], {"content-length": "11"}), tmp_path) is None
    assert os.listdir(tmp_path) == []


def test_pdf_download_stops_streaming_past_limit(tmp_path):
    assert _download(_Stream([b"%PDF-1.7", b"more bytes"], {}), tmp_path) is None
    assert os.listdir(tmp_path) == []


def test_pdf_download_within_limit_is_written(tmp_path):
    assert _download(_Stream([b"%PDF-1.7"], {}), tmp_path).endswith(".pdf")


def test_native_download_over_limit_is_removed(tmp_path):
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-" + b"x" * 20)
    with patch.object(server, "MAX_PDF_BYTES", 10), \
         patch.object(server.arxiv_searcher, "download_pdf", return_value=str(big)), \
         patch.object(server, "_try_repository_fallback", return_value=(None, "none")):
        result = asyncio.run(server.download_paper("arxiv", "1", save_path=str(tmp_path)))
    assert "download limit" in result
    assert not big.exists()


def test_dblp_bot_challenge_is_an_error_not_empty_results():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from paper_search_mcp.academic_platforms.dblp import DBLPSearcher

    challenge = SimpleNamespace(status_code=202, text="<title>Making sure you're not a bot!</title>",
                                content=b"<html></html>", raise_for_status=lambda: None)
    searcher = DBLPSearcher()
    searcher.session = Mock(get=Mock(return_value=challenge))
    with pytest.raises(RuntimeError, match="bot challenge"):
        searcher.search("deep residual learning", max_results=3)
