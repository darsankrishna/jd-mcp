"""Identity resolution: the duplicate/false-merge cases from the base-repo analysis."""

import random

import pytest

from paper_search_mcp.identity import (
    IdentityStore,
    author_key,
    normalize_title,
    resolve,
    title_similarity,
    title_tokens,
)


def rec(source, paper_id, title, authors="", doi="", date="", **extra):
    return {
        "source": source, "paper_id": paper_id, "title": title, "authors": authors,
        "doi": doi, "published_date": date, "abstract": "", "url": "", "pdf_url": "",
        "citations": 0, "extra": extra,
    }


@pytest.fixture
def store(tmp_path):
    return IdentityStore(tmp_path / "identity.sqlite3")


def works_of(results, store=None):
    return resolve(results, store)


# --- the analysis table ----------------------------------------------------

def test_preprint_vs_journal_reworded_title_via_arxiv_doi_field():
    works = works_of({
        "arxiv": [rec("arxiv", "1706.03762v5", "Attention Is All You Need",
                      "Ashish Vaswani; Noam Shazeer", doi="10.5555/3295222.3295349", date="2017-06-12")],
        "crossref": [rec("crossref", "10.5555/3295222.3295349",
                         "Attention is all you need: transformers for sequence transduction",
                         "Vaswani, Ashish; Shazeer, Noam", doi="10.5555/3295222.3295349", date="2017-12-04")],
    })
    assert len(works) == 1
    assert "shared identifier doi:10.5555/3295222.3295349" in works[0]["merge_evidence"][0]["reason"]


def test_preprint_vs_journal_reworded_title_via_journal_ref():
    works = works_of({
        "arxiv": [rec("arxiv", "1512.03385v1", "Deep Residual Learning for Image Recognition",
                      "Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun", date="2015-12-10",
                      journal_ref="CVPR 2016")],
        "dblp": [rec("dblp", "conf/cvpr/HeZRS16", "Residual learning for image recognition (CVPR)",
                     "Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun", date="2016-06-27")],
    })
    assert len(works) == 1
    assert "journal-ref" in works[0]["merge_evidence"][0]["reason"]


def test_crossref_is_preprint_of_relation_links_different_dois():
    relation = {"is-preprint-of": [{"id-type": "doi", "id": "10.1038/s41586-020-0001-1"}]}
    works = works_of({
        "crossref": [
            rec("crossref", "10.1101/2020.01.01.000001", "A preprint title", "Ada Lovelace",
                doi="10.1101/2020.01.01.000001", date="2020-01-01",
                crossref_type="posted-content", relation=relation),
            rec("crossref", "10.1038/s41586-020-0001-1", "Totally rewritten journal title",
                "Lovelace, A.", doi="10.1038/s41586-020-0001-1", date="2020-09-01"),
        ],
    })
    assert len(works) == 1
    work = works[0]
    assert work["doi"] == "10.1038/s41586-020-0001-1"  # published DOI is canonical
    assert work["ids"]["preprint_doi"] == "10.1101/2020.01.01.000001"
    assert not work["quarantined_ids"]


def test_same_title_arxiv_doi_vs_conference_doi_resnet():
    authors = "Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun"
    works = works_of({
        "semantic": [rec("semantic", "s2a", "Deep Residual Learning for Image Recognition", authors,
                         doi="10.48550/arXiv.1512.03385", date="2015-12-10")],
        "crossref": [rec("crossref", "10.1109/CVPR.2016.90", "Deep Residual Learning for Image Recognition",
                         authors, doi="10.1109/CVPR.2016.90", date="2016-06-01")],
    })
    assert len(works) == 1
    assert works[0]["doi"] == "10.1109/cvpr.2016.90"
    assert works[0]["ids"]["arxiv"] == "1512.03385"


def test_title_punctuation_and_case_differ_no_doi():
    works = works_of({
        "core": [rec("core", "1", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                     "Jacob Devlin; Ming-Wei Chang")],
        "hal": [rec("hal", "2", "bert pre training of deep bidirectional transformers for language understanding.",
                    "Jacob Devlin; Ming-Wei Chang")],
    })
    assert len(works) == 1


def test_initials_vs_full_name_no_doi():
    works = works_of({
        "dblp": [rec("dblp", "1", "Attention Is All You Need", "A. Vaswani; N. Shazeer")],
        "core": [rec("core", "2", "Attention is all you need", "Ashish Vaswani; Noam Shazeer")],
    })
    assert len(works) == 1


def test_surname_comma_vs_given_first():
    works = works_of({
        "europepmc": [rec("europepmc", "1", "Deep residual learning", "He, Kaiming; Sun, Jian")],
        "openaire": [rec("openaire", "2", "Deep Residual Learning", "Kaiming He; Jian Sun")],
    })
    assert len(works) == 1


def test_same_doi_with_url_prefix():
    works = works_of({
        "openalex": [rec("openalex", "W1", "X", doi="https://doi.org/10.1000/ABC")],
        "crossref": [rec("crossref", "10.1000/abc", "Y", doi="10.1000/abc")],
    })
    assert len(works) == 1


def test_generic_title_without_authors_is_not_merged():
    works = works_of({
        "crossref": [rec("crossref", "10.1/a", "Introduction", "Jane Roe", doi="10.1/a", date="2010")],
        "doaj": [rec("doaj", "x", "Introduction", "", date="2010")],
    })
    assert len(works) == 2


def test_part_one_vs_part_two_same_author_vetoed(store):
    works = works_of({
        "crossref": [
            rec("crossref", "10.1/p1", "Thermodynamics of black holes, Part I", "S. Hawking", doi="10.1/p1", date="1976"),
            rec("crossref", "10.1/p2", "Thermodynamics of black holes, Part II", "S. Hawking", doi="10.1/p2", date="1976"),
        ],
    }, store)
    assert len(works) == 2
    explanation = store.explain("10.1/p1", "10.1/p2")
    assert explanation["merged"] is False
    assert "numbers/parts differ" in explanation["reason"]


def test_three_records_linked_only_through_each_other():
    # A~B share a DOI, B~C share an arXiv id; A and C share nothing directly.
    a = rec("crossref", "10.1/x", "Some work", "Ada Lovelace", doi="10.1/x")
    b = rec("arxiv", "2101.00001v2", "Some different-looking title", "Ada Lovelace", doi="10.1/x")
    c = rec("semantic", "s2c", "Yet another wording entirely", "Ada Lovelace", arxiv_ids=["2101.00001"])
    for order in ([a, b, c], [c, b, a], [b, c, a]):
        works = works_of({r["source"]: [r] for r in order})
        assert len(works) == 1, order


# --- vetoes ----------------------------------------------------------------

@pytest.mark.parametrize("left,right,authors_b,date_b", [
    ("Deep learning for protein folding", "Erratum: Deep learning for protein folding", "A. Smith", "2020"),
    ("Deep learning for protein folding", "Corrigendum to deep learning for protein folding", "A. Smith", "2020"),
    ("Deep learning for protein folding", "Deep learning for protein folding", "B. Jones", "2020"),
    ("Deep learning for protein folding", "Deep learning for protein folding", "A. Smith", "2012"),
])
def test_hard_vetoes(left, right, authors_b, date_b):
    works = works_of({
        "core": [rec("core", "1", left, "A. Smith", date="2020")],
        "hal": [rec("hal", "2", right, authors_b, date=date_b)],
    })
    assert len(works) == 2


def test_veto_blocks_transitive_merge_through_cluster():
    # Two Part I records and one Part II: Part II must not join through a chain.
    works = works_of({
        "core": [rec("core", "1", "Hydrodynamics of stars Part I", "A. Smith", date="2000")],
        "hal": [rec("hal", "2", "Hydrodynamics of Stars, Part I", "A. Smith", date="2000")],
        "doaj": [rec("doaj", "3", "Hydrodynamics of stars Part II", "A. Smith", date="2000")],
    })
    assert sorted(len(w["records"]) for w in works) == [1, 2]


def test_result_independent_of_record_order():
    records = [
        rec("core", "1", "Graph neural networks: a review", "Jie Zhou; Ganqu Cui", date="2020"),
        rec("hal", "2", "Graph Neural Networks - A Review", "Zhou, Jie; Cui, Ganqu", date="2020"),
        rec("doaj", "3", "Graph neural networks: a review, Part 2", "Jie Zhou", date="2020"),
        rec("dblp", "4", "Graph neural networks: a review", "Jie Zhou", date="2021"),
    ]
    seen = set()
    for seed in range(5):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        works = works_of({r["source"]: [r] for r in shuffled})
        seen.add(frozenset(frozenset(x["paper_id"] for x in w["records"]) for w in works))
    assert len(seen) == 1


# --- conflicts, provenance, ranking -----------------------------------------

def test_conflicting_ids_are_quarantined_not_overwritten():
    works = works_of({
        "pubmed": [rec("pubmed", "111", "A cohort study of sleep", "Ann Lee", date="2019")],
        "europepmc": [rec("europepmc", "e1", "A cohort study of sleep", "Ann Lee", date="2019", pmid="222")],
    })
    assert len(works) == 1
    assert "pmid" not in works[0]["ids"]
    assert {e["value"] for e in works[0]["quarantined_ids"]["pmid"]} == {"111", "222"}


def test_per_field_provenance_prefers_published_record():
    works = works_of({
        "arxiv": [rec("arxiv", "2101.00001", "Preprint title", "Ada Lovelace", doi="10.1/pub")],
        "crossref": [rec("crossref", "10.1/pub", "Published title", "Ada Lovelace", doi="10.1/pub")],
    })
    assert works[0]["title"] == "Published title"
    assert works[0]["provenance"]["title"] == "crossref"


def test_rrf_ranks_fused_works_not_source_order():
    works = works_of({
        "arxiv": [rec("arxiv", "1", "Alpha study of things", "A. B"),
                  rec("arxiv", "2", "Beta study of stuff", "C. D", doi="10.1/beta")],
        "crossref": [rec("crossref", "10.1/beta", "Beta study of stuff", "C. D", doi="10.1/beta")],
    })
    assert [w["title"] for w in works] == ["Beta study of stuff", "Alpha study of things"]
    assert works[0]["rrf_score"] == round(1 / 62 + 1 / 61, 6)


# --- persistence ------------------------------------------------------------

def test_upid_is_stable_across_queries_and_merges_redirect(store):
    first = resolve({"arxiv": [rec("arxiv", "2101.00001", "T", "Ada Lovelace")]}, store)[0]["upid"]
    again = resolve({"arxiv": [rec("arxiv", "2101.00001v3", "T", "Ada Lovelace")]}, store)[0]["upid"]
    assert first == again
    other = resolve({"crossref": [rec("crossref", "10.1/pub", "U", "Ada Lovelace", doi="10.1/pub")]}, store)[0]["upid"]
    assert other != first
    # A later search links the two: one UPID survives, the other redirects to it.
    merged = resolve({"arxiv": [rec("arxiv", "2101.00001", "T", "Ada Lovelace", doi="10.1/pub")]}, store)[0]["upid"]
    assert merged in {first, other}
    assert store.describe(first)["upid"] == store.describe(other)["upid"] == merged
    # The crosswalk alone now merges the two source records in a fresh query.
    works = resolve({
        "arxiv": [rec("arxiv", "2101.00001", "T", "Ada Lovelace")],
        "crossref": [rec("crossref", "10.1/pub", "U", "Ada Lovelace", doi="10.1/pub")],
    }, store)
    assert len(works) == 1 and works[0]["upid"] == merged


def test_explain_match_reports_confidence_and_reason(store):
    resolve({
        "core": [rec("core", "1", "Graph neural networks: a review", "Jie Zhou; Ganqu Cui")],
        "hal": [rec("hal", "2", "Graph Neural Networks - A Review", "Zhou, Jie; Cui, Ganqu")],
    }, store)
    explanation = store.explain("src:core:1", "src:hal:2")
    assert explanation["merged"] is True
    assert 0.8 <= explanation["confidence"] <= 1.0
    assert "title similarity" in explanation["reason"]


# --- Unicode ----------------------------------------------------------------

def test_unicode_titles_keep_tokens_in_every_script():
    assert title_tokens("ആഴത്തിലുള്ള പഠനം") == ["ആഴത്തിലുള്ള", "പഠനം"]  # Malayalam
    assert title_tokens("गहन अधिगम का परिचय") == ["गहन", "अधिगम", "का", "परिचय"]  # Hindi
    assert normalize_title("深度学习：综述") == "深度学习 综述"
    assert normalize_title("ＢＥＲＴ") == "bert"  # full-width folds via NFKC
    assert title_similarity("深度学习的综述研究", "深度学习综述研究") >= 0.8


def test_non_latin_duplicates_merge():
    works = works_of({
        "core": [rec("core", "1", "深度学习在医学图像分析中的应用综述", "王伟; 李娜")],
        "hal": [rec("hal", "2", "深度学习在医学图像分析中的应用综述。", "王伟; 李娜")],
    })
    assert len(works) == 1


def test_author_keys():
    assert author_key("He, Kaiming") == author_key("Kaiming He") == ("he", "k")
    assert author_key("A. Vaswani") == ("vaswani", "a")
    assert author_key("Müller, Jörg") == author_key("Jorg Muller")


def test_persisted_wrong_merge_is_split_when_a_veto_appears(store):
    # First search: no years yet, so the two records merge on title and author.
    first = resolve({
        "core": [rec("core", "1", "Hydrodynamics of stellar interiors", "A. Smith")],
        "hal": [rec("hal", "2", "Hydrodynamics of Stellar Interiors", "A. Smith")],
    }, store)
    assert len(first) == 1
    upid = first[0]["upid"]
    # Later metadata reveals a 20-year gap: the stored merge must not survive it.
    later = resolve({
        "core": [rec("core", "1", "Hydrodynamics of stellar interiors", "A. Smith", date="1990")],
        "hal": [rec("hal", "2", "Hydrodynamics of Stellar Interiors", "A. Smith", date="2010")],
    }, store)
    assert len(later) == 2
    upids = {w["records"][0]["source"]: w["upid"] for w in later}
    assert upids["core"] == upid  # smallest record key keeps the UPID on a tie
    assert upids["hal"] != upid
    assert store.describe("src:hal:2")["split_from"] == [upid]
    explanation = store.explain("src:core:1", "src:hal:2")
    assert explanation["merged"] is False
    assert "year gap 20" in explanation["reason"]
    # The split is itself persistent: a year-less search no longer re-merges them.
    again = resolve({
        "core": [rec("core", "1", "Hydrodynamics of stellar interiors", "A. Smith")],
        "hal": [rec("hal", "2", "Hydrodynamics of Stellar Interiors", "A. Smith")],
    }, store)
    assert {w["upid"] for w in again} == set(upids.values())


def test_persisted_link_still_merges_records_without_a_veto(store):
    resolve({"arxiv": [rec("arxiv", "2101.00001", "T", "Ada Lovelace", doi="10.1/pub")]}, store)
    works = resolve({
        "arxiv": [rec("arxiv", "2101.00001", "T", "Ada Lovelace", date="2021")],
        "crossref": [rec("crossref", "10.1/pub", "U", "Ada Lovelace", doi="10.1/pub", date="2022")],
    }, store)
    assert len(works) == 1
