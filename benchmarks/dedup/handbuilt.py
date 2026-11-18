"""Hand-built cases mirroring the base-repo analysis table (test split only).

    uv run python -m benchmarks.dedup.handbuilt
"""

import json

from .generate import DATA, record

RESNET = ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]


def case(case_id, category, records, clusters):
    label = "same" if len(clusters) == 1 else "different"
    return {"id": case_id, "split": "test", "category": category, "label": label,
            "records": records, "clusters": clusters}


CASES = [
    case("hb-preprint-reworded", "arxiv_doi_field", [
        record("arxiv", "1706.03762v5", "Attention Is All You Need", ["Ashish Vaswani", "Noam Shazeer"],
               "10.5555/3295222.3295349", 2017),
        record("crossref", "10.5555/3295222.3295349", "Attention is all you need: transformers for sequence transduction",
               ["Vaswani, Ashish", "Shazeer, Noam"], "10.5555/3295222.3295349", 2017),
    ], [[0, 1]]),
    case("hb-resnet-dois", "arxiv_doi_vs_venue_doi", [
        record("semantic", "s2resnet", "Deep Residual Learning for Image Recognition", RESNET,
               "10.48550/arXiv.1512.03385", 2015),
        record("crossref", "10.1109/CVPR.2016.90", "Deep Residual Learning for Image Recognition", RESNET,
               "10.1109/CVPR.2016.90", 2016),
    ], [[0, 1]]),
    case("hb-punctuation", "punctuation_case", [
        record("core", "1", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
               ["Jacob Devlin", "Ming-Wei Chang"]),
        record("hal", "2", "bert pre training of deep bidirectional transformers for language understanding.",
               ["Jacob Devlin", "Ming-Wei Chang"]),
    ], [[0, 1]]),
    case("hb-initials", "author_initials", [
        record("dblp", "1", "Attention Is All You Need", ["A. Vaswani", "N. Shazeer"]),
        record("core", "2", "Attention is all you need", ["Ashish Vaswani", "Noam Shazeer"]),
    ], [[0, 1]]),
    case("hb-comma", "author_comma", [
        record("europepmc", "1", "Deep residual learning", ["He, Kaiming", "Sun, Jian"]),
        record("openaire", "2", "Deep Residual Learning", ["Kaiming He", "Jian Sun"]),
    ], [[0, 1]]),
    case("hb-doi-prefix", "doi_prefix", [
        record("openalex", "W1", "X", [], "https://doi.org/10.1000/ABC"),
        record("crossref", "10.1000/abc", "Y", [], "10.1000/abc"),
    ], [[0, 1]]),
    case("hb-introduction", "generic_title", [
        record("crossref", "10.1/a", "Introduction", ["Jane Roe"], "10.1/a", 2010),
        record("doaj", "x", "Introduction", [], year=2010),
    ], [[0], [1]]),
    case("hb-part-one-two", "part_numbers", [
        record("core", "p1", "Thermodynamics of black holes, Part I", ["S. Hawking"], year=1976),
        record("hal", "p2", "Thermodynamics of black holes, Part II", ["S. Hawking"], year=1976),
    ], [[0], [1]]),
    case("hb-chain", "chain_three", [
        record("crossref", "10.1/x", "Some work", ["Ada Lovelace"], "10.1/x"),
        record("arxiv", "2101.00001v2", "Some different-looking title", ["Ada Lovelace"], "10.1/x"),
        record("semantic", "s2c", "Yet another wording entirely", ["Ada Lovelace"], arxiv_ids=["2101.00001"]),
    ], [[0, 1, 2]]),
]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "handbuilt.json"
    path.write_text(json.dumps(CASES, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(CASES)} cases to {path}")


if __name__ == "__main__":
    main()
