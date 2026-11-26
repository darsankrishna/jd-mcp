"""The dedup benchmark data, trained model and system ports stay reproducible."""

import json

import pytest

from benchmarks.dedup import adapters, generate, handbuilt, run, train
from paper_search_mcp import identity


def test_committed_data_matches_the_generators():
    with open(generate.DATA / "generated.jsonl", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == generate.generate()
    assert json.loads((generate.DATA / "handbuilt.json").read_text(encoding="utf-8")) == handbuilt.CASES


def test_train_and_test_splits_share_no_seed_work():
    seed = lambda case: case["id"].split("-")[1]
    train_seeds = {seed(c) for c in run.load_cases("train")}
    test_seeds = {seed(c) for c in run.load_cases("test") if c["id"].startswith("gen-")}
    assert train_seeds and test_seeds and not train_seeds & test_seeds


def test_shipped_model_is_reproducible_from_the_train_split():
    xs, ys = train.classifier_pairs("train")
    weights, bias = train.fit(xs, ys)
    assert identity.MODEL["weights"] == pytest.approx(weights, abs=1e-9)
    assert identity.MODEL["bias"] == pytest.approx(bias, abs=1e-9)
    assert identity.MODEL["features"] == list(identity.FEATURES)


def test_classifier_respects_feature_signs():
    by_name = dict(zip(identity.FEATURES, identity.MODEL["weights"]))
    assert by_name.pop("year_gap") <= 0
    assert all(weight >= 0 for weight in by_name.values())


def labels(system, case_id):
    case = next(c for c in handbuilt.CASES if c["id"] == case_id)
    found = system(case["records"])
    return found[0] == found[1]


@pytest.mark.parametrize("case_id,merged", [
    ("hb-punctuation", False), ("hb-comma", False), ("hb-doi-prefix", False),
])
def test_upstream_port_reproduces_the_analysis(case_id, merged):
    assert labels(adapters.paper_search_mcp, case_id) is merged


@pytest.mark.parametrize("case_id,merged", [
    ("hb-resnet-dois", False), ("hb-initials", False),
    ("hb-introduction", True), ("hb-part-one-two", True),  # the two wrong merges
])
def test_scholarmcp_port_reproduces_the_analysis(case_id, merged):
    assert labels(adapters.scholarmcp, case_id) is merged


def test_scholarmcp_port_misses_part_of_the_three_record_chain():
    case = next(c for c in handbuilt.CASES if c["id"] == "hb-chain")
    assert len(set(adapters.scholarmcp(case["records"]))) == 2  # analysis: 3 -> 2


def test_jd_mcp_gets_every_handbuilt_case_right():
    result = run.score(adapters.jd_mcp, handbuilt.CASES)["all"]
    assert result["fp"] == 0 and result["fn"] == 0


def test_live_pairs_use_s2_labels_and_reviews_override():
    case = {"records": [{}, {}, {}, {}], "gold": ["P1", "P1", "P2", None],
            "review": {(0, 1): "different", (2, 3): "same", (0, 2): "unsure"}}
    assert sorted(run.gold_pairs(case)) == [
        (0, 1, False),  # review overrides S2
        (1, 2, False),  # S2: different papers
        (2, 3, True),   # unlabelled record scored only through review
    ]


@pytest.mark.parametrize("split", ["live", "live-test"])
def test_committed_live_data_is_fully_labelled_and_reviewable(split):
    live = generate.DATA / split
    if not (live / "records.jsonl").exists():
        pytest.skip("live capture not committed")
    cases = run.load_cases(split)
    assert cases and all(len(c["gold"]) == len(c["records"]) for c in cases)
    assert any(any(c["gold"]) for c in cases)
