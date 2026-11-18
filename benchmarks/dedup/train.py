"""Train the pairwise match classifier and write paper_search_mcp/match_model.json.

    uv run python -m benchmarks.dedup.train

Logistic regression by projected batch gradient descent with L2 and per-feature
sign constraints, standard library only.
It trains on the train split, on exactly the pairs the classifier decides in
``resolve``: both author lists present, no generic title, no hard veto, and no
identifier link, direct or through a chain (identifier edges merge those).
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import List, Tuple

from paper_search_mcp.identity import (
    FEATURES, MODEL_PATH, _Features, match_probability, pair_features, record_identifiers, veto,
)

from .run import gold_pairs, load_cases

EPOCHS = 20000
LEARNING_RATE = 0.5
L2 = 1e-3
# Sign constraints (projected gradient descent): more title or author agreement
# can never lower the match probability, a larger year gap can never raise it.
# A small constructed set otherwise rewards its generator's quirks.
SIGNS = {"year_gap": -1}


def identifier_components(records) -> List[int]:
    """Component per record over shared identifiers, chains included (A~B~C)."""
    ids = [{k for k, _, _ in record_identifiers(r) if not k.startswith("src:")} for r in records]
    component = list(range(len(records)))
    changed = True
    while changed:
        changed = False
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if ids[i] & ids[j] and component[i] != component[j]:
                    low = min(component[i], component[j])
                    component = [low if c in (component[i], component[j]) else c for c in component]
                    changed = True
    return component


def classifier_pairs(split: str) -> Tuple[List[List[float]], List[int]]:
    xs, ys = [], []
    for case in load_cases(split):
        records = case["records"]
        features = [_Features(r) for r in records]
        linked = identifier_components(records)
        for i, j, same in gold_pairs(case):
            a, b = features[i], features[j]
            if a.generic or b.generic or not a.authors or not b.authors or veto(a, b) or linked[i] == linked[j]:
                continue
            xs.append(pair_features(a, b))
            ys.append(int(same))
    return xs, ys


def fit(xs: List[List[float]], ys: List[int]) -> Tuple[List[float], float]:
    weights, bias, n = [0.0] * len(FEATURES), 0.0, len(xs)
    # Balanced class weights: hard negatives are the minority but cost wrong merges.
    class_weight = {label: n / (2 * ys.count(label)) for label in (0, 1)}
    for _ in range(EPOCHS):
        grad_w, grad_b = [0.0] * len(weights), 0.0
        for x, y in zip(xs, ys):
            error = class_weight[y] * (
                1.0 / (1.0 + math.exp(-(bias + sum(w * v for w, v in zip(weights, x))))) - y
            )
            grad_b += error
            for k, value in enumerate(x):
                grad_w[k] += error * value
        bias -= LEARNING_RATE * grad_b / n
        weights = [w - LEARNING_RATE * (g / n + L2 * w) for w, g in zip(weights, grad_w)]
        weights = [max(0.0, w) if SIGNS.get(name, 1) > 0 else min(0.0, w) for name, w in zip(FEATURES, weights)]
    return weights, bias


def confusion(model, xs, ys, threshold: float) -> Tuple[int, int, int, int]:
    predicted = [match_probability(x, model) >= threshold for x in xs]
    tp = sum(p and y for p, y in zip(predicted, ys))
    fp = sum(p and not y for p, y in zip(predicted, ys))
    fn = sum(not p and y for p, y in zip(predicted, ys))
    return tp, fp, fn, len(ys) - tp - fp - fn


def mcc_at(model, xs, ys, threshold: float) -> float:
    """Matthews correlation: unlike F1 it does not reward merging everything when
    positives outnumber negatives, and wrong merges are the costlier error."""
    tp, fp, fn, tn = confusion(model, xs, ys, threshold)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def main() -> None:
    xs, ys = classifier_pairs("train")
    weights, bias = fit(xs, ys)
    model = {"features": list(FEATURES), "weights": weights, "bias": bias}
    # Best train MCC; ties go to the threshold closest to 0.5.
    candidates = [round(0.05 * k, 2) for k in range(2, 19)]
    threshold = max(candidates, key=lambda t: (round(mcc_at(model, xs, ys, t), 6), -abs(t - 0.5)))
    tp, fp, fn, tn = confusion(model, xs, ys, threshold)
    model.update({
        "threshold": threshold,
        "trained_on": {
            "data": "benchmarks/dedup constructed gold set, train split",
            "pairs": len(xs), "positives": sum(ys),
            "train_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "train_mcc": round(mcc_at(model, xs, ys, threshold), 4),
            "date": date.today().isoformat(),
        },
    })
    MODEL_PATH.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(model, indent=1))


if __name__ == "__main__":
    main()
