"""Generate the constructed deduplication gold set.

Every record is synthetic: titles come from word lists, DOIs use the 10.5555
test prefix, and labels hold by construction. Seeds with an even index go to
the train split, odd ones to test, so a work never appears in both.

    uv run python -m benchmarks.dedup.generate
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).with_name("data")

METHODS = ["Residual", "Sparse", "Graph", "Contrastive", "Bayesian", "Recurrent", "Federated",
           "Causal", "Diffusion", "Spectral", "Adversarial", "Hierarchical", "Variational", "Attentive"]
OBJECTS = ["Networks", "Embeddings", "Transformers", "Kernels", "Autoencoders", "Policies",
           "Estimators", "Models", "Encoders", "Ensembles"]
TASKS = ["Image Recognition", "Protein Folding", "Speech Separation", "Traffic Forecasting",
         "Code Summarization", "Drug Discovery", "Climate Downscaling", "Legal Retrieval",
         "Crop Yield Prediction", "Machine Translation", "Anomaly Detection", "Question Answering"]
GIVEN = ["Ada", "Kaiming", "Noam", "Priya", "José", "Mei", "Olga", "Tomás", "Aisha", "Lars",
         "Hiro", "Zoë", "Ravi", "Ingrid", "Chidi", "Sofía", "Jonas", "Leila"]
SURNAMES = ["Lovelace", "He", "Shazeer", "Raman", "Muñoz", "Chen", "Ivanova", "Kovář", "Okafor",
            "Lindqvist", "Tanaka", "Müller", "Iyer", "Berg", "Nwosu", "García", "Weber", "Haddad"]
VENUES = ["CVPR", "NeurIPS", "ICML", "Nature Methods", "ACL", "KDD", "Bioinformatics", "ICLR"]
NON_LATIN = [
    ("深度残差网络在医学图像分割中的应用", ["王伟", "李娜"]),
    ("हिंदी पाठ वर्गीकरण के लिए गहन शिक्षण", ["अमित शर्मा", "प्रिया वर्मा"]),
    ("മലയാളം സംഭാഷണ തിരിച്ചറിയലിനുള്ള ആഴത്തിലുള്ള പഠനം", ["അനിൽ കുമാർ", "ലക്ഷ്മി നായർ"]),
    ("Глубокое обучение для прогнозирования трафика", ["Иван Петров", "Анна Смирнова"]),
]
POSITIVE = ["doi_prefix", "punctuation_case", "author_comma", "author_initials", "arxiv_doi_field",
            "arxiv_doi_vs_venue_doi", "journal_ref_reworded", "crossref_relation", "chain_three",
            "accents_dropped", "hyphenation"]
NEGATIVE = ["part_numbers", "erratum", "different_authors", "year_gap", "generic_title",
            "sibling_task", "one_word_method", "numbered_sequel"]


def record(source, paper_id, title, authors, doi="", year=None, **extra):
    return {"source": source, "paper_id": paper_id, "title": title, "authors": "; ".join(authors),
            "doi": doi, "published_date": str(year) if year else "", "abstract": "", "url": "",
            "pdf_url": "", "citations": 0, "extra": extra}


def strip_accents(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")


def seed_works(rng):
    works = []
    for n in range(80):
        method, obj, task = rng.choice(METHODS), rng.choice(OBJECTS), rng.choice(TASKS)
        names = rng.sample(range(len(GIVEN)), rng.randint(2, 4))
        works.append({
            "n": n, "method": method, "object": obj, "task": task,
            "title": f"{method} {obj} for {task}",
            "authors": [f"{GIVEN[i]} {SURNAMES[(i + n) % len(SURNAMES)]}" for i in names],
            "year": rng.randint(2012, 2024), "doi": f"10.5555/bench.{n:04d}",
            "arxiv": f"{rng.randint(15, 24)}{rng.randint(1, 12):02d}.{n:05d}", "venue": rng.choice(VENUES),
        })
    for k, (title, authors) in enumerate(NON_LATIN):
        works.append({"n": 80 + k, "method": "", "object": "", "task": "", "title": title, "authors": authors,
                      "year": 2020 + k, "doi": f"10.5555/bench.{80 + k:04d}", "arxiv": f"2{k}01.{80 + k:05d}",
                      "venue": rng.choice(VENUES)})
    return works


def positive(kind, w, rng):
    t, a, y, d = w["title"], w["authors"], w["year"], w["doi"]
    if kind == "doi_prefix":
        return [record("crossref", d, t, a, d, y), record("openalex", f"W{w['n']}", t, a, "https://doi.org/" + d.upper(), y)]
    if kind == "punctuation_case":
        return [record("core", f"c{w['n']}", t + ": A Study", a, year=y),
                record("hal", f"h{w['n']}", (t + " - a study.").lower(), a, year=y)]
    if kind == "author_comma":
        comma = [f"{n.split()[-1]}, {' '.join(n.split()[:-1])}" for n in a]
        return [record("dblp", f"d{w['n']}", t, a, year=y), record("europepmc", f"e{w['n']}", t, comma, year=y)]
    if kind == "author_initials":
        initials = [f"{n.split()[0][0]}. {n.split()[-1]}" for n in a]
        return [record("core", f"c{w['n']}", t, a, year=y), record("doaj", f"j{w['n']}", t, initials, year=y)]
    if kind == "arxiv_doi_field":
        return [record("arxiv", w["arxiv"] + "v2", "Towards " + t, a, d, y),
                record("crossref", d, t, a, d, y + 1)]
    if kind == "arxiv_doi_vs_venue_doi":
        return [record("semantic", f"s2{w['n']}", t, a, f"10.48550/arXiv.{w['arxiv']}", y),
                record("crossref", d, t, a, d, y + 1)]
    if kind == "journal_ref_reworded":
        reworded = f"{w['object']} with {w['method']} Structure for {w['task']}" if w["method"] else t + " (extended)"
        return [record("arxiv", w["arxiv"] + "v1", reworded, a, year=y, journal_ref=f"{w['venue']} {y + 1}"),
                record("dblp", f"d{w['n']}", t, a, year=y + 1, container_title=w["venue"])]
    if kind == "crossref_relation":
        relation = {"is-preprint-of": [{"id-type": "doi", "id": d}]}
        return [record("crossref", f"10.1101/bench.{w['n']}", "Preprint: " + t, a, f"10.1101/bench.{w['n']}", y,
                       crossref_type="posted-content", relation=relation),
                record("crossref", d, "Revised " + t, a, d, y + 1)]
    if kind == "chain_three":
        return [record("crossref", d, t, a, d, y), record("arxiv", w["arxiv"], "Notes on " + t, a, d, y),
                record("semantic", f"s2{w['n']}", "An Unrelated Sounding Name", a[:1], year=y, arxiv_ids=[w["arxiv"]])]
    if kind == "accents_dropped":
        if not w["method"]:
            # Indic vowel signs and viramas are marks too; no source strips them.
            return positive("author_comma", w, rng)
        return [record("openaire", f"o{w['n']}", t, a, year=y),
                record("base", f"b{w['n']}", strip_accents(t), [strip_accents(n) for n in a], year=y)]
    if kind == "hyphenation":
        words = t.split()
        i = rng.randrange(len(words) - 1) if len(words) > 1 else 0
        hyphenated = " ".join(words[:i] + [words[i] + "-" + words[i + 1]] + words[i + 2:]) if len(words) > 1 else t + "-"
        return [record("zenodo", f"z{w['n']}", t, a, year=y), record("hal", f"h{w['n']}", hyphenated, a, year=y)]
    raise ValueError(kind)


def negative(kind, w, rng):
    t, a, y = w["title"], w["authors"], w["year"]
    if kind == "part_numbers":
        if rng.random() < 0.5:  # repository records often carry no DOI
            return [record("core", f"c{w['n']}", t + ", Part I", a, year=y),
                    record("hal", f"h{w['n']}", t + ", Part II", a, year=y)]
        return [record("crossref", f"10.5555/p1.{w['n']}", t + ", Part I", a, f"10.5555/p1.{w['n']}", y),
                record("crossref", f"10.5555/p2.{w['n']}", t + ", Part II", a, f"10.5555/p2.{w['n']}", y)]
    if kind == "erratum":
        return [record("core", f"c{w['n']}", t, a, year=y), record("hal", f"h{w['n']}", "Erratum: " + t, a, year=y)]
    if kind == "different_authors":
        other = [f"{g} {s}" for g, s in zip(rng.sample(GIVEN, 2), rng.sample(SURNAMES, 2)) if f"{g} {s}" not in a]
        return [record("core", f"c{w['n']}", t, a, year=y), record("doaj", f"j{w['n']}", t, other or ["Q. Nobody"], year=y)]
    if kind == "year_gap":
        return [record("dblp", f"d{w['n']}", t, a, year=y), record("openaire", f"o{w['n']}", t, a, year=y + 8)]
    if kind == "generic_title":
        return [record("crossref", f"10.5555/intro.{w['n']}", "Introduction", a, f"10.5555/intro.{w['n']}", y),
                record("doaj", f"j{w['n']}", "Introduction", [], year=y)]
    if kind == "sibling_task":
        if not w["task"]:
            return negative("erratum", w, rng)
        other = rng.choice([task for task in TASKS if task != w["task"]])
        return [record("core", f"c{w['n']}", t, a, year=y),
                record("hal", f"h{w['n']}", f"{w['method']} {w['object']} for {other}", a, year=y)]
    if kind == "one_word_method":
        if not w["method"]:
            return negative("year_gap", w, rng)
        other = rng.choice([m for m in METHODS if m != w["method"]])
        return [record("core", f"c{w['n']}", t, a, year=y),
                record("hal", f"h{w['n']}", f"{other} {w['object']} for {w['task']}", a, year=y)]
    if kind == "numbered_sequel":
        return [record("core", f"c{w['n']}", t, a, year=y), record("hal", f"h{w['n']}", t + " 2", a, year=y + 1)]
    raise ValueError(kind)


# Cases the classifier decides (no shared identifier). Their second record gets
# realistic noise so features vary within each label, not just between labels.
CLASSIFIED_POSITIVE = {"punctuation_case", "author_comma", "author_initials", "accents_dropped",
                       "hyphenation", "journal_ref_reworded", "arxiv_doi_vs_venue_doi"}
CLASSIFIED_NEGATIVE = {"sibling_task", "one_word_method"}


def vary(records, rng, positive_case):
    other = records[1]
    authors = other["authors"].split("; ")
    if positive_case:
        if len(authors) > 1 and rng.random() < 0.3:  # truncated author list
            authors = authors[: rng.randint(1, len(authors) - 1)]
        shift = rng.choice([0, 0, 1])
    else:
        roll = rng.random()
        if roll < 0.3 and len(authors) > 1:
            authors = authors[1:]  # follow-up paper by part of the team
        elif roll < 0.6:
            authors = authors + [f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"]
        shift = rng.choice([0, 1, 2])
    other["authors"] = "; ".join(authors)
    if other["published_date"] and records[0]["published_date"]:
        other["published_date"] = str(int(records[0]["published_date"]) + shift)
    return records


def generate(seed: int = 7):
    rng = random.Random(seed)
    cases = []
    for w in seed_works(rng):
        split = "train" if w["n"] % 2 == 0 else "test"
        for kind in rng.sample(POSITIVE, 4):
            records = positive(kind, w, rng)
            if kind in CLASSIFIED_POSITIVE:
                vary(records, rng, True)
            cases.append({"id": f"gen-{w['n']}-{kind}", "split": split, "category": kind, "label": "same",
                          "records": records, "clusters": [list(range(len(records)))]})
        for kind in rng.sample(NEGATIVE, 3):
            records = negative(kind, w, rng)
            if kind in CLASSIFIED_NEGATIVE:
                vary(records, rng, False)
            cases.append({"id": f"gen-{w['n']}-{kind}", "split": split, "category": kind, "label": "different",
                          "records": records, "clusters": [[i] for i in range(len(records))]})
    return cases


def main() -> None:
    DATA.mkdir(exist_ok=True)
    cases = generate()
    with open(DATA / "generated.jsonl", "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {DATA / 'generated.jsonl'}")


if __name__ == "__main__":
    main()
