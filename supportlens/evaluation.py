"""Development retrieval evaluation. Live runs use the configured real embedder."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import statistics
import tempfile


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Each evaluation case must contain a relevance label.")
    # Repeated retrievals do not earn extra credit.
    ranked = list(dict.fromkeys(ranked_ids))
    return {
        "precision_at_3": len(set(ranked[:3]) & relevant) / 3,
        "recall_at_5": len(set(ranked[:5]) & relevant) / len(relevant),
        "mrr_at_5": next((1 / rank for rank, item in enumerate(ranked[:5], 1) if item in relevant), 0.0),
    }


def evaluate(store, cases: list[dict] | None = None, modes: tuple[str, ...] = ("keyword", "vector", "hybrid")) -> dict:
    """Evaluate already imported data; no query implicitly changes the index."""
    if cases is None:
        cases = json.loads(files("supportlens").joinpath("fixtures/evaluation.json").read_text())
    output = {"kind": "development_retrieval", "case_count": len(cases), "dataset": store.metadata(), "modes": {}}
    for mode in modes:
        rows = []
        for case in cases:
            found = store.search(case["query"], mode=mode, filters=case.get("filters", {}), limit=5)
            ids = [result["ticket"]["id"] for result in found["results"]]
            rows.append({"id": case["id"], "query": case["query"], "filters": case.get("filters", {}),
                         "relevant_ids": case["relevant_ids"], "ranked_ids": ids,
                         "timing_ms": found["timing_ms"], **retrieval_metrics(ids, case["relevant_ids"])})
        keys = ("precision_at_3", "recall_at_5", "mrr_at_5", "timing_ms")
        output["modes"][mode] = {"mean": {key: statistics.mean(row[key] for row in rows) if rows else 0.0 for key in keys}, "cases": rows}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SupportLens retrieval with real local embeddings.")
    parser.add_argument("--output", type=Path, required=True, help="Save the measured JSON report here.")
    parser.add_argument("--modes", nargs="+", choices=["keyword", "vector", "hybrid"], default=["keyword", "vector", "hybrid"])
    args = parser.parse_args()
    from supportlens.store import Store
    # A fresh index makes the report reproducible and cannot change the user's workspace.
    with tempfile.TemporaryDirectory(prefix="supportlens-evaluation-") as directory:
        store = Store(Path(directory))
        try:
            store.load_sample()
            report = evaluate(store, modes=tuple(args.modes))
            report["embedding_model"] = store.embedder.model
        finally:
            store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: report["modes"][mode]["mean"] for mode in args.modes}, indent=2))


if __name__ == "__main__":
    main()
