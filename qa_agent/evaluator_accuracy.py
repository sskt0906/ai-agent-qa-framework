"""Measure structured grounding against independently labeled synthetic evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .core import Call, grounded_answer


def classify(example, snapshots):
    def snapshot(name):
        return {
            table: {int(oid): value for oid, value in records.items()}
            for table, records in snapshots[example[name]].items()
        }

    return not grounded_answer(
        example["answer"],
        [Call(**call) for call in example["trace"]],
        snapshot("before"),
        snapshot("after"),
        example["request"],
    )


def metrics(*, tp, tn, fp, fn):
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
    }


def measure(dataset):
    outcomes = {(True, True): "tp", (False, False): "tn",
                (False, True): "fp", (True, False): "fn"}
    results = []
    counts = Counter()
    for example in dataset["examples"]:
        expected = example["hallucinated"]
        if type(expected) is not bool:
            raise ValueError(f"{example['id']}: hallucinated must be boolean")
        predicted = classify(example, dataset["snapshots"])
        outcome = outcomes[expected, predicted]
        counts[outcome] += 1
        results.append({
            "id": example["id"], "category": example["category"],
            "expected_hallucinated": expected, "predicted_hallucinated": predicted,
            "outcome": outcome,
        })
    return {
        "mode": "synthetic_evaluator_accuracy",
        "positive_class": "hallucinated",
        "total": len(results),
        "metrics": metrics(**{key: counts[key] for key in outcomes.values()}),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/evaluator_accuracy.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluator-accuracy.json"))
    args = parser.parse_args()
    report = measure(json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return int(not report["total"] or report["metrics"]["fp"] or report["metrics"]["fn"])


if __name__ == "__main__":
    raise SystemExit(main())
