from pathlib import Path
import json

import pytest

from qa_agent.core import Call, Run, evaluate

DATA = json.loads(Path("datasets/evaluator_accuracy.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", DATA["examples"], ids=lambda row: row["id"])
def test_grounding_against_manual_label(row):
    from qa_agent.evaluator_accuracy import classify

    assert classify(row, DATA["snapshots"]) == row["hallucinated"]


def test_grounding_rejects_amount_from_superseded_lookup():
    trace = [
        Call("lookup_order", {"order_id": 123}, {
            "status": "found", "order_id": 123, "amount": 2500,
        }),
        Call("lookup_order", {"order_id": 123}, {
            "status": "found", "order_id": 123, "amount": 3000,
        }),
    ]
    response = {"status": "found", "order_id": 123, "amount": 2500}
    case = {
        "request": {"operation": "lookup", "order_id": 123},
        "expected": {"status": "found", "order_id": 123, "amount": 3000},
        "allowed_trace": [{"tool": call.name, "args": call.args} for call in trace],
    }
    snapshot = {
        "orders": {123: {"amount": 3000, "eligible": True, "refunded": False}},
        "refunds": {},
    }
    checks = evaluate(case, Run(response, trace), snapshot, snapshot)
    assert checks["tool_policy"] and checks["db_integrity"]
    assert not checks["grounded"]


def test_confusion_metrics_with_all_outcomes():
    from qa_agent.evaluator_accuracy import metrics

    assert metrics(tp=3, tn=5, fp=1, fn=2) == {
        "tp": 3, "tn": 5, "fp": 1, "fn": 2,
        "precision": 3 / 4, "recall": 3 / 5, "f1": 6 / 9,
        "false_positive_rate": 1 / 6, "false_negative_rate": 2 / 5,
    }


def test_confusion_metrics_with_empty_denominators():
    from qa_agent.evaluator_accuracy import metrics

    assert metrics(tp=0, tn=0, fp=0, fn=0) == dict.fromkeys(
        ("tp", "tn", "fp", "fn"), 0
    ) | dict.fromkeys(
        ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate")
    )
