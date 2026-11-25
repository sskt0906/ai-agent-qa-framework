import json
import sys
from pathlib import Path

import pytest

from qa_agent import cli
from qa_agent.core import ScriptedModel, Store, answer, evaluate, run_agent


CASES = json.loads(Path("datasets/cases.json").read_text())


class Model:
    def __init__(self, *steps):
        self.steps = iter(steps)

    def step(self, prompt, feedback):
        event = next(self.steps)
        if isinstance(event, Exception):
            raise event
        return event


def calls(*operations):
    return None, [(str(i), name, {"order_id": oid})
                  for i, (name, oid) in enumerate(operations)]


@pytest.fixture
def store():
    db = Store()
    yield db
    db.close()


@pytest.mark.parametrize(("operation", "target", "steps"), [
    ("lookup", 123, [calls(("lookup_order", 123)), calls(("refund_order", 123))]),
    ("refund", None, [calls(("lookup_order", 123)), calls(("refund_order", 123))]),
    ("refund", 123, [calls(("lookup_order", 456)), calls(("refund_order", 123))]),
    ("refund", 999, [calls(("lookup_order", 999)), calls(("refund_order", 999))]),
    ("refund", 123, [calls(("lookup_order", 123), ("refund_order", 123))]),
], ids=["read-only", "missing-id", "wrong-lookup", "not-found", "same-response"])
def test_refund_guard(store, operation, target, steps, monkeypatch):
    original_call = store.call

    def lookup_only(name, args):
        assert name != "refund_order", "blocked refund reached Store"
        return original_call(name, args)

    monkeypatch.setattr(store, "call", lookup_only)
    before = store.full_snapshot()
    model = Model(*steps, (answer("forbidden", target), []))
    run = run_agent(model, "注文123を返金して", store,
                    request={"operation": operation, "order_id": target})
    assert store.full_snapshot() == before
    assert run.trace[-1].name == "refund_order"
    assert run.trace[-1].blocked
    assert run.trace[-1].result["status"] == "forbidden"


@pytest.mark.parametrize("mutation", [
    "UPDATE orders SET refunded=0 WHERE id=123",
    "UPDATE orders SET amount=1 WHERE id=123",
    "UPDATE orders SET eligible=1 WHERE id=456",
], ids=["refund-flag", "amount", "other-order"])
def test_evaluator_checks_both_tables(store, mutation):
    case = CASES[0]
    before = store.full_snapshot()
    run = run_agent(ScriptedModel(case["script"]), case["prompt"], store,
                    request={"operation": "refund", "order_id": 123})
    assert evaluate(case, run, before, store.full_snapshot())["passed"]
    store.db.execute(mutation)
    checks = evaluate(case, run, before, store.full_snapshot())
    assert not checks["db_integrity"]
    assert not checks["passed"]


def test_transaction_failure_rolls_back_and_is_audited(store):
    store.db.execute("""CREATE TRIGGER abort_update BEFORE UPDATE ON orders
                     BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    before = store.full_snapshot()
    model = Model(calls(("lookup_order", 123)), calls(("refund_order", 123)))
    run = run_agent(model, "", store, request={"operation": "refund", "order_id": 123})
    assert store.full_snapshot() == before
    assert run.answer is None
    assert run.error == "IntegrityError"
    assert run.trace[-1].name == "refund_order"
    assert run.trace[-1].result["status"] == "execution_error"


@pytest.fixture
def failed_report(monkeypatch, tmp_path):
    original = ScriptedModel

    def model(steps):
        if steps[0].get("args") == {"order_id": 123} and steps[-1]["final"]["status"] == "refunded":
            return Model(calls(("lookup_order", 123)), calls(("refund_order", 123)),
                         RuntimeError("injected"))
        return original(steps)

    output = tmp_path / "report.json"
    monkeypatch.setattr(cli, "ScriptedModel", model)
    monkeypatch.setattr(sys, "argv", ["qa", "--mode", "mock", "--output", str(output)])
    code = cli.main()
    report = json.loads(output.read_text())
    return code, report, report["results"][0]


def test_model_failure_distinguishes_update_from_answer(failed_report):
    _, _, row = failed_report
    assert row["error"] == "RuntimeError"
    assert row["run"]["answer"] is None
    assert row["business_updated"] is True


def test_model_failure_preserves_trace(failed_report):
    _, _, row = failed_report
    assert [call["name"] for call in row["run"]["trace"]] == ["lookup_order", "refund_order"]
    assert row["run"]["trace"][-1]["result"]["status"] == "refunded"


def test_model_failure_preserves_db_changes(failed_report):
    _, _, row = failed_report
    assert "123" not in row["before"]["refunds"]
    assert row["after"]["refunds"]["123"] == 3000
    assert row["after"]["orders"]["123"]["refunded"] is True


def test_execution_errors_count_in_all_trials(failed_report):
    code, report, _ = failed_report
    assert code == 1
    assert (report["total"], report["completed"], report["errors"]) == (9, 8, 1)
    assert report["rates_on_completed"]["passed"] == 1
    assert report["rates_on_all_trials"]["passed"] == pytest.approx(8 / 9)
