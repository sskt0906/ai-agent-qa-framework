import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from .core import Store, ScriptedModel, OpenAIModel, run_agent, evaluate


def run_trial(case, trial, args):
    row = {"case": case["id"], "trial": trial}
    store = None
    try:
        store = Store(**case.get("store", {}))
        before = row["before"] = store.full_snapshot()
        model = ScriptedModel(case["script"]) if args.mode == "mock" else OpenAIModel(args.model)
        run = run_agent(model, case["prompt"], store, request=case.get("request"))
        row["run"] = asdict(run)
        after = row["after"] = store.full_snapshot()
        row["business_updated"] = before != after
        if run.error:
            row["error"] = run.error
        else:
            row["checks"] = evaluate(case, run, before, after)
    except Exception as exc:
        row["error"] = type(exc).__name__
    finally:
        if store is not None:
            store.close()
    return row


def rates(completed, denominator):
    if not denominator:
        return {}
    metrics = ("passed", "answer_correct", "grounded", "tool_policy", "db_integrity", "bounded")
    return {key: sum(row["checks"][key] for row in completed) / denominator for key in metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output", default="reports/evaluation.json")
    args = p.parse_args()
    if args.repeat < 1 or (args.mode == "live" and (not args.model or not os.getenv("OPENAI_API_KEY"))):
        p.error("repeat >= 1; live requires OPENAI_API_KEY and --model")
    cases = json.loads(Path("datasets/cases.json").read_text())
    rows = [
        run_trial(case, trial + 1, args)
        for case in cases
        for trial in range(args.repeat)
    ]
    completed = [row for row in rows if "checks" in row]
    report = {
        "mode": args.mode,
        "model": args.model if args.mode == "live" else "scripted-fixture",
        "total": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "rates_on_completed": rates(completed, len(completed)),
        "rates_on_all_trials": rates(completed, len(rows)),
        "results": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return int(not rows or len(completed) != len(rows) or not all(r["checks"]["passed"] for r in completed))


if __name__ == "__main__":
    raise SystemExit(main())
