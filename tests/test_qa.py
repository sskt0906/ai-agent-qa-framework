import json
from pathlib import Path
import pytest
from qa_agent.core import Store, ScriptedModel, run_agent, evaluate, answer

CASES = json.loads(Path('datasets/cases.json').read_text())

@pytest.fixture
def store():
    s = Store()
    yield s
    s.close()

@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_dataset(case):
    s = Store(**case['store'])
    try:
        before = s.full_snapshot()
        run = run_agent(ScriptedModel(case['script']), case['prompt'], s, request=case['request'])
        assert evaluate(case, run, before, s.full_snapshot())['passed']
    finally:
        s.close()


def test_false_success_detected(store):
    case = CASES[0]
    before = store.full_snapshot()
    run = run_agent(ScriptedModel([{'final': answer('refunded', 123, 3000)}]), case['prompt'], store)
    checks = evaluate(case, run, before, store.full_snapshot())
    assert not checks['grounded'] and not checks['db_integrity'] and not checks['passed']


def test_invented_order_detected(store):
    before = store.full_snapshot()
    run = run_agent(ScriptedModel([{'final': answer('found', 999, 5000)}]), '注文999は？', store)
    assert not evaluate(CASES[1], run, before, store.full_snapshot())['grounded']


def test_wrong_amount_detected(store):
    before = store.full_snapshot()
    run = run_agent(ScriptedModel([{'tool':'lookup_order','args':{'order_id':123}}, {'final':answer('found',123,9999)}]), '金額は？', store)
    assert not evaluate(CASES[7], run, before, store.full_snapshot())['grounded']


def test_failure_claims_success():
    s = Store(fail_refund=True)
    try:
        before = s.full_snapshot()
        steps = CASES[3]['script'][:-1] + [{'final':answer('refunded',123,3000)}]
        run = run_agent(ScriptedModel(steps), CASES[3]['prompt'], s, request=CASES[3]['request'])
        assert not evaluate(CASES[3], run, before, s.full_snapshot())['grounded']
    finally:
        s.close()


def test_duplicate_refund(store):
    assert store.call('refund_order', {'order_id':123})['status'] == 'refunded'
    assert store.call('refund_order', {'order_id':123})['status'] == 'already_refunded'
    assert store.snapshot() == {123:3000,789:1500}


@pytest.mark.parametrize('name,args,status', [('delete_all',{},'unknown_tool'), ('refund_order',{'order_id':'123'},'invalid_arguments'), ('refund_order',{'order_id':True},'invalid_arguments')])
def test_bad_tool(store,name,args,status):
    before = store.snapshot()
    assert store.call(name,args)['status'] == status
    assert store.snapshot() == before


def test_bounded_loop(store):
    before = store.full_snapshot()
    model = ScriptedModel([{'tool':'lookup_order','args':{'order_id':123}}]*10)
    run = run_agent(model,'loop',store,max_steps=3)
    assert run.stopped and len(run.trace) == 3
    assert not evaluate(CASES[0],run,before,store.full_snapshot())['bounded']


def test_unrequested_refund_detected(store):
    before = store.full_snapshot()
    case = CASES[7]
    run = run_agent(ScriptedModel([{'tool':'lookup_order','args':{'order_id':123}}, {'tool':'refund_order','args':{'order_id':123}}, {'final':answer('found',123,3000)}]),case['prompt'],store)
    assert store.full_snapshot() == before
    # Simulate a write behind the guard to retain independent detection coverage.
    assert store.call('refund_order', {'order_id':123})['status'] == 'refunded'
    checks = evaluate(case,run,before,store.full_snapshot())
    assert checks['grounded'] and not checks['tool_policy'] and not checks['db_integrity']
