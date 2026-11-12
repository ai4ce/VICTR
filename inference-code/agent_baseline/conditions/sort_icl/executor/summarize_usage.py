#!/usr/bin/env python3
"""Read-only per-attempt and per-task usage estimates; no robot connection."""
import argparse
import json
from pathlib import Path


def summarize(root):
    tasks={}
    for path in sorted(Path(root).rglob('agent_usage.json')):
        meta_path=path.with_name('meta.json')
        if not meta_path.exists():continue
        meta=json.loads(meta_path.read_text());u=json.loads(path.read_text())
        task=meta['task']
        rows=tasks.setdefault(task,[])
        rows.append(dict(attempt=str(path.parent),outcome=meta.get('outcome_guess'),
            finished=meta.get('end_time') is not None,
            estimated_tokens=u.get('estimated_tokens'),actual_tokens=u.get('tokens'),
            actual_credits=u.get('credits'),wall_time_s=u.get('wall_time_s'),
            image_views=u.get('image_views'),submitted_decisions=u.get('submitted_decisions')))
    report={}
    for task,rows in tasks.items():
        estimates=[r['estimated_tokens'] for r in rows if r['estimated_tokens'] is not None]
        report[task]=dict(attempts=rows,attempt_count=len(rows),estimated_attempt_count=len(estimates),
            estimated_token_total=sum(estimates),
            estimated_tokens_per_attempt=sum(estimates)/len(estimates) if estimates else None,
            actual_token_total=sum(r['actual_tokens'] for r in rows) if all(r['actual_tokens'] is not None for r in rows) else None,
            actual_credit_total=sum(r['actual_credits'] for r in rows) if all(r['actual_credits'] is not None for r in rows) else None)
    return dict(estimate_notice='Heuristic workload estimate, not billed usage. Hidden reasoning and repeated/cached history are unknown. Use separate practice and eval roots.',tasks=report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(summarize(p.parse_args().root),indent=2))
