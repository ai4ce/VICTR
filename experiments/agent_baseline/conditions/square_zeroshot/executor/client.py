#!/usr/bin/env python3
"""Frozen agent helper: read snapshots, form current-scene chunks, report and stop."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import re
import time
from pathlib import Path

from files import atomic_json,event,usage
from safety import ARMS,interpolate


def snapshot(out,sid=None):
    if sid is not None and not re.fullmatch(r'snap_\d{6,}',sid):raise ValueError('invalid snapshot id')
    path=out/'latest.json' if sid is None else out/'snapshots'/f'{sid}.json'
    data=json.loads(path.read_text())
    if sid is not None and data['snapshot_id']!=sid:raise ValueError('snapshot mismatch')
    return data


def enqueue(out,data):
    data['submitted_at']=time.time()
    name=f'{time.time_ns()}.json'
    atomic_json(out/'inbox'/name,data)
    return name


def build_chunk(snap,phase,reason,seconds,delta=None,goal=None,quat=None,gripper=None,active_arm=None):
    locked=snap['executor']['active_arm']
    arm=active_arm if active_arm is not None else locked
    if arm not in ARMS:raise ValueError('choose --active-arm left or right for the first chunk')
    if locked is not None and arm!=locked:raise ValueError('active arm differs from attempt')
    idle='right' if arm=='left' else 'left'
    start=list(snap['state'][arm+'_ee']);target=start.copy()
    if delta is not None:target[4:]=[v+d for v,d in zip(start[4:],delta)]
    if goal is not None:target[4:]=goal
    if quat is not None:target[:4]=quat
    commands=snap['executor'].get('gripper_commands') or {a:snap['state'][a+'_gripper'] for a in ARMS}
    if gripper is None:gripper=commands[arm]
    n=round(seconds*30)
    steps=[]
    for i in range(n):
        steps.append({arm+'_ee':interpolate(start,target,(i+1)/n),
                      idle+'_ee':snap['executor']['start_anchors'][idle],
                      arm+'_gripper':float(gripper),idle+'_gripper':float(commands[idle])})
    return dict(type='chunk',chunk_id=time.time_ns(),snapshot_id=snap['snapshot_id'],
                phase=phase,reason=reason,active_arm=arm,dt=1/30,steps=steps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('snapshot');s.add_argument('--camera',choices=('zed','fish0','fish1'),nargs='+',default=['zed'])
    sub.add_parser('status')
    s=sub.add_parser('submit');s.add_argument('--snapshot',required=True)
    s.add_argument('--phase',required=True);s.add_argument('--reason',required=True)
    s.add_argument('--active-arm',choices=ARMS,help='required for first chunk; then locked for this attempt')
    s.add_argument('--seconds',type=float,required=True)
    g=s.add_mutually_exclusive_group();g.add_argument('--delta',type=float,nargs=3);g.add_argument('--goal',type=float,nargs=3)
    s.add_argument('--quat',type=float,nargs=4);s.add_argument('--gripper',type=float)
    s=sub.add_parser('judge');s.add_argument('--kind',choices=('grasp','placement','phase','note'),required=True)
    s.add_argument('--confirmed',choices=('yes','no'),required=True);s.add_argument('--reason',required=True)
    s.add_argument('--snapshot',required=True)
    s=sub.add_parser('finish');s.add_argument('--outcome',choices=('success','partial','failure'),required=True)
    s.add_argument('--reason',required=True)
    s.add_argument('--tokens',type=int);s.add_argument('--credits',type=float)
    s.add_argument('--usage-source',default='session totals unavailable; external accounting required')
    s=sub.add_parser('usage');s.add_argument('--tokens',type=int);s.add_argument('--credits',type=float)
    s.add_argument('--usage-source',required=True)
    a=p.parse_args();out=a.out.resolve()
    if a.cmd=='snapshot':
        snap=snapshot(out)
        paths={cam:str(out/snap['frames'][cam]) for cam in a.camera}
        event(out,'agent_view',snapshot_id=snap['snapshot_id'],frames=paths)
        rendered=json.dumps(dict(snapshot_id=snap['snapshot_id'],age_s=round(time.monotonic()-snap['now'],3),
              frames=paths,executor=snap['executor'],state={k:v for k,v in snap['state'].items()
              if k.endswith(('_ee','_gripper','_ik_solved')) or k=='is_recording'}),indent=2)
        usage(out,add_image_views=len(paths),add_text_chars=len(rendered))
        print(rendered)
    elif a.cmd=='status':
        meta=json.loads((out/'meta.json').read_text())
        latest=snapshot(out) if (out/'latest.json').exists() else None
        rendered=json.dumps(dict(meta=meta,executor=latest['executor'] if latest else None,
            latest_snapshot=latest['snapshot_id'] if latest else None,
            snapshot_age_s=time.monotonic()-latest['now'] if latest else None,
            usage=json.loads((out/'agent_usage.json').read_text())),indent=2)
        usage(out,add_text_chars=len(rendered));print(rendered)
    elif a.cmd=='submit':
        if not 1<=a.seconds<=3:raise ValueError('chunks must last 1-3 seconds')
        if json.loads((out/'meta.json').read_text())['end_time'] is not None:raise ValueError('executor has stopped')
        chunk=build_chunk(snapshot(out,a.snapshot),a.phase,a.reason,a.seconds,a.delta,a.goal,a.quat,a.gripper,a.active_arm)
        enqueue(out,chunk)
        usage(out,add_decisions=1,add_text_chars=len(a.reason)+len(a.phase)+200)
        print(json.dumps(dict(chunk_id=chunk['chunk_id'],ack=str(out/'acks'/f'{chunk["chunk_id"]}.json'),steps=len(chunk['steps']))))
    elif a.cmd=='judge':
        enqueue(out,dict(type='judgement',kind=a.kind,confirmed=a.confirmed=='yes',reason=a.reason,snapshot_id=a.snapshot))
        usage(out,add_text_chars=len(a.reason)+100)
    elif a.cmd in ('finish','usage'):
        if a.tokens is not None and a.tokens<0:raise ValueError('negative token usage')
        if a.credits is not None and a.credits<0:raise ValueError('negative credits')
        totals={k:v for k,v in dict(tokens=a.tokens,credits=a.credits).items() if v is not None}
        if totals:totals['usage_source']=a.usage_source
        usage(out,**totals)
        if a.cmd=='finish':
            result=dict(type='finish',outcome='failure' if a.reason=='usage_limit' else a.outcome,reason=a.reason)
            atomic_json(out/'agent_final.json',result)
            meta=json.loads((out/'meta.json').read_text())
            if meta['end_time'] is None:enqueue(out,result)
            else:
                meta['outcome_guess']=result['outcome'];meta['agent_reason']=a.reason
                atomic_json(out/'meta.json',meta)
                event(out,'agent_outcome_after_stop',**result)
            print('Finish recorded. Verify meta.json end_time via status; do not leave the executor running.')


if __name__=='__main__':main()
