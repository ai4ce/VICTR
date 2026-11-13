"""Offline tests only: robot connections and publications are mocked."""
import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from client import build_chunk,enqueue
from core import Scheduler,validate_chunk
from files import atomic_json,usage
from safety import Abort,Safety
from streams import Freshness
import run_executor
from summarize_usage import summarize


def state():
    return dict(timestamp=-1000000,left_ee=[1,0,0,0,.22,.02,.30],
        right_ee=[1,0,0,0,.22,-.59,.30],left_gripper=1.,right_gripper=1.,
        is_recording=True,left_ik_solved=True,right_ik_solved=True)


def snap(s=None,now=1,state_age=0,cam_age=0):
    s=s or state()
    return dict(state=s,state_age=state_age,camera_ages={k:cam_age for k in ('zed','fish0','fish1')},
        now=now,state_id=1,frame_id=1,snapshot_id='snap_000001',
        executor=dict(active_arm='left',idle_anchor=s['right_ee'],gripper_commands=dict(left=1,right=1)))


def chunk(cid=1,seconds=1,delta=(.01,0,0),gripper=1):
    c=build_chunk(snap(),'approach','tab ahead in current view',[dict(episode=243,start_frame=95,how='tab grasp')],seconds,delta=delta,gripper=gripper)
    c['chunk_id']=cid
    return c


class Log:
    def __init__(self):self.rows=[];self.events=[];self.acks={}
    def write(self,name,value):self.rows.append((name,value))
    def event(self,event,**fields):self.events.append((event,fields))
    def ack(self,cid,data):self.acks[cid]=data


class CoreTests(unittest.TestCase):
    def test_validate_full_chunk_and_reject_bad_inputs(self):
        validate_chunk(chunk(),snap(),'left',{'243':[95]})
        for change in ('nan','workspace','idle','demo','step','duration'):
            c=chunk()
            if change=='nan':c['steps'][0]['left_ee'][0]=float('nan')
            if change=='workspace':c['steps'][-1]['left_ee'][6]=.1
            if change=='idle':c['steps'][-1]['right_ee'][5]+=.01
            if change=='demo':c['demo_chunks'][0]['episode']=42
            if change=='step':c['steps'][0]['left_ee'][4]+=.03
            if change=='duration':c['steps']=c['steps'][:3]
            with self.subTest(change=change),self.assertRaises((ValueError,Abort)):
                validate_chunk(c,snap(),'left',{'243':[95]})

    def test_preemption_full_logging_and_gripper_hold(self):
        log=Log();safety=Safety(0);scheduler=Scheduler(safety,log)
        scheduler.accept(chunk(1,gripper=0))
        cmd,cid,index,mode=scheduler.tick(snap());scheduler.sent(cmd,cid,index)
        scheduler.accept(chunk(2,delta=(0,0,0),gripper=0))
        self.assertEqual(log.acks[1]['status'],'preempted')
        for i in range(30):
            cmd,cid,index,mode=scheduler.tick(snap());self.assertEqual(cid,2);scheduler.sent(cmd,cid,index)
        self.assertEqual(log.acks[2]['status'],'ran_fully')
        cmd,cid,index,mode=scheduler.tick(snap());self.assertIsNone(cid)
        self.assertEqual(mode,'queue_empty_hold');self.assertEqual(cmd['left_gripper'],0)

    def test_hold_does_not_follow_sag(self):
        scheduler=Scheduler(Safety(0),Log());s=state()
        cmd,cid,i,_=scheduler.tick(snap(s));scheduler.sent(cmd,cid,i)
        for sag in (.002,.004,.010):
            drift=copy.deepcopy(s);drift['left_ee'][6]-=sag
            cmd,cid,i,_=scheduler.tick(snap(drift));scheduler.sent(cmd,cid,i)
            self.assertEqual(cmd['left_ee'],s['left_ee'])

    def test_persistent_chunk_reaches_command_goal_with_sag(self):
        c=chunk(seconds=3,delta=(.04,0,0))
        log=Log();scheduler=Scheduler(Safety(0),log);scheduler.accept(c);s=state()
        for i in range(90):
            cmd,cid,step,_=scheduler.tick(snap(s,now=1+i/30));scheduler.sent(cmd,cid,step)
            s=copy.deepcopy(s);s['left_ee']=cmd['left_ee'].copy();s['left_ee'][6]-=.012
        self.assertAlmostEqual(cmd['left_ee'][4],.26)
        self.assertEqual(log.acks[1]['status'],'ran_fully')

    def test_local_freshness_repeats_and_stop_limits(self):
        f=Freshness()
        for source in (-1e9,-1e9,1e9,None):
            f.update(source,10);self.assertAlmostEqual(f.age(10.1),.1)
        safety=Safety(0)
        self.assertTrue(safety.check(snap(cam_age=.99)))
        self.assertFalse(safety.check(snap(cam_age=1.01)))
        self.assertFalse(safety.check(snap(state_age=.51)))
        for args in (dict(cam_age=2.01),dict(state_age=2.01)):
            with self.assertRaisesRegex(Abort,'receipt age'):safety.check(snap(**args))

    def test_stale_interrupt_does_not_resume_old_chunk(self):
        log=Log();scheduler=Scheduler(Safety(0),log);scheduler.accept(chunk())
        cmd,cid,i,_=scheduler.tick(snap());scheduler.sent(cmd,cid,i)
        self.assertIsNone(scheduler.tick(snap(state_age=.7))[0])
        self.assertEqual(log.acks[1]['status'],'preempted')
        self.assertEqual(scheduler.tick(snap())[3],'queue_empty_hold')

    def test_immediate_stops(self):
        for key,value in [('is_recording',False),('left_ik_solved',False),('right_ik_solved',False),('left_gripper',float('nan'))]:
            s=state();s[key]=value
            with self.assertRaises(Abort):Safety(0).check(snap(s))
        safety=Safety(0);safety.check(snap());s=state();s['right_ee'][6]-=.024
        self.assertTrue(safety.check(snap(s)));s['right_ee'][6]-=.002
        with self.assertRaisesRegex(Abort,'idle'):safety.check(snap(s))
        safety=Safety(0);safety.previous=state();s=state();s['left_ee'][4]+=.036
        with self.assertRaisesRegex(Abort,'tracking'):safety.check(snap(s))
        with self.assertRaisesRegex(Abort,'timeout'):Safety(0).check(snap(now=600))

    def test_usage_estimate_is_labelled_and_actual_separate(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);atomic_json(out/'agent_usage.json',dict(started_at=time.time(),initial_context_chars=4000,tokens=None))
            usage(out,add_image_views=2,add_text_chars=400,add_decisions=1)
            u=json.loads((out/'agent_usage.json').read_text())
            self.assertEqual(u['estimated_tokens'],1100+512+512);self.assertIsNone(u['tokens'])
            usage(out,tokens=12345,usage_source='actual session counter')
            self.assertEqual(json.loads((out/'agent_usage.json').read_text())['tokens'],12345)
            usage(out,ended_at=time.time())
            finished=json.loads((out/'agent_usage.json').read_text())['wall_time_s']
            usage(out,add_text_chars=100)
            self.assertEqual(json.loads((out/'agent_usage.json').read_text())['wall_time_s'],finished)

    def test_per_task_usage_preserves_unknown_actuals(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for i in range(2):
                out=root/str(i);out.mkdir()
                atomic_json(out/'meta.json',dict(task='square',end_time=10,outcome_guess='failure'))
                atomic_json(out/'agent_usage.json',dict(estimated_tokens=1000*(i+1),tokens=123 if i else None))
            task=summarize(root)['tasks']['square']
            self.assertEqual(task['estimated_token_total'],3000)
            self.assertEqual(task['estimated_tokens_per_attempt'],1500)
            self.assertIsNone(task['actual_token_total'])


class Integration(unittest.TestCase):
    def test_files_inbox_preemption_snapshots_and_exact_wire(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'notes').mkdir();(root/'attempt_prompt.md').write_text('test prompt')
            (root/'notes/demo_analysis.md').write_text('test context')
            (root/'notes/demo_index.json').write_text('{"243":[95]}')
            out=root/'attempt';messages=[];current=state();errors=[]
            class Pub:
                def setsockopt(self,*a):pass
                def bind(self,*a):pass
                def close(self,**kw):pass
                def send_multipart(self,wire):
                    messages.append(wire);cmd=json.loads(wire[1])
                    for arm in ('left','right'):current[arm+'_ee']=cmd[arm+'_ee']
            class Context:
                def socket(self,*a):return Pub()
            class Inputs:
                def __init__(self,log,*a):self.lock=threading.RLock();self.log=log;self.i=0
                def snapshot(self):
                    self.i+=1;ss=copy.deepcopy(current);now=time.monotonic()
                    self.log.write('state',dict(state_id=self.i,packet=ss,received_monotonic=now))
                    s=snap(ss,now=now);s.update(state_id=self.i,frame_id=self.i,camera_timestamp=-1e9,
                        state_received_monotonic=now,camera_received_monotonic=now,
                        images={k:np.zeros((180,320,3),'uint8') for k in ('zed','fish0','fish1')})
                    return s
                def close(self):pass
            def wait_for(predicate):
                deadline=time.monotonic()+6
                while time.monotonic()<deadline:
                    if predicate():return
                    time.sleep(.01)
                raise AssertionError('offline integration timed out')
            def agent():
                try:
                    wait_for(lambda:(out/'latest.json').exists())
                    obs=json.loads((out/'latest.json').read_text())
                    first=build_chunk(obs,'approach','test',[dict(episode=243,start_frame=95,how='grasp')],3,delta=(.01,0,0));first['chunk_id']=1
                    enqueue(out,first)
                    wait_for(lambda:(out/'acks/1.json').exists())
                    time.sleep(.2)
                    obs=json.loads((out/'latest.json').read_text())
                    second=build_chunk(obs,'grasp','test',[dict(episode=243,start_frame=95,how='grasp')],1,delta=(.002,0,0));second['chunk_id']=2
                    enqueue(out,second)
                    wait_for(lambda:(out/'acks/2.json').exists() and json.loads((out/'acks/2.json').read_text())['status']=='ran_fully')
                    enqueue(out,dict(type='finish',outcome='failure',reason='offline test complete'))
                except Exception as e:
                    errors.append(str(e))
                    if (out/'inbox').exists():enqueue(out,dict(type='finish',outcome='failure',reason='offline test error'))
            t=threading.Thread(target=agent,daemon=True);t.start()
            argv=['run_executor.py','--out',str(out)]
            with patch.object(run_executor,'ROOT',root),patch.object(run_executor,'git',side_effect=lambda *a:'testcommit' if a[0]=='rev-parse' else ''),patch.object(run_executor,'Streams',Inputs),patch.object(run_executor.zmq.Context,'instance',return_value=Context()),patch('sys.argv',argv):
                self.assertEqual(run_executor.main(),0)
            t.join(1);self.assertEqual(errors,[])
            self.assertEqual(json.loads((out/'acks/1.json').read_text())['status'],'preempted')
            self.assertEqual(json.loads((out/'acks/2.json').read_text())['status'],'ran_fully')
            rows=[json.loads(x) for x in (out/'commands.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),len(messages))
            for row,wire in zip(rows,messages):self.assertEqual(row['wire_json'].encode(),wire[1])
            submissions=[json.loads(x) for x in (out/'chunks.jsonl').read_text().splitlines()]
            self.assertEqual([len(x['submission']['steps']) for x in submissions if x['record_type']=='submitted'],[90,30])
            self.assertTrue((out/'latest.json').exists());self.assertTrue(list((out/'frames').glob('*.jpg')))
            self.assertIsNotNone(json.loads((out/'meta.json').read_text())['end_time'])


if __name__=='__main__':unittest.main()
