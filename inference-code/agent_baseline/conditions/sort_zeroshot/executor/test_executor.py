"""Offline tests only: robot connections and publications are mocked."""
import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import client
from client import build_chunk as _build_chunk,enqueue
from core import Scheduler,validate_chunk as _validate_chunk
from files import atomic_json,usage
from safety import Abort,Safety
from streams import Freshness
import run_executor
from summarize_usage import summarize


def build_chunk(*args,**kwargs):
    kwargs.setdefault('active_arm','left')
    return _build_chunk(*args,**kwargs)


def validate_chunk(chunk,snapshot,active_arm):
    return _validate_chunk(chunk,snapshot,active_arm)


def state():
    return dict(timestamp=-1000000,left_ee=[1,0,0,0,.22,.02,.30],
        right_ee=[1,0,0,0,.22,-.59,.30],left_gripper=1.,right_gripper=1.,
        is_recording=True,left_ik_solved=True,right_ik_solved=True)


def snap(s=None,now=1,state_age=0,cam_age=0):
    s=s or state()
    return dict(state=s,state_age=state_age,camera_ages={k:cam_age for k in ('zed','fish0','fish1')},
        now=now,state_id=1,frame_id=1,snapshot_id='snap_000001',
        executor=dict(active_arm='left',idle_anchor=s['right_ee'],
            start_anchors={a:s[a+'_ee'] for a in ('left','right')},gripper_commands=dict(left=1,right=1)))


def chunk(cid=1,seconds=1,delta=(.01,0,0),gripper=1):
    c=build_chunk(snap(),'approach','synthetic object visible',seconds,delta=delta,gripper=gripper)
    c['chunk_id']=cid
    return c


class Log:
    def __init__(self):self.rows=[];self.events=[];self.acks={}
    def write(self,name,value):self.rows.append((name,value))
    def event(self,event,**fields):self.events.append((event,fields))
    def ack(self,cid,data):self.acks[cid]=data


class CoreTests(unittest.TestCase):
    def test_submission_without_demo_is_accepted_and_counts_only_local_text(self):
        from streams import Log as DiskLog
        with tempfile.TemporaryDirectory() as td:
            log=DiskLog(Path(td)/'out');out=log.out
            scheduler=Scheduler(Safety(time.monotonic()),log)
            observed=snap(now=time.monotonic());scheduler.tick(observed)
            observed['executor']=scheduler.summary()
            atomic_json(out/'meta.json',dict(end_time=None))
            atomic_json(out/'agent_usage.json',dict(started_at=time.time(),tokens=None))
            atomic_json(out/'snapshots/snap_000001.json',observed)
            class Inputs:
                def snapshot(self):return observed
            meta=dict(active_arm=None,end_time=None)
            argv=['client.py','--out',str(out),'submit','--snapshot','snap_000001',
                  '--active-arm','left','--phase','lift','--reason','observed grasp','--seconds','1']
            for _ in range(2):
                with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                    client.main()
                path=sorted((out/'inbox').glob('*.json'))[-1]
                submitted=json.loads(path.read_text())
                self.assertNotIn('demo_chunks',submitted)
                run_executor.process_request(submitted,None,path.name,log,Inputs(),scheduler,meta)
                self.assertEqual(scheduler.current['chunk_id'],submitted['chunk_id'])
                ack=json.loads((out/'acks'/f'{submitted["chunk_id"]}.json').read_text())
                self.assertEqual(ack['status'],'running')
            result=json.loads((out/'agent_usage.json').read_text())
            self.assertEqual(result['submitted_decisions'],2)
            self.assertEqual(result['observed_text_chars'],2*(len('observed grasp')+len('lift')+200))
            self.assertEqual(len(list((out/'inbox').glob('*.json'))),2)

    def test_deadline_3600(self):
        safety=Safety(0)
        self.assertEqual(safety.timeout,3600)
        for now in (600,3599.999):self.assertTrue(safety.check(snap(now=now)))
        for now in (3600,3601):
            with self.assertRaisesRegex(Abort,'hard timeout'):safety.check(snap(now=now))

    def test_switch_reanchors_at_acceptance_and_keeps_gripper(self):
        for first in ('left','right'):
            other='right' if first=='left' else 'left'
            scheduler=Scheduler(Safety(0),Log());scheduler.tick(snap())
            c=build_chunk(snap(),'grasp','synthetic',1,active_arm=first,gripper=0)
            scheduler.accept(c)
            cmd,cid,i,_=scheduler.tick(snap());scheduler.sent(cmd,cid,i)
            observed=snap();observed['executor']=scheduler.summary()
            switch=build_chunk(observed,'transfer','synthetic',1,active_arm=other,gripper=.4)
            measured=state();measured[first+'_ee'][4]+=.030
            # Simulate the first arm having moved after the agent's observation.
            scheduler.safety.previous[first+'_ee']=measured[first+'_ee'].copy()
            scheduler.safety.check(snap(measured))
            submitted=copy.deepcopy(switch)
            scheduler.accept(switch,measured)
            self.assertEqual(switch,submitted)
            anchor=measured[first+'_ee'].copy()
            self.assertEqual(scheduler.summary()['idle_anchor'],anchor)
            self.assertEqual(scheduler.summary()['start_anchors'][first],state()[first+'_ee'])
            cmd,cid,i,_=scheduler.tick(snap(measured));scheduler.sent(cmd,cid,i)
            self.assertEqual(cmd[first+'_ee'],anchor)
            self.assertEqual(cmd[first+'_gripper'],0)
            self.assertEqual(cmd[other+'_gripper'],.4)
            scheduler.discard('preempted','synthetic hold')
            held=scheduler.tick(snap(measured))[0]
            self.assertEqual(held[first+'_gripper'],0)
            self.assertEqual(held[other+'_gripper'],.4)
            drift=copy.deepcopy(measured);drift[first+'_ee'][4]+=.039
            self.assertTrue(scheduler.safety.check(snap(drift)))
            drift[first+'_ee'][4]+=.002
            with self.assertRaisesRegex(Abort,'idle arm drift'):scheduler.safety.check(snap(drift))
            drift=copy.deepcopy(measured);theta=math.radians(15.1)/2
            drift[first+'_ee'][:4]=[math.cos(theta),math.sin(theta),0,0]
            with self.assertRaisesRegex(Abort,'idle arm drift'):scheduler.safety.check(snap(drift))

    def test_switch_request_uses_current_state_and_rejects_idle_gripper_change(self):
        from streams import Log as DiskLog
        with tempfile.TemporaryDirectory() as td:
            log=DiskLog(Path(td)/'out');scheduler=Scheduler(Safety(time.monotonic()),log)
            observed=snap(now=time.monotonic());scheduler.tick(observed)
            scheduler.accept(chunk(gripper=0))
            cmd,cid,i,_=scheduler.tick(observed);scheduler.sent(cmd,cid,i)
            observed['executor']=scheduler.summary()
            atomic_json(log.out/'snapshots'/f'{observed["snapshot_id"]}.json',observed)
            current=copy.deepcopy(observed);current['state']['left_ee'][4]+=.020
            class Inputs:
                def snapshot(self):return current
            meta=dict(active_arm='left')
            c=build_chunk(observed,'transfer','switch after motion',1,active_arm='right')
            bad=copy.deepcopy(c);bad['steps'][0]['left_gripper']=1
            run_executor.process_request(bad,None,'bad.json',log,Inputs(),scheduler,meta)
            self.assertEqual(scheduler.safety.active,'left')
            run_executor.process_request(c,None,'good.json',log,Inputs(),scheduler,meta)
            self.assertEqual(scheduler.safety.active,'right')
            self.assertEqual(scheduler.safety.idle_anchor,current['state']['left_ee'])
            self.assertTrue(all(s['left_gripper']==0 for s in scheduler.current['steps']))

    def test_required_metadata_and_arm_selection(self):
        for field in ('phase','reason','snapshot_id','active_arm'):
            c=chunk();del c[field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_chunk(c,snap(),None)
        c=chunk();c['active_arm']='neither'
        with self.assertRaises(ValueError):validate_chunk(c,snap(),None)
        self.assertNotIn('demo_chunks',chunk())

    def test_unselected_summary_and_start_pose_hold(self):
        scheduler=Scheduler(Safety(0),Log())
        self.assertIsNone(scheduler.summary()['active_arm'])
        self.assertIsNone(scheduler.summary()['idle_anchor'])
        original=state();scheduler.safety.check(snap(original))
        drift=copy.deepcopy(original)
        for arm in ('left','right'):drift[arm+'_ee'][6]-=.004
        cmd,cid,i,mode=scheduler.tick(snap(drift));scheduler.sent(cmd,cid,i)
        self.assertEqual(mode,'queue_empty_hold');self.assertIsNone(cid)
        for arm in ('left','right'):self.assertEqual(cmd[arm+'_ee'],original[arm+'_ee'])
        self.assertIsNone(scheduler.tick(snap(drift,state_age=.6))[0])
        cmd,_,_,_=scheduler.tick(snap(drift))
        for arm in ('left','right'):self.assertEqual(cmd[arm+'_ee'],original[arm+'_ee'])

    def test_arm_switch_both_choices_and_duplicate_does_not_preempt(self):
        for arm in ('left','right'):
            with self.subTest(arm=arm):
                log=Log();scheduler=Scheduler(Safety(0),log);scheduler.tick(snap())
                obs=snap();obs['executor']=scheduler.summary()
                with self.assertRaisesRegex(ValueError,'choose --active-arm'):
                    _build_chunk(obs,'approach','synthetic observation',1)
                c=build_chunk(obs,'approach','synthetic observation',1,active_arm=arm);c['chunk_id']=1
                validate_chunk(c,obs,None);self.assertTrue(scheduler.accept(c))
                self.assertEqual(scheduler.safety.active,arm)
                self.assertEqual(scheduler.summary()['idle_anchor'],state()[scheduler.safety.idle+'_ee'])
                accepted=scheduler.current
                self.assertFalse(scheduler.accept(c));self.assertIs(scheduler.current,accepted)
                other=scheduler.safety.idle
                obs['executor']=scheduler.summary()
                switch=build_chunk(obs,'transfer','synthetic switch',1,active_arm=other);switch['chunk_id']=2
                validate_chunk(switch,obs,arm)
                self.assertTrue(scheduler.accept(switch,state()))
                self.assertEqual(scheduler.safety.active,other)
                self.assertEqual(log.acks[1]['status'],'preempted')
                self.assertEqual(sum(e=='active_arm_selected' for e,_ in log.events),1)
                self.assertEqual(sum(e=='active_arm_switched' for e,_ in log.events),1)
                self.assertFalse(scheduler.accept(c,state()))
                self.assertEqual(scheduler.safety.active,other)

    def test_idle_drift_uses_original_anchor_for_either_choice(self):
        for arm in ('left','right'):
            for rotation in (False,True):
                with self.subTest(arm=arm,rotation=rotation):
                    scheduler=Scheduler(Safety(0),Log());scheduler.tick(snap())
                    obs=snap();obs['executor']=scheduler.summary()
                    drift=state();idle='right' if arm=='left' else 'left'
                    drift[idle+'_ee'][6]-=.004
                    scheduler.tick(snap(drift))
                    scheduler.accept(build_chunk(obs,'approach','synthetic observation',1,active_arm=arm))
                    if rotation:
                        theta=math.radians(15.1)/2
                        drift[idle+'_ee'][:4]=[math.cos(theta),math.sin(theta),0,0]
                    else:drift[idle+'_ee'][6]-=.037
                    with self.assertRaisesRegex(Abort,'idle arm drift'):scheduler.tick(snap(drift))

    def test_rejected_first_chunk_does_not_lock(self):
        from streams import Log as DiskLog
        with tempfile.TemporaryDirectory() as td:
            log=DiskLog(Path(td)/'out');scheduler=Scheduler(Safety(time.monotonic()),log)
            obs=snap(now=time.monotonic());scheduler.tick(obs);obs['executor']=scheduler.summary()
            atomic_json(log.out/'snapshots'/f'{obs["snapshot_id"]}.json',obs)
            class Inputs:
                def snapshot(self):return obs
            meta=dict(active_arm=None)
            for change in ('reason','idle_gripper','old_snapshot','id','dt','limit'):
                c=build_chunk(obs,'approach','synthetic observation',1,active_arm='right');c['chunk_id']=1
                if change=='reason':c['reason']=''
                if change=='idle_gripper':c['steps'][0]['left_gripper']=0
                if change=='old_snapshot':
                    old=copy.deepcopy(obs);old['now']-=31
                    atomic_json(log.out/'snapshots'/f'{obs["snapshot_id"]}.json',old)
                if change=='id':c['chunk_id']=-1
                if change=='dt':c['dt']=.1
                if change=='limit':c=build_chunk(obs,'approach','synthetic observation',3,delta=(.066,0,0),active_arm='right')
                run_executor.process_request(c,None,'test.json',log,Inputs(),scheduler,meta)
                with self.subTest(change=change):
                    self.assertIsNone(scheduler.safety.active);self.assertIsNone(scheduler.current)
                    self.assertIsNone(meta['active_arm'])
                    statuses=[json.loads(line) for line in (log.out/'chunks.jsonl').read_text().splitlines()]
                    self.assertEqual(statuses[-1]['status'],'rejected_by_safety')
                atomic_json(log.out/'snapshots'/f'{obs["snapshot_id"]}.json',obs)

    def test_validate_full_chunk_and_reject_bad_inputs(self):
        validate_chunk(chunk(),snap(),'left')
        for change in ('nan','workspace','idle','arm','step','duration'):
            c=chunk()
            if change=='nan':c['steps'][0]['left_ee'][0]=float('nan')
            if change=='workspace':c['steps'][-1]['left_ee'][6]=.1
            if change=='idle':c['steps'][-1]['right_ee'][5]+=.01
            if change=='arm':c['active_arm']='invalid'
            if change=='step':c['steps'][0]['left_ee'][4]+=.03
            if change=='duration':c['steps']=c['steps'][:3]
            with self.subTest(change=change),self.assertRaises((ValueError,Abort)):
                validate_chunk(c,snap(),'left')

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
        safety=Safety(0,'left');safety.check(snap());s=state();s['right_ee'][6]-=.039
        self.assertTrue(safety.check(snap(s)));s['right_ee'][6]-=.002
        with self.assertRaisesRegex(Abort,'idle'):safety.check(snap(s))
        safety=Safety(0);safety.previous=state();s=state();s['left_ee'][4]+=.036
        with self.assertRaisesRegex(Abort,'tracking'):safety.check(snap(s))
        with self.assertRaisesRegex(Abort,'timeout'):Safety(0).check(snap(now=3600))

    def test_measured_tracking_limits_follow_arm_role(self):
        for active in ('left','right'):
            with self.subTest(active=active):
                safety=Safety(0,active);safety.check(snap());safety.previous=state()
                idle=safety.idle
                measured=state();measured[idle+'_ee'][6]-=.039
                self.assertTrue(safety.check(snap(measured)))
                measured=state();measured[active+'_ee'][6]-=.036
                with self.assertRaisesRegex(Abort,active+' tracking error >35mm'):
                    safety.check(snap(measured))
                # Isolate tracking from the separate anchor-drift check.
                measured=state();measured[idle+'_ee'][6]-=.041
                safety.idle_anchor=measured[idle+'_ee'].copy()
                with self.assertRaisesRegex(Abort,idle+' tracking error >40mm'):
                    safety.check(snap(measured))
                safety.idle_anchor=state()[idle+'_ee']
                for arm in ('left','right'):
                    for degrees in (14.9,15.1):
                        measured=state();theta=math.radians(degrees)/2
                        measured[arm+'_ee'][:4]=[math.cos(theta),math.sin(theta),0,0]
                        if degrees<15:self.assertTrue(safety.check(snap(measured)))
                        else:
                            with self.assertRaisesRegex(Abort,'15deg'):safety.check(snap(measured))

    def test_proposed_tracking_limits_follow_arm_role(self):
        for active in ('left','right'):
            with self.subTest(active=active):
                safety=Safety(0,active);safety.check(snap());safety.previous=state()
                idle=safety.idle;target=state()
                measured=state();measured[idle+'_ee'][6]-=.039
                result=safety.command(target,measured)
                self.assertEqual(result[idle+'_ee'],target[idle+'_ee'])
                measured[idle+'_ee'][6]-=.002
                with self.assertRaisesRegex(Abort,idle+' proposed tracking error >40mm'):
                    safety.command(target,measured)
                measured=state();measured[active+'_ee'][6]-=.036
                with self.assertRaisesRegex(Abort,active+' proposed tracking error >35mm'):
                    safety.command(target,measured)
                for arm in ('left','right'):
                    for degrees in (14.9,15.1):
                        measured=state();theta=math.radians(degrees)/2
                        measured[arm+'_ee'][:4]=[math.cos(theta),math.sin(theta),0,0]
                        if degrees<15:safety.command(target,measured)
                        else:
                            with self.assertRaisesRegex(Abort,'proposed angular tracking error >15deg'):
                                safety.command(target,measured)

    def test_both_arms_keep_35mm_15deg_tracking_before_selection(self):
        for arm in ('left','right'):
            safety=Safety(0);safety.check(snap());safety.previous=state()
            self.assertIsNone(safety.idle)
            for distance in (.034,.036):
                measured=state();measured[arm+'_ee'][6]-=distance
                if distance<.035:
                    self.assertTrue(safety.check(snap(measured)))
                    safety.command(state(),measured)
                else:
                    with self.assertRaisesRegex(Abort,arm+' tracking error >35mm'):
                        safety.check(snap(measured))
                    with self.assertRaisesRegex(Abort,arm+' proposed tracking error >35mm'):
                        safety.command(state(),measured)
            for degrees in (14.9,15.1):
                measured=state();theta=math.radians(degrees)/2
                measured[arm+'_ee'][:4]=[math.cos(theta),math.sin(theta),0,0]
                if degrees<15:
                    self.assertTrue(safety.check(snap(measured)))
                    safety.command(state(),measured)
                else:
                    with self.assertRaisesRegex(Abort,'15deg'):safety.check(snap(measured))
                    with self.assertRaisesRegex(Abort,'15deg'):safety.command(state(),measured)

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


class FreezeTests(unittest.TestCase):
    def test_two_commit_freeze_and_integrity_rejections(self):
        import hashlib
        # Temporary fixture commits only; never commit the sort working repository here.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),'-c','user.name=Offline Test',
                    '-c','user.email=offline@example.invalid','-c','core.hooksPath=/dev/null',*args],
                    text=True,stderr=subprocess.DEVNULL).strip()
            git('init','-q');(root/'.gitignore').write_text('/eval/\n')
            prompt=root/'attempt_prompt.md';prompt.write_text('synthetic prompt')
            git('add','.');git('commit','-qm','fixture code and prompt')
            base=git('rev-parse','HEAD');digest=hashlib.sha256(prompt.read_bytes()).hexdigest()
            (root/'FROZEN').write_text(base+'\n'+digest+'  attempt_prompt.md\n')
            git('add','FROZEN');git('commit','-qm','fixture freeze marker')
            def check(label,expected,starts=False):
                out=root/'eval'/label
                with patch.object(run_executor,'ROOT',root),patch.object(run_executor,'Streams',side_effect=Abort('offline stream sentinel')) as inputs,patch.object(run_executor.zmq.Context,'instance',side_effect=AssertionError('network forbidden')),patch('sys.argv',['run_executor.py','--out',str(out)]):
                    self.assertEqual(run_executor.main(),1)
                self.assertEqual(inputs.called,starts)
                meta=json.loads((out/'meta.json').read_text())
                self.assertIn(expected,meta['exit_reason']);self.assertIsNotNone(meta['end_time'])
                u=json.loads((out/'agent_usage.json').read_text())
                self.assertEqual(u['initial_context_chars'],len(prompt.read_text()))
            check('valid','offline stream sentinel',starts=True)
            prompt.write_text('dirty prompt');check('dirty','repo is not clean')
            prompt.write_text('synthetic prompt')
            (root/'untracked').write_text('unexpected');check('untracked','repo is not clean')
            (root/'untracked').unlink()
            (root/'FROZEN').write_text(base+'\n'+'0'*64+'  attempt_prompt.md\n')
            git('add','FROZEN');git('commit','--amend','-qm','fixture bad hash')
            check('bad_hash','FROZEN commit or prompt hash mismatch')
            (root/'FROZEN').write_text(base+'\n'+digest+'  attempt_prompt.md\n')
            (root/'extra').write_text('unexpected code change in marker commit')
            git('add','FROZEN','extra');git('commit','--amend','-qm','fixture extra change')
            check('extra_change','FROZEN commit or prompt hash mismatch')
            (root/'extra').unlink();git('add','extra');git('commit','--amend','-qm','fixture restored marker')
            git('commit','--allow-empty','-qm','fixture later commit')
            check('later_commit','FROZEN commit or prompt hash mismatch')


class Integration(unittest.TestCase):
    """Real main loop, file workers and helper; no network sockets or robot inputs."""
    def run_case(self,arm,stale=None):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'attempt_prompt.md').write_text('test prompt')
            saved=os.environ.get('SORT_OFFLINE_ROOT')
            out=(Path(saved)/f'{arm}_{stale or "finish"}') if saved else root/'attempt'
            messages=[];current=state();errors=[];ages=dict(state=0,camera=0);ids={}
            closed=[];bound=[];observations={}
            class Pub:
                def setsockopt(self,*a):pass
                def bind(self,address):bound.append(address)  # in memory only
                def close(self,**kw):closed.append(True)
                def send_multipart(self,wire):
                    messages.append(wire);cmd=json.loads(wire[1])
                    for a in ('left','right'):current[a+'_ee']=cmd[a+'_ee']
            class Context:
                def socket(self,kind):
                    if kind!=run_executor.zmq.PUB:raise AssertionError('unexpected socket')
                    return Pub()
            class Inputs:
                def __init__(self,log,*a):self.lock=threading.RLock();self.log=log;self.i=0
                def snapshot(self):
                    with self.lock:
                        self.i+=1;ss=copy.deepcopy(current);now=time.monotonic()
                        self.log.write('state',dict(state_id=self.i,packet=ss,received_monotonic=now-ages['state']))
                        s=snap(ss,now=now,state_age=ages['state'],cam_age=ages['camera'])
                        s.update(state_id=self.i,frame_id=self.i,camera_timestamp=-1e9,
                            state_received_monotonic=now-ages['state'],camera_received_monotonic=now-ages['camera'],
                            images={k:np.zeros((360,640,3),'uint8') for k in ('zed','fish0','fish1')})
                        return s
                def close(self):pass
            def wait_for(predicate):
                deadline=time.monotonic()+6
                while time.monotonic()<deadline:
                    if predicate():return
                    time.sleep(.01)
                raise AssertionError('offline integration timed out')
            def rows(name):
                # append may be in progress; ignore only an incomplete final line.
                text=(out/(name+'.jsonl')).read_text()
                return [json.loads(x) for x in text.splitlines(keepends=True) if x.endswith('\n')]
            def helper(*args):
                output=io.StringIO()
                with patch('sys.argv',['client.py','--out',str(out),*args]),contextlib.redirect_stdout(output):
                    client.main()
                return output.getvalue()
            def ack(cid,status):
                p=out/'acks'/f'{cid}.json'
                return p.exists() and json.loads(p.read_text())['status']==status
            def agent():
                try:
                    wait_for(lambda:(out/'latest.json').exists() and len(messages)>=2)
                    observations['status']=json.loads(helper('status'))
                    observations['before']=json.loads(helper('snapshot'))
                    self.assertIsNone(observations['status']['executor']['active_arm'])
                    self.assertIsNone(observations['before']['executor']['active_arm'])
                    for row in rows('commands'):
                        self.assertEqual(row['mode'],'queue_empty_hold')
                        for a in ('left','right'):
                            self.assertEqual(row['message'][a+'_ee'],state()[a+'_ee'])
                    sid=observations['before']['snapshot_id']
                    first=json.loads(helper('submit','--snapshot',sid,'--active-arm',arm,
                        '--phase','approach','--reason','synthetic first chunk','--seconds','3',
                        '--delta','.01','0','0','--gripper','0'))
                    ids['first']=first['chunk_id'];wait_for(lambda:ack(ids['first'],'running'))
                    self.assertEqual(json.loads((out/'meta.json').read_text())['active_arm'],arm)
                    # Invalid required metadata must not pre-empt the running chunk.
                    obs=json.loads((out/'snapshots'/f'{sid}.json').read_text())
                    other='right' if arm=='left' else 'left'
                    bad=build_chunk(obs,'approach','synthetic invalid metadata',1,active_arm=other)
                    bad['phase']=''
                    ids['wrong']=bad['chunk_id'];enqueue(out,bad)
                    wait_for(lambda:ack(ids['wrong'],'rejected_by_safety'))
                    self.assertTrue(ack(ids['first'],'running'))
                    wait_for(lambda:json.loads((out/'latest.json').read_text())['executor']['active_arm']==arm)
                    obs=json.loads(helper('snapshot','--camera','fish0','fish1'))
                    second=json.loads(helper('submit','--snapshot',obs['snapshot_id'],
                        '--active-arm',other,'--phase','grasp','--reason','synthetic replacement and switch','--seconds','1',
                        '--delta','.002','0','0','--gripper','0'))
                    ids['second']=second['chunk_id'];wait_for(lambda:ack(ids['second'],'running'))
                    self.assertTrue(ack(ids['first'],'preempted'))
                    if stale:
                        ages[stale]=1.1 if stale=='camera' else .6
                        wait_for(lambda:ack(ids['second'],'preempted'))
                        if stale=='camera':
                            wait_for(lambda:any(r['mode']=='stale_hold' for r in rows('commands')))
                            holds=[r for r in rows('commands') if r['mode']=='stale_hold']
                            self.assertTrue(all(r['message'][arm+'_gripper']==0 for r in holds))
                        else:
                            # State-stale hold stops sending targets until measurements return.
                            count=len(messages);time.sleep(.12);self.assertEqual(len(messages),count)
                        ages[stale]=0
                        wait_for(lambda:rows('commands')[-1]['mode']=='queue_empty_hold')
                        self.assertTrue(ack(ids['second'],'preempted'))
                        ages[stale]=2.01
                        wait_for(lambda:json.loads((out/'meta.json').read_text())['end_time'] is not None)
                    else:
                        wait_for(lambda:ack(ids['second'],'ran_fully'))
                        wait_for(lambda:rows('commands')[-1]['mode']=='queue_empty_hold')
                        self.assertEqual(rows('commands')[-1]['message'][arm+'_gripper'],0)
                    helper('judge','--kind','note','--confirmed','no','--snapshot',obs['snapshot_id'],
                           '--reason','synthetic offline check only')
                    helper('finish','--outcome','failure','--reason','offline_mock_complete')
                except Exception as e:
                    errors.append(repr(e))
                    if (out/'inbox').exists():enqueue(out,dict(type='finish',outcome='failure',reason='offline test error'))
            argv=['run_executor.py','--out',str(out)]
            with patch.object(run_executor,'ROOT',root),patch.object(run_executor,'git',side_effect=lambda *a:'testcommit' if a[0]=='rev-parse' else ''),patch.object(run_executor,'Streams',Inputs),patch.object(run_executor,'Safety',wraps=Safety) as safety_factory,patch.object(run_executor.zmq.Context,'instance',return_value=Context()),patch('sys.argv',argv):
                t=threading.Thread(target=agent,daemon=True);t.start()
                self.assertEqual(run_executor.main(),1 if stale else 0)
                t.join(7)
            self.assertEqual(safety_factory.call_args.kwargs['timeout'],3600)
            self.assertFalse(t.is_alive());self.assertEqual(errors,[])
            self.assertEqual(bound,['tcp://0.0.0.0:5572']);self.assertEqual(closed,[True])
            self.assertTrue(ack(ids['wrong'],'rejected_by_safety'))
            self.assertTrue(ack(ids['first'],'preempted'))
            self.assertTrue(ack(ids['second'],'preempted' if stale else 'ran_fully'))
            commands=rows('commands');self.assertEqual(len(commands),len(messages))
            for row,wire in zip(commands,messages):
                self.assertEqual(row['wire_json'].encode(),wire[1])
                self.assertEqual(wire[0],b'inference/bimanual_ee_action')
            submitted=[x['submission'] for x in rows('chunks') if x['record_type']=='submitted']
            self.assertEqual([len(x['steps']) for x in submitted],[90,30,30])
            self.assertTrue(all('demo_chunks' not in x for x in submitted))
            self.assertTrue(all('demo_chunks' not in json.dumps(x) for x in rows('chunks')))
            self.assertTrue(all(x['reason'] and x['snapshot_id'] for x in submitted))
            events=rows('events');choices=[e for e in events if e.get('event')=='active_arm_selected']
            self.assertEqual(len(choices),1);self.assertEqual(choices[0]['active_arm'],arm)
            switches=[e for e in events if e.get('event')=='active_arm_switched']
            self.assertEqual(len(switches),1);self.assertEqual(switches[0]['previous_arm'],arm)
            self.assertEqual(switches[0]['idle_gripper'],0)
            self.assertTrue(rows('state'));self.assertTrue(list((out/'snapshots').glob('*.json')))
            for cam in ('zed','fish0','fish1'):
                files=list((out/'frames').glob(f'*_{cam}.jpg'));self.assertTrue(files)
                with Image.open(files[0]) as im:self.assertEqual(max(im.size),320)
            meta=json.loads((out/'meta.json').read_text());self.assertIsNotNone(meta['end_time'])
            self.assertEqual(meta['task'],'sort the items into their containers')
            self.assertEqual(meta['active_arm'],'right' if arm=='left' else 'left')
            if stale:self.assertIn('receipt age >2s',meta['exit_reason'])
            else:self.assertTrue(any(e.get('event')=='agent_judgement' for e in events))
            usage_data=json.loads((out/'agent_usage.json').read_text())
            self.assertEqual(usage_data['initial_context_chars'],len('test prompt'))
            self.assertEqual(usage_data['submitted_decisions'],2);self.assertEqual(usage_data['image_views'],3)
            self.assertIsNone(usage_data['tokens']);self.assertIn('ended_at',usage_data)
            self.assertTrue((out/'agent_final.json').exists())

    def test_files_inbox_preemption_snapshots_and_exact_wire(self):
        for arm in ('left','right'):
            with self.subTest(arm=arm):self.run_case(arm)

    def test_stale_camera_hold_recovery_and_stop(self):self.run_case('right','camera')

    def test_stale_state_hold_recovery_and_stop(self):self.run_case('left','state')


if __name__=='__main__':unittest.main()
