#!/usr/bin/env python3
"""Async agent executor. This file contains no task policy or model invocation."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import queue
import re
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

import zmq

from core import Scheduler, validate_chunk
from files import atomic_json, usage
from safety import Abort, Safety
from streams import Log, Streams

ROOT=Path(__file__).resolve().parents[1]
TOPIC=b'inference/bimanual_ee_action'
TASK='sort the items into their containers'


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


class Workers:
    def __init__(self,log,streams,scheduler):
        self.log,self.streams,self.scheduler=log,streams,scheduler
        self.stop=threading.Event();self.requests=queue.Queue();self.fault=None
        self.threads=[threading.Thread(target=self.snapshots,daemon=True),
                      threading.Thread(target=self.inbox,daemon=True)]
        for t in self.threads:t.start()

    def snapshots(self):
        serial=0
        try:
            while not self.stop.is_set():
                snap=self.streams.snapshot()
                if snap['state'] is not None and snap['images'] is not None:
                    serial+=1;sid=f'snap_{serial:06d}'
                    self.log.frames(snap['frame_id'],snap['images'])
                    data={k:v for k,v in snap.items() if k!='images'}
                    data.update(snapshot_id=sid,created_at=time.time(),executor=self.scheduler.summary(),
                                frames={c:f'frames/{snap["frame_id"]}_{c}.jpg' for c in ('zed','fish0','fish1')})
                    atomic_json(self.log.out/'snapshots'/f'{sid}.json',data)
                    atomic_json(self.log.out/'latest.json',data)
                self.stop.wait(.25)
        except Exception as e:
            if not self.stop.is_set():self.fault='snapshot writer: '+str(e)

    def inbox(self):
        try:
            while not self.stop.is_set():
                for path in sorted((self.log.out/'inbox').glob('*.json')):
                    raw=path.read_text()
                    try:
                        data=json.loads(raw);error=None
                        if not isinstance(data,dict):data={};error='expected JSON object'
                    except Exception as e:data={};error=str(e)
                    received=time.time()
                    # Preserve the exact submission before processing or publishing.
                    self.log.write('chunks' if data.get('type','chunk')=='chunk' else 'events',
                        dict(record_type='submitted',time_submitted=data.get('submitted_at',received),time_received=received,file=path.name,
                             chunk_id=data.get('chunk_id'),submission=data,raw_json=raw,parse_error=error))
                    path.rename(self.log.out/'processed'/path.name)
                    self.requests.put((data,error,path.name))
                self.stop.wait(.015)
        except Exception as e:self.fault='inbox reader: '+str(e)

    def close(self):
        self.stop.set()
        for t in self.threads:t.join(timeout=2)


def process_request(data,error,filename,log,streams,scheduler,meta):
    if error or not isinstance(data,dict):
        scheduler.status(filename,'rejected_by_safety',reason='malformed JSON: '+str(error))
        return False
    kind=data.get('type','chunk')
    if kind=='finish':
        outcome=data.get('outcome')
        if outcome not in ('success','partial','failure') or not data.get('reason'):
            log.event('control_rejected',reason='finish requires outcome and reason');return False
        if data['reason']=='usage_limit':outcome='failure'
        meta.update(exit_reason=data['reason'],outcome_guess=outcome)
        log.event('agent_outcome',outcome=outcome,reason=data['reason'])
        scheduler.discard('preempted','agent finished attempt')
        return True
    if kind=='judgement':
        if data.get('kind') not in ('grasp','placement','phase','note') or not isinstance(data.get('reason'),str):
            log.event('control_rejected',reason='invalid judgement');return False
        log.event('agent_judgement',judgement=data)
        if data['kind']=='grasp' and data.get('confirmed') is True:
            meta['outcome_guess']='partial'
        return False
    if kind!='chunk':
        log.event('control_rejected',reason='unknown message type');return False
    cid=data.get('chunk_id',filename)
    try:
        sid=data.get('snapshot_id','')
        if not re.fullmatch(r'snap_\d{6,}',sid):raise ValueError('invalid snapshot id')
        observed=json.loads((log.out/'snapshots'/f'{sid}.json').read_text())
        if time.monotonic()-observed['now']>30:raise ValueError('snapshot more than 30s old; read a new one')
        current=streams.snapshot()
        if not scheduler.safety.check(current):raise ValueError('stream stale; hold until fresh and resubmit')
        validate_chunk(data,observed,scheduler.safety.active)
        # Do not let a chunk covertly change the idle gripper.
        idle='right' if data['active_arm']=='left' else 'left'
        grippers=scheduler.grippers or {a:scheduler.safety.initial[a+'_gripper'] for a in ('left','right')}
        if any(abs(s[idle+'_gripper']-grippers[idle])>.01 for s in data['steps']):
            raise ValueError('idle gripper must retain its last command')
        if scheduler.safety.active is not None and data['active_arm']!=scheduler.safety.active:
            # Validation can take time; anchor a switch to the latest receipt.
            current=streams.snapshot()
            if not scheduler.safety.check(current):raise ValueError('stream stale; hold until fresh and resubmit')
        if scheduler.accept(data,current['state']):
            if meta['active_arm']!=scheduler.safety.active:
                meta['active_arm']=scheduler.safety.active
                log.meta(meta)
    except Abort:
        scheduler.status(cid,'rejected_by_safety',reason='hard safety check failed')
        raise
    except Exception as e:
        scheduler.status(cid,'rejected_by_safety',reason=str(e))
        log.event('chunk_rejected',chunk_id=cid,reason=str(e))
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',default=TASK)
    p.add_argument('--out',required=True)
    p.add_argument('--agent-started-at',type=float)
    p.add_argument('--cam-host',default='192.168.123.117')
    args=p.parse_args()
    log=Log(args.out);start=time.monotonic()
    meta=dict(task=args.task,executor_git_commit=git('rev-parse','HEAD'),
              prompt_sha256=hashlib.sha256((ROOT/'attempt_prompt.md').read_bytes()).hexdigest(),
              start_time=time.time(),end_time=None,exit_reason='running',outcome_guess='failure',active_arm=None)
    log.meta(meta)
    atomic_json(log.out/'agent_usage.json',dict(started_at=args.agent_started_at or time.time(),
        wall_time_s=0,tokens=None,credits=None,usage_source='not yet supplied by session accounting',
        initial_context_chars=len((ROOT/'attempt_prompt.md').read_text()),
        image_views=0,wall_time_source='agent supplied' if args.agent_started_at else 'executor start proxy'))
    usage(log.out)
    streams=workers=pub=scheduler=None
    code=1
    def stop_signal(signum,frame):raise Abort('signal '+str(signum))
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    try:
        if args.task!=TASK:raise Abort('unsupported task')
        if git('status','--porcelain','--untracked-files=normal'):raise Abort('executor repo is not clean')
        if (ROOT/'FROZEN').exists():
            lines=(ROOT/'FROZEN').read_text().splitlines()
            # FROZEN is committed after the code/prompt commit it records.
            commit_ok=len(lines)==2 and (lines[0]==meta['executor_git_commit'] or (
                lines[0]==git('rev-parse','HEAD^') and
                git('diff','--name-only',lines[0],'HEAD','--')=='FROZEN'))
            if not commit_ok or not lines[1].split() or lines[1].split()[0]!=meta['prompt_sha256']:
                raise Abort('FROZEN commit or prompt hash mismatch')
        safety=Safety(start,timeout=3600)
        scheduler=Scheduler(safety,log)
        streams=Streams(log,args.cam_host)
        waiting=time.monotonic();next_notice=waiting
        while True:
            snap=streams.snapshot()
            if safety.check(snap,startup=True) and snap['images'] is not None:break
            now=time.monotonic()
            if now>=next_notice:
                text='waiting for streams (is the joystick loop started?)'
                print(text,flush=True);log.event('waiting_for_streams',message=text);next_notice=now+1
            if now-waiting>=10:raise Abort('startup streams unavailable after 10s')
            time.sleep(.02)
        pub=zmq.Context.instance().socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER,0);pub.setsockopt(zmq.SNDHWM,1)
        pub.bind('tcp://0.0.0.0:5572')
        connected=time.monotonic()
        while time.monotonic()-connected<.55:
            safety.check(streams.snapshot());time.sleep(.02)
        workers=Workers(log,streams,scheduler)
        print('Executor publishing at 30 Hz; holding measured pose until an agent chunk arrives.',flush=True)
        log.event('publishing_started',topic=TOPIC.decode())
        due=time.monotonic()
        while True:
            if workers.fault:raise Abort(workers.fault)
            finish=False
            for _ in range(10):
                try:data,error,name=workers.requests.get_nowait()
                except queue.Empty:break
                if process_request(data,error,name,log,streams,scheduler,meta):finish=True;break
            if finish:code=0;break
            with streams.lock:
                snap=streams.snapshot()
                command,cid,step,mode=scheduler.tick(snap)
                if command is not None:
                    command['timestamp']=time.time()
                    wire=json.dumps(command,allow_nan=False,separators=(',',':'))
                    pub.send_multipart([TOPIC,wire.encode()])
                    log.write('commands',dict(time=time.time(),chunk_id=cid,step=step,mode=mode,
                        state_id=snap['state_id'],message=command,topic=TOPIC.decode(),wire_json=wire))
                    scheduler.sent(command,cid,step)
            due+=1/30
            delay=due-time.monotonic()
            if delay>0:time.sleep(delay)
            else:due=time.monotonic()  # never burst old commands after a slow tick
    except BaseException as e:
        meta['exit_reason']='abort: '+str(e)
        log.event('safety_stop',reason=str(e),traceback=traceback.format_exc())
        if scheduler:scheduler.discard('rejected_by_safety',str(e))
        print(meta['exit_reason'],file=sys.stderr,flush=True)
    finally:
        if pub is not None:pub.close(linger=0)
        if workers is not None:workers.close()
        if streams is not None:streams.close()
        # Preserve pending requests even if an abort raced with their arrival.
        pending=[]
        if workers is not None:
            while not workers.requests.empty():pending.append(workers.requests.get_nowait())
        for file in sorted((log.out/'inbox').glob('*.json')):
            raw=file.read_text()
            try:
                data=json.loads(raw)
                if not isinstance(data,dict):data={}
            except Exception:data={}
            log.write('chunks' if data.get('type','chunk')=='chunk' else 'events',dict(record_type='submitted',time_submitted=time.time(),raw_json=raw,submission=data))
            file.rename(log.out/'processed'/file.name);pending.append((data,None,file.name))
        for data,error,name in pending:
            if data.get('type','chunk')=='chunk' and scheduler:
                scheduler.status(data.get('chunk_id',name),'rejected_by_safety',reason='executor stopped before execution')
        if (log.out/'agent_final.json').exists():
            final=json.loads((log.out/'agent_final.json').read_text())
            meta.update(outcome_guess=final['outcome'],agent_reason=final['reason'])
        meta.update(end_time=time.time(),duration_s=time.monotonic()-start)
        log.meta(meta);usage(log.out,ended_at=time.time());log.close()
    return code


if __name__=='__main__':raise SystemExit(main())
