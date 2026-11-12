"""Chunk scheduling only. All task decisions come from the external agent."""
import math
import time
import threading
from functools import wraps
import numpy as np

from safety import ARMS, Abort, angle, workspace


def synchronized(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        with self.lock:return fn(self,*args,**kwargs)
    return wrapped


def validate_chunk(chunk,snapshot,active_arm,demo_index):
    cid=chunk.get('chunk_id')
    if type(cid) is not int or cid<0:raise ValueError('chunk_id must be a nonnegative integer')
    for k in ('phase','reason','snapshot_id'):
        if not isinstance(chunk.get(k),str) or not chunk[k].strip():raise ValueError('missing '+k)
    if chunk.get('active_arm')!=active_arm:raise ValueError('active arm differs from attempt')
    if chunk['snapshot_id']!=snapshot['snapshot_id']:raise ValueError('snapshot id mismatch')
    refs=chunk.get('demo_chunks')
    if not isinstance(refs,list) or not refs:raise ValueError('demo references required')
    for r in refs:
        if r.get('start_frame') not in demo_index.get(str(r.get('episode')),[]):
            raise ValueError('demo chunk outside this task context')
        if not isinstance(r.get('how'),str) or not r['how'].strip():raise ValueError('demo reference needs how')
    steps=chunk.get('steps')
    if not isinstance(steps,list) or not 30<=len(steps)<=90:raise ValueError('30 to 90 steps required (1-3 seconds)')
    if chunk.get('dt')!=1/30:raise ValueError('dt must be 1/30')
    idle='right' if active_arm=='left' else 'left'
    anchor=snapshot['executor']['idle_anchor']
    prev={a+'_ee':snapshot['state'][a+'_ee'] for a in ARMS}
    for step in steps:
        for arm in ARMS:
            workspace(arm,step[arm+'_ee'])
            g=step[arm+'_gripper']
            if not isinstance(g,(int,float)) or not math.isfinite(g) or not 0<=g<=1:
                raise ValueError('invalid gripper command')
            if math.dist(step[arm+'_ee'][4:],prev[arm+'_ee'][4:])>.005001 or angle(step[arm+'_ee'],prev[arm+'_ee'])>math.radians(2.001):
                # The idle anchor can be displaced by normal arm coupling.
                if arm!=idle:raise ValueError('predicted step exceeds 5mm or 2 degrees')
        if math.dist(step[idle+'_ee'][4:],anchor[4:])>.001 or angle(step[idle+'_ee'],anchor)>math.radians(.5):
            raise ValueError('idle arm must hold its measured start anchor')
        prev=step
    if max(math.dist(s[active_arm+'_ee'][4:],snapshot['state'][active_arm+'_ee'][4:]) for s in steps)>.065:
        raise ValueError('chunk exceeds 65mm from its observation')


class Scheduler:
    def __init__(self,safety,log):
        self.lock=threading.RLock()
        self.safety,self.log=safety,log
        self.current=None
        self.step=0
        self.last_id=-1
        self.hold=None
        self.mode=None
        self.grippers=None
        self.phase=None

    def status(self,cid,status,**fields):
        self.log.write('chunks',dict(record_type='status',time=time.time(),chunk_id=cid,status=status,**fields))
        self.log.ack(cid,dict(chunk_id=cid,status=status,**fields))

    @synchronized
    def discard(self,status,reason):
        if self.current is not None:
            cid=self.current['chunk_id']
            self.status(cid,status,reason=reason,steps_executed=self.step)
            self.log.event('chunk_'+status,chunk_id=cid,reason=reason,steps_executed=self.step)
            self.current=None

    @synchronized
    def accept(self,chunk):
        if chunk['chunk_id']<=self.last_id:
            self.status(chunk['chunk_id'],'rejected_by_safety',reason='older or duplicate chunk id')
            return
        self.last_id=chunk['chunk_id']
        self.discard('preempted','newer chunk received')
        self.current=chunk;self.step=0;self.hold=None;self.mode='chunk'
        if chunk['phase']!=self.phase:
            self.log.event('phase',previous=self.phase,phase=chunk['phase'],reason=chunk['reason'])
            self.phase=chunk['phase']
        self.status(chunk['chunk_id'],'running',steps_executed=0)
        self.log.event('chunk_start',chunk_id=chunk['chunk_id'],snapshot_id=chunk['snapshot_id'])

    @synchronized
    def tick(self,snap):
        fresh=self.safety.check(snap)
        s=snap['state']
        if self.grippers is None and s is not None:
            self.grippers={a:float(s[a+'_gripper']) for a in ARMS}
        if not fresh:
            self.discard('preempted','stream stale; remaining steps discarded')
            mode='stale_hold'
        elif self.current is None:
            mode='queue_empty_hold'
        else:
            mode='chunk'
        if mode!='chunk':
            if self.mode!=mode or self.hold is None:
                self.hold=None if s is None or snap['state_age']>.5 else {a+'_ee':list(s[a+'_ee']) for a in ARMS}
                self.mode=mode
                self.log.event('hold',reason=mode,state_age=snap['state_age'],camera_ages=snap['camera_ages'])
            if s is None or snap['state_age']>.5:
                return None,None,None,mode
            if self.hold is None:self.hold={a+'_ee':list(s[a+'_ee']) for a in ARMS}
            target=dict(self.hold,**{a+'_gripper':self.grippers[a] for a in ARMS})
            return self.safety.command(target,s),None,None,mode
        cid,index=self.current['chunk_id'],self.step
        target=self.current['steps'][index]
        command=self.safety.command(target,s)
        return command,cid,index,'chunk'

    @synchronized
    def sent(self,command,cid,index):
        self.safety.previous=command
        self.grippers={a:command[a+'_gripper'] for a in ARMS}
        if cid is not None:
            self.step=index+1
            if self.step==len(self.current['steps']):
                self.status(cid,'ran_fully',steps_executed=self.step)
                self.log.event('chunk_finish',chunk_id=cid,steps_executed=self.step)
                self.current=None;self.hold=None

    @synchronized
    def summary(self):
        return dict(mode=self.mode,phase=self.phase,chunk_id=self.current['chunk_id'] if self.current else None,
                    next_step=self.step,remaining_steps=len(self.current['steps'])-self.step if self.current else 0,
                    gripper_commands=self.grippers,
                    idle_anchor=self.safety.initial[self.safety.idle+'_ee'] if self.safety.initial else None,
                    active_arm=self.safety.active)
