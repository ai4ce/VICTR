"""Chunk scheduling only. All task decisions come from the external agent."""
import math
import copy
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


def validate_chunk(chunk,snapshot,active_arm):
    cid=chunk.get('chunk_id')
    if type(cid) is not int or cid<0:raise ValueError('chunk_id must be a nonnegative integer')
    for k in ('phase','reason','snapshot_id'):
        if not isinstance(chunk.get(k),str) or not chunk[k].strip():raise ValueError('missing '+k)
    if chunk.get('active_arm') not in ARMS:raise ValueError('active_arm must be left or right')
    active_arm=chunk['active_arm']
    if chunk['snapshot_id']!=snapshot['snapshot_id']:raise ValueError('snapshot id mismatch')
    steps=chunk.get('steps')
    if not isinstance(steps,list) or not 30<=len(steps)<=90:raise ValueError('30 to 90 steps required (1-3 seconds)')
    if chunk.get('dt')!=1/30:raise ValueError('dt must be 1/30')
    idle='right' if active_arm=='left' else 'left'
    selected=snapshot['executor']['active_arm']
    anchor=(snapshot['executor']['start_anchors'][idle] if selected is None else
            snapshot['executor']['idle_anchor'] if selected==active_arm else snapshot['state'][idle+'_ee'])
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
            raise ValueError('idle arm must hold its measured anchor')
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
    def accept(self,chunk,state=None):
        if chunk['chunk_id']<=self.last_id:
            self.status(chunk['chunk_id'],'rejected_by_safety',reason='older or duplicate chunk id')
            return False
        arm=chunk.get('active_arm')
        if arm not in ARMS:raise ValueError('active_arm must be left or right')
        previous_arm=self.safety.active
        if previous_arm is not None and arm!=previous_arm:
            if state is None:raise ValueError('switch requires current measured state')
            self.safety.active=arm
            self.safety.idle=previous_arm
            self.safety.idle_anchor=list(state[previous_arm+'_ee'])
            self.log.event('active_arm_switched',previous_arm=previous_arm,active_arm=arm,
                           idle_arm=previous_arm,idle_anchor=self.safety.idle_anchor,
                           idle_gripper=(self.grippers[previous_arm] if self.grippers is not None else
                                         self.safety.initial[previous_arm+'_gripper']),chunk_id=chunk['chunk_id'])
        if self.safety.active is None:
            self.safety.active=arm
            self.safety.idle='right' if arm=='left' else 'left'
            if self.safety.initial is not None:
                self.safety.idle_anchor=list(self.safety.initial[self.safety.idle+'_ee'])
            self.log.event('active_arm_selected',active_arm=arm,idle_arm=self.safety.idle,
                           chunk_id=chunk['chunk_id'])
        self.last_id=chunk['chunk_id']
        self.discard('preempted','newer chunk received')
        # The helper used its observation; acceptance anchors the idle arm now.
        # Preserve the exact submitted chunk in the log and normalize only this copy.
        if self.safety.idle_anchor is not None:
            chunk=copy.deepcopy(chunk)
            for step in chunk['steps']:
                step[self.safety.idle+'_ee']=list(self.safety.idle_anchor)
                if self.grippers is not None:step[self.safety.idle+'_gripper']=self.grippers[self.safety.idle]
        self.current=chunk;self.step=0;self.hold=None;self.mode='chunk'
        if chunk['phase']!=self.phase:
            self.log.event('phase',previous=self.phase,phase=chunk['phase'],reason=chunk['reason'])
            self.phase=chunk['phase']
        self.status(chunk['chunk_id'],'running',steps_executed=0)
        self.log.event('chunk_start',chunk_id=chunk['chunk_id'],snapshot_id=chunk['snapshot_id'])
        return True

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
                anchor=self.safety.initial if self.safety.active is None else s
                self.hold=None if s is None or snap['state_age']>.5 else {a+'_ee':list(anchor[a+'_ee']) for a in ARMS}
                self.mode=mode
                self.log.event('hold',reason=mode,state_age=snap['state_age'],camera_ages=snap['camera_ages'])
            if s is None or snap['state_age']>.5:
                return None,None,None,mode
            if self.hold is None:
                anchor=self.safety.initial if self.safety.active is None else s
                self.hold={a+'_ee':list(anchor[a+'_ee']) for a in ARMS}
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
                    idle_anchor=self.safety.idle_anchor,
                    start_anchors={a:self.safety.initial[a+'_ee'] for a in ARMS} if self.safety.initial else None,
                    active_arm=self.safety.active)
