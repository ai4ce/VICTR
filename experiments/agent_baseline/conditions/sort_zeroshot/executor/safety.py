"""Protocol limits and geometry, reused from step.py and the previous executor."""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ARMS = ('left', 'right')
BOXES = {'left': ((.10,.38),(-.24,.22),(.225,.46)),
         'right': ((.10,.38),(-.70,-.18),(.225,.46))}


class Abort(RuntimeError):
    pass


def pose(value):
    a = np.asarray(value, float)
    if a.shape != (7,) or not np.isfinite(a).all() or not .95 < np.linalg.norm(a[:4]) < 1.05:
        raise Abort('invalid pose')
    return a


def angle(a,b):
    a,b=pose(a)[:4],pose(b)[:4]
    return 2*math.acos(float(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1)))


def interpolate(a,b,f):
    a,b=pose(a),pose(b)
    rots=Rotation.from_quat(np.array([a,b])[:,[1,2,3,0]])
    q=Slerp([0,1],rots)(float(np.clip(f,0,1))).as_quat()[[3,0,1,2]]
    return np.r_[q,a[4:]+float(f)*(b[4:]-a[4:])].tolist()


def bounded(a,b,linear=.005,angular=math.radians(2)):
    a,b=pose(a),pose(b)
    f=min(1,linear/max(np.linalg.norm(b[4:]-a[4:]),1e-12),angular/max(angle(a,b),1e-12))
    return interpolate(a,b,f)


def workspace(arm,value):
    p=pose(value)
    if any(not lo <= v <= hi for v,(lo,hi) in zip(p[4:],BOXES[arm])):
        raise Abort(arm+' target outside workspace')


class Safety:
    def __init__(self,start,active_arm=None,timeout=3600):
        self.start,self.timeout,self.active=start,timeout,active_arm
        self.idle=None if active_arm is None else ('right' if active_arm=='left' else 'left')
        self.initial=None
        self.idle_anchor=None
        self.previous=None

    def check(self,snap,startup=False):
        if snap['now']-self.start >= self.timeout:
            raise Abort('hard timeout')
        s=snap['state']
        if s is not None:
            if s.get('is_recording') is not True: raise Abort('joystick gate off')
            for arm in ARMS:
                pose(s[arm+'_ee'])
                g=s.get(arm+'_gripper')
                if g is None or not math.isfinite(g) or not -.02<=g<=1.02:
                    raise Abort('invalid gripper readback')
                if s.get(arm+'_ik_solved') is False: raise Abort(arm+' IK failure')
            if self.initial is None:
                self.initial=s
            if self.idle is not None:
                if self.idle_anchor is None:self.idle_anchor=list(self.initial[self.idle+'_ee'])
                p,anchor=s[self.idle+'_ee'],self.idle_anchor
                if math.dist(p[4:],anchor[4:])>.040 or angle(p,anchor)>math.radians(15):
                    raise Abort('idle arm drift >40mm or 15deg')
            if self.previous:
                for arm in ARMS:
                    p,t=s[arm+'_ee'],self.previous[arm+'_ee']
                    tracking_limit=.040 if arm==self.idle else .035
                    if math.dist(p[4:],t[4:])>tracking_limit or angle(p,t)>math.radians(15):
                        raise Abort(f'{arm} tracking error >{tracking_limit*1000:g}mm or 15deg')
        ages=list(snap['camera_ages'].values())
        state_age=snap['state_age']
        # Protocol's >2s stop means packet age, not 2s after the hold threshold.
        if not startup and (state_age>2 or len(ages)!=3 or any(a>2 for a in ages)):
            raise Abort('stream receipt age >2s')
        return (s is not None and state_age<=.5 and len(ages)==3 and all(a<=1 for a in ages))

    def command(self,target,s):
        result={}
        for arm in ARMS:
            workspace(arm,target[arm+'_ee'])
            start=self.previous[arm+'_ee'] if self.previous else s[arm+'_ee']
            result[arm+'_ee']=bounded(start,target[arm+'_ee'])
            tracking_limit=.040 if arm==self.idle else .035
            if math.dist(result[arm+'_ee'][4:],s[arm+'_ee'][4:])>tracking_limit:
                raise Abort(f'{arm} proposed tracking error >{tracking_limit*1000:g}mm')
            if angle(result[arm+'_ee'],s[arm+'_ee'])>math.radians(15):
                raise Abort(arm+' proposed angular tracking error >15deg')
            g=target[arm+'_gripper']
            if not isinstance(g,(int,float)) or not math.isfinite(g) or not 0<=g<=1:
                raise Abort('invalid gripper command')
            result[arm+'_gripper']=float(g)
        return result
