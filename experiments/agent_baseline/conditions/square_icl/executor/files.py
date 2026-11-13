"""Atomic publications and process-safe JSONL logging for executor and client."""
import contextlib
import fcntl
import json
import os
import math
from pathlib import Path
import tempfile
import time


def atomic_json(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,allow_nan=False,separators=(',',':'));f.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextlib.contextmanager
def locked(out):
    with open(Path(out)/'.log.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def append(out,name,value):
    line=json.dumps(value,allow_nan=False,separators=(',',':'))+'\n'
    with locked(out):
        with open(Path(out)/(name+'.jsonl'),'a') as f:f.write(line)


def event(out,name,**fields):
    append(out,'events',dict(time=time.time(),event=name,**fields))


def usage(out,**changes):
    with locked(out):
        path=Path(out)/'agent_usage.json'
        value=json.loads(path.read_text())
        views=changes.pop('add_image_views',0)
        text_chars=changes.pop('add_text_chars',0)
        decisions=changes.pop('add_decisions',0)
        value.update(changes)
        value['image_views']=value.get('image_views',0)+views
        value['observed_text_chars']=value.get('observed_text_chars',0)+text_chars
        value['submitted_decisions']=value.get('submitted_decisions',0)+decisions
        value['estimated_tokens']=math.ceil((value.get('initial_context_chars',0)+value['observed_text_chars'])/4
            +value['image_views']*256+value['submitted_decisions']*512)
        value['estimate_method']='workload proxy: context/text chars / 4 + 256 per image view + 512 planning tokens per submitted decision'
        value['estimate_limitations']='Heuristic, not billed usage; unknown image encoding, hidden reasoning and repeated/cached history can materially change actual totals.'
        value['wall_time_s']=value.get('ended_at',time.time())-value['started_at']
        atomic_json(path,value)
