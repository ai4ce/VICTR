#!/usr/bin/env python3
"""Build-time stream health only: SUB/connect, no PUB/bind, no saved scene data."""
import argparse
import io
import json
import subprocess
import threading
import time
from pathlib import Path

import zmq
from commlink import Subscriber
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    ports=subprocess.run(['ss','-ltnp'],text=True,capture_output=True,check=True)
    owners=[line for line in ports.stdout.splitlines()[1:] if line.split()[3].rsplit(':',1)[-1]=='5572']
    report=dict(check='read-only live stream health',checked_at=time.time(),
                command_port_5572_free=None if ports.stderr else not owners,command_port_owners=owners,
                port_check_error=ports.stderr.strip() or None,
                publisher_created=False,command_port_bound=False,scene_data_saved=False)
    context=zmq.Context()
    state=context.socket(zmq.SUB)
    state.setsockopt(zmq.SUBSCRIBE,b'teleop/episode');state.setsockopt(zmq.LINGER,0)
    state.connect('tcp://192.168.123.55:5568')
    camera=Subscriber('192.168.123.117',port=5550,topics=['cameras'],buffer=True,queue_size=1)
    stop=threading.Event();lock=threading.Lock();camera_health={};errors=[]
    def read_cameras():
        try:
            while not stop.is_set():
                packet=camera.get('cameras');received=time.monotonic();valid={}
                for name in ('zed','fish0','fish1','fish2'):
                    with Image.open(io.BytesIO(packet['cameras'][name])) as im:
                        im.load();valid[name]=dict(size=list(im.size),received=received)
                with lock:camera_health.update(valid)
        except Exception as exc:
            if not stop.is_set():errors.append(str(exc))
    worker=threading.Thread(target=read_cameras,daemon=True);worker.start()
    state_received=None;gate=None;gate_readable=False;state_valid=False;passed=False
    deadline=time.monotonic()+12
    try:
        while time.monotonic()<deadline:
            if state.poll(50):
                packet=json.loads(state.recv_multipart()[-1]);state_received=time.monotonic()
                gate=packet.get('is_recording');gate_readable=type(gate) is bool
                state_valid=all(isinstance(packet.get(a+'_ee'),list) and len(packet[a+'_ee'])==7
                                for a in ('left','right'))
            now=time.monotonic()
            with lock:
                cameras={name:dict(age_s=now-value['received'],size=value['size'])
                         for name,value in camera_health.items()}
            state_age=None if state_received is None else now-state_received
            passed=(not ports.stderr and not owners and gate_readable and state_valid and state_age is not None and state_age<=.5
                    and len(cameras)==4 and all(v['age_s']<=1 for v in cameras.values()) and not errors)
            if passed:break
    finally:
        stop.set();state.close();context.term();camera.stop();worker.join(timeout=2)
    report.update(passed=passed,state_age_s=state_age,state_pose_fields_readable=state_valid,
                  gate_readable=gate_readable,gate_enabled=gate,cameras=cameras,errors=errors)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
