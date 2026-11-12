"""Local subscribers only. No RPC, SSH or robot-side process control."""
import io
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
import zmq
from commlink import Subscriber

from safety import Abort
from files import append, atomic_json

CAMERAS = ('zed', 'fish0', 'fish1')


class Log:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/'frames').mkdir()
        self.lock = threading.RLock()
        for name in ('chunks', 'commands', 'state', 'events'):
            (self.out/(name+'.jsonl')).touch()
        for name in ('inbox','processed','acks','snapshots'):
            (self.out/name).mkdir()
        self.saved = set()

    def write(self, name, data):
        with self.lock:
            append(self.out,name,data)

    def event(self, event, **data):
        self.write('events', dict(time=time.time(), event=event, **data))

    def meta(self, data):
        atomic_json(self.out/'meta.json',data)

    def frames(self, frame_id, images):
        with self.lock:
            if frame_id in self.saved:
                return
            for cam, image in images.items():
                im = Image.fromarray(image)
                im.thumbnail((320, 320))
                im.save(self.out/'frames'/f'{frame_id}_{cam}.jpg', quality=85)
            self.saved.add(frame_id)

    def close(self):
        pass

    def ack(self,cid,data):
        import hashlib
        name=str(cid) if type(cid) is int and cid>=0 else 'rejected_'+hashlib.sha256(str(cid).encode()).hexdigest()[:16]
        atomic_json(self.out/'acks'/f'{name}.json',data)


class Freshness:
    """Packet age uses only this PC's monotonic arrival clock."""
    def __init__(self):
        self.source = None
        self.received = -math.inf

    def update(self, stamp, now):
        self.source, self.received = stamp, now  # source timestamp: logging only

    def age(self, now):
        return max(0., now-self.received)


class Streams:
    def __init__(self, log, cam_host='192.168.123.117'):
        self.log = log
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.fault = None
        self.state = None
        self.state_id = 0
        self.frame_id = 0
        self.images = None
        self.sf = Freshness()
        self.cf = Freshness()
        self.gate = None
        self.cam = Subscriber(cam_host, port=5550, topics=['cameras'], buffer=True, queue_size=1)
        self.threads = [threading.Thread(target=self._state_loop, daemon=True),
                        threading.Thread(target=self._camera_loop, daemon=True)]
        for t in self.threads:
            t.start()

    def _fail(self, message):
        with self.lock:
            if self.fault is None:
                self.fault = str(message)

    def _state_loop(self):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'teleop/episode')
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect('tcp://192.168.123.55:5568')
        try:
            while not self.stop_event.is_set():
                if not sock.poll(50):
                    continue
                wire = sock.recv_multipart()
                state = json.loads(wire[-1])
                now = time.monotonic()
                with self.lock:
                    self.state_id += 1
                    self.log.write('state', dict(time=time.time(), received_monotonic=now,
                        state_id=self.state_id, packet=state, wire=wire[-1].decode()))
                    gate = state.get('is_recording')
                    if gate != self.gate:
                        self.log.event('gate', enabled=gate)
                        self.gate = gate
                    if gate is not True:
                        self._fail('joystick gate off')
                    if any(state.get(a+'_ik_solved') is False for a in ('left', 'right')):
                        self._fail('IK failed')
                    self.sf.update(state.get('timestamp'), now)
                    self.state = state
        except Exception as e:
            self._fail('state receiver: '+str(e))
        finally:
            sock.close()

    def _camera_loop(self):
        last_saved = 0
        try:
            while not self.stop_event.is_set():
                packet = self.cam.get('cameras')
                now = time.monotonic()
                images = {}
                for cam in CAMERAS:
                    data = packet['cameras'][cam]
                    im = Image.open(io.BytesIO(data)).convert('RGB')
                    im.thumbnail((320,320))
                    images[cam] = np.array(im)
                with self.lock:
                    self.cf.update(packet.get('timestamp'), now)
                    self.images = images
                    self.frame_id += 1
                    frame_id = self.frame_id
                if now-last_saved >= .25:
                    self.log.frames(frame_id, images)
                    last_saved = now
        except Exception as e:
            if not self.stop_event.is_set():
                self._fail('camera receiver: '+str(e))

    def snapshot(self):
        with self.lock:
            if self.fault:
                raise Abort(self.fault)
            now = time.monotonic()
            ages = {cam: self.cf.age(now) for cam in CAMERAS}
            return dict(state=self.state, state_id=self.state_id, state_age=self.sf.age(now),
                camera_ages=ages, images=self.images, frame_id=self.frame_id,
                camera_timestamp=self.cf.source, now=now,
                state_received_monotonic=self.sf.received,
                camera_received_monotonic=self.cf.received)

    def close(self):
        self.stop_event.set()
        self.cam.stop()
        for t in self.threads:
            t.join(timeout=2)
