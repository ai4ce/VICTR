#!/usr/bin/env python3
"""One build-time read-only check. No executor, publisher, bind, or saved scene.

State uses SUB; commlink pull mode uses a DEALER and read-only PULL requests.
Pull mode avoids commlink push mode's internal inproc control-socket bind.
"""
import io
import json
import subprocess
import threading
import time
from pathlib import Path

import zmq
from PIL import Image
from commlink import Subscriber
from commlink.subscriber import OP_PULL


def forbidden(*args, **kwargs):
    raise AssertionError('bind/publication forbidden in read-only check')


def ports():
    result = subprocess.run(['ss', '-H', '-ltnp'], capture_output=True, text=True, check=True)
    if result.stderr.strip():
        raise RuntimeError('port inspection: '+result.stderr.strip())
    if any(line.split()[3].endswith(':5572') for line in result.stdout.splitlines()):
        raise RuntimeError('5572 is occupied')
    return True


def main():
    report = dict(timestamp=time.time(), no_bind=True, no_publish=True, passed=False)
    cam = sock = None
    try:
        report['port_5572_free_before'] = ports()
        # Enforce no bind/PUB and allow only commlink camera read requests.
        original_send = zmq.Socket.send
        original_multipart = zmq.Socket.send_multipart
        def send_read(socket, data, *args, **kwargs):
            # send_multipart delegates its individual frames to send(). The
            # complete message is checked below; only the camera DEALER sends.
            if socket.getsockopt(zmq.TYPE) != zmq.DEALER:
                return forbidden()
            return original_send(socket, data, *args, **kwargs)
        def read_request(socket, parts, *args, **kwargs):
            if (socket.getsockopt(zmq.TYPE) != zmq.DEALER or len(parts) != 3 or
                    parts[:2] != [OP_PULL, b'cameras']):
                return forbidden()
            return original_multipart(socket, parts, *args, **kwargs)
        zmq.Socket.bind = forbidden
        zmq.Socket.send = send_read
        zmq.Socket.send_multipart = read_request
        original_socket = zmq.Context.socket
        def subscribe_only(context, kind, *args, **kwargs):
            if kind not in (zmq.SUB, zmq.DEALER):
                raise AssertionError('only state SUB and camera read DEALER allowed')
            return original_socket(context, kind, *args, **kwargs)
        zmq.Context.socket = subscribe_only
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'teleop/episode')
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect('tcp://192.168.123.55:5568')
        cam = Subscriber('192.168.123.117', port=5550, topics=['cameras'], buffer=False)
        received, errors = {}, []
        def cameras():
            try:
                packet = cam.get('cameras')
                receipt = time.monotonic()
                sizes = {}
                for name in ('zed', 'fish0', 'fish1'):
                    with Image.open(io.BytesIO(packet['cameras'][name])) as image:
                        image.load();sizes[name] = list(image.size)
                received.update(cameras=sizes, camera_receipt=receipt)
            except Exception as error:
                errors.append(str(error))
        thread = threading.Thread(target=cameras, daemon=True)
        thread.start()
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if sock.poll(50):
                state = json.loads(sock.recv_multipart()[-1])
                received.update(state=state, state_receipt=time.monotonic())
            if errors:
                raise RuntimeError('; '.join(errors))
            if 'state' in received and 'cameras' in received:
                break
        if 'state' not in received or 'cameras' not in received:
            raise RuntimeError('state or cameras unavailable after 10 seconds')
        state = received['state']
        assert type(state.get('is_recording')) is bool, 'gate unreadable'
        assert all(len(state[a+'_ee']) == 7 for a in ('left', 'right')), 'state poses unreadable'
        now = time.monotonic()
        report.update(gate_readable=True, gate_enabled=state['is_recording'],
                      state_readable=True, state_age_s=now-received['state_receipt'],
                      camera_age_s=now-received['camera_receipt'], camera_sizes=received['cameras'])
        assert report['state_age_s'] <= .5 and report['camera_age_s'] <= 1, 'stale stream'
        report['port_5572_free_after'] = ports()
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
    finally:
        if sock is not None:
            sock.close(linger=0)
        if cam is not None:
            cam.stop()
        (Path(__file__).parent/'live_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
