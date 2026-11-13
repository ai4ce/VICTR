#!/usr/bin/env python3
"""One bounded, read-only connectivity check. No executor, binds, or publishers."""
import datetime
import io
import json
import math
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import zmq
from commlink import Subscriber
from commlink.subscriber import OP_PULL
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STATE = 'tcp://192.168.123.55:5568'
CAMERA = 'tcp://192.168.123.117:5550'
CAMERAS = ('zed', 'fish0', 'fish1')


def main():
    report = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  mode='read-only state SUB and commlink pull subscriber; no executor',
                  state_endpoint=STATE, camera_endpoint=CAMERA,
                  socket_types=[], connections=[], forbidden_operations=[],
                  bind_calls=0, publisher_sockets_created=0, commands_published=0,
                  gate_changes_requested=0, camera_read_requests=0,
                  state_packets=0, camera_packets=0, gate_values=[], cameras={}, errors=[])
    start = time.monotonic()
    original_socket = zmq.Context.socket
    original_connect = zmq.Socket.connect
    original_send = zmq.Socket.send_multipart

    def forbidden(description):
        report['forbidden_operations'].append(description)
        raise RuntimeError(description)

    def socket(context, kind, *args, **kwargs):
        if kind not in (zmq.SUB, zmq.DEALER):
            forbidden('non-subscriber socket requested: '+str(kind))
        result = original_socket(context, kind, *args, **kwargs)
        report['socket_types'].append('SUB' if kind == zmq.SUB else 'DEALER (camera reads only)')
        return result

    def connect(sock, address):
        expected = STATE if sock.type == zmq.SUB else CAMERA
        if address != expected:
            forbidden('unexpected connection: '+address)
        original_connect(sock, address)
        report['connections'].append(address)

    def send(sock, parts, *args, **kwargs):
        if not (sock.type == zmq.DEALER and len(parts) == 3 and
                parts[:2] == [OP_PULL, b'cameras'] and len(parts[2]) == 8):
            forbidden('non-camera-read send requested')
        report['camera_read_requests'] += 1
        return original_send(sock, parts, *args, **kwargs)

    def port_check():
        result = subprocess.run(['ss', '-H', '-ltn', 'sport = :5572'],
                                capture_output=True, text=True, timeout=5)
        checked = dict(returncode=result.returncode, stdout=result.stdout,
                       stderr=result.stderr, checked_at=time.time())
        checked['inspection_ok'] = result.returncode == 0 and not result.stderr.strip()
        checked['free'] = checked['inspection_ok'] and not result.stdout.strip()
        return checked

    state_sock = camera = None
    try:
        report['port_5572_before'] = port_check()
        if not report['port_5572_before']['free']:
            raise RuntimeError('port 5572 preflight failed; no stream connections opened')
        with patch.object(zmq.Context, 'socket', socket), \
                patch.object(zmq.Socket, 'connect', connect), \
                patch.object(zmq.Socket, 'bind', lambda *a, **kw: forbidden('bind requested')), \
                patch.object(zmq.Socket, 'send_multipart', send):
            try:
                state_sock = zmq.Context.instance().socket(zmq.SUB)
                state_sock.setsockopt(zmq.SUBSCRIBE, b'teleop/episode')
                state_sock.setsockopt(zmq.LINGER, 0)
                state_sock.connect(STATE)
                # Buffered Subscriber creates an inproc PAIR bind. Pull mode does not.
                camera = Subscriber('192.168.123.117', port=5550, topics=['cameras'], buffer=False)
                camera_sock, _ = camera._get_req_socket('cameras')
                camera_sock.setsockopt(zmq.RCVTIMEO, 200)
                camera_sock.setsockopt(zmq.SNDTIMEO, 200)
                camera_sock.setsockopt(zmq.LINGER, 0)
                deadline = time.monotonic()+12
                while time.monotonic() < deadline:
                    while state_sock.poll(0):
                        wire = state_sock.recv_multipart()
                        if wire[0] != b'teleop/episode':
                            raise ValueError('unexpected state topic')
                        state = json.loads(wire[-1])
                        for arm in ('left', 'right'):
                            pose = state[arm+'_ee']
                            if len(pose) != 7 or not all(math.isfinite(x) for x in pose):
                                raise ValueError('unreadable measured pose')
                            if not math.isfinite(state[arm+'_gripper']):
                                raise ValueError('unreadable gripper')
                        gate = state['is_recording']
                        if type(gate) is not bool:
                            raise ValueError('gate is not a boolean')
                        if gate not in report['gate_values']:
                            report['gate_values'].append(gate)
                        report['state_packets'] += 1
                        report['last_state_received_s'] = time.monotonic()-start
                    try:
                        # One bounded read; do not use get(), which waits indefinitely for first data.
                        changed, packet = camera._pull_once('cameras')
                    except zmq.Again:
                        changed, packet = False, None
                    if changed:
                        for name in CAMERAS:
                            data = packet['cameras'][name]
                            with Image.open(io.BytesIO(data)) as im:
                                im.load()
                                report['cameras'][name] = dict(readable=True, width=im.width,
                                    height=im.height, mode=im.mode, encoded_bytes=len(data))
                        report['camera_packets'] += 1
                        report['last_cameras_received_s'] = time.monotonic()-start
                    if report['state_packets'] >= 2 and report['camera_packets'] >= 2:
                        break
                    time.sleep(.02)
            finally:
                if camera is not None:
                    camera.stop()
                if state_sock is not None:
                    state_sock.close(linger=0)
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        try:
            report['port_5572_after'] = port_check()
        except Exception as exc:
            report['errors'].append('postflight: '+str(exc))
        report['duration_s'] = time.monotonic()-start
        report['state_readable'] = report['state_packets'] >= 2
        report['gate_readable'] = bool(report['gate_values'])
        report['passed'] = (not report['errors'] and not report['forbidden_operations']
            and report.get('port_5572_before', {}).get('free', False)
            and report.get('port_5572_after', {}).get('free', False)
            and report['state_readable'] and report['gate_readable']
            and report['camera_packets'] >= 2 and all(c in report['cameras'] for c in CAMERAS))
        (ROOT/'checks/live_check.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
