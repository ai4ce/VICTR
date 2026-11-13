"""
Deploy-time camera observation streamer (Orin side).

Owns the four cameras (ZED + 3 fisheyes) and at SYNC_HZ publishes a
JPEG-encoded camera bundle over commlink for the inference server.

This script publishes on loopback :5549 only. It does NOT feed the reverse
SSH tunnel directly — camera_proxy.py sits in between (SUB :5549 → PUB :5550)
and the tunnel forwards the proxy's :5550 output. See CLAUDE.md "Deployment
workflow" for the full path.

Server-side wiring (server is on a different network, so use SSH forwarding):
    # on the orin: run this script and the proxy, then reverse-tunnel :5550
    python deploy_observer.py                 # PUB obs/cameras on :5549
    python camera_proxy.py                     # SUB :5549 → PUB cameras on :5550
    ssh -N -R 5550:127.0.0.1:5550 <user>@<server>
    # then on the server (topic is "cameras", port 5550):
    from commlink import Subscriber
    sub = Subscriber("127.0.0.1", port=5550, topics=["cameras"])
    pkt = sub.get("cameras")

Robot proprioceptive state is published by the Pi and reaches the server
through its own channel — this script does not touch it.
"""

import threading
import time
import cv2
import pyzed.sl as sl
from commlink import Publisher

# --- Network ---
PUB_HOST  = "127.0.0.1"   # bind loopback; SSH tunnel reaches via localhost
PUB_PORT  = 5549
OBS_TOPIC = "obs/cameras"

# --- Capture / send ---
SYNC_HZ      = 30   # match the 30 Hz action rate during inference
JPEG_QUALITY = 70

# Cameras are addressed by stable USB-port path, NOT /dev/videoN (match collect.py).
# The two gripper fisheyes are the same model with the same serial (SN0001), so raw
# indices can silently swap left/right on replug/reboot. by-path tracks the port.
_BYPATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:{port}:1.0-video-index0"
FISH0_DEV = _BYPATH.format(port="4.2.1")   # left-hand  (Innomaker)
FISH1_DEV = _BYPATH.format(port="4.2.3")   # right-hand (Innomaker)
FISH2_DEV = _BYPATH.format(port="4.2.2")   # under-ZED  (16MP)

def _v4l2_set_ctrl(dev, **ctrls):
    """Best-effort UVC control set via v4l2-ctl (OpenCV can't reach these vendor
    controls). Used to disable exposure_dynamic_framerate: with it on the sensor may
    HALVE its own frame rate to lengthen exposure in dim light, so a camera facing a
    darker scene silently drops well below 30 fps (measured: the 16MP fell to 20 fps)
    — bad for a 30 Hz inference stream. Pinning it off holds a solid 30 fps.

    Also pins auto_exposure=3 (Aperture Priority = auto). UVC control state lives in
    the camera firmware and survives release()/restart/reboot, resetting only on USB
    re-enumeration, so a camera left in Manual Mode by a stray v4l2-ctl/guvcview
    session stays blown out indefinitely — and inference would then see a very
    different image from the one collect.py trained on."""
    import subprocess
    for name, val in ctrls.items():
        try:
            subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={name}={val}"],
                           check=False, capture_output=True, timeout=2)
        except Exception as e:
            print(f"[v4l2] could not set {name}={val} on {dev}: {e}")

# ---------------------------------------------------------------------------
# Shared latest-frame buffer
# ---------------------------------------------------------------------------
_lock   = threading.Lock()
_latest = {"zed": None, "fish0": None, "fish1": None, "fish2": None}

def _set(key, value):
    with _lock:
        _latest[key] = value

def _snapshot():
    with _lock:
        return dict(_latest)

# ---------------------------------------------------------------------------
# Camera threads
# ---------------------------------------------------------------------------
def zed_thread():
    zed  = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps        = 30
    init.depth_mode        = sl.DEPTH_MODE.NONE
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("[ZED] failed to open")
        return
    img = sl.Mat()
    rt  = sl.RuntimeParameters()
    print("[ZED] opened HD720 @ 30 fps")
    while True:
        if zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(img, sl.VIEW.LEFT)
            _set("zed", img.get_data().copy())

def fisheye_thread(idx, key, flip=False):
    # Outer loop reopens the device after a stall so a wedged capture handle (every
    # read() returning ok=False) can't freeze this camera for the rest of the run.
    while True:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[{key}] failed to open {idx}, retrying...")
            time.sleep(1)
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # BUFFERSIZE=1 kills double-buffering on this V4L2 stack and halves the rate
        # to ~15 fps; 2 restores 30 fps at ~1 frame latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # Don't let the sensor drop its frame rate to lengthen exposure in dim light.
        _v4l2_set_ctrl(idx, exposure_dynamic_framerate=0, auto_exposure=3)
        print(f"[{key}] opened {idx} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        fail_count = 0
        while True:
            ok, frame = cap.read()
            if ok:
                fail_count = 0
                if flip:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                _set(key, frame)
            else:
                fail_count += 1
                time.sleep(0.01)
                if fail_count > 50:      # ~0.5s of dead reads → reconnect
                    print(f"[{key}] repeated read failures, reconnecting {idx}...")
                    cap.release()
                    time.sleep(0.2)
                    break

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def _jpeg(img):
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None

# ---------------------------------------------------------------------------
# Main publish loop
# ---------------------------------------------------------------------------
def main():
    for target, args in [
        (zed_thread,     ()),
        (fisheye_thread, (FISH0_DEV, "fish0", False)),
        (fisheye_thread, (FISH1_DEV, "fish1", False)),
        (fisheye_thread, (FISH2_DEV, "fish2", True)),
    ]:
        threading.Thread(target=target, args=args, daemon=True).start()

    pub = Publisher(PUB_HOST, port=PUB_PORT)
    print(f"[pub] {OBS_TOPIC} on tcp://{PUB_HOST}:{PUB_PORT}")

    interval  = 1.0 / SYNC_HZ
    next_tick = time.time()
    frames    = 0
    last_log  = time.time()
    while True:
        next_tick += interval
        s = _snapshot()

        pkt = {
            "timestamp": time.time(),
            "cameras": {
                "zed":   _jpeg(s["zed"]),
                "fish0": _jpeg(s["fish0"]),
                "fish1": _jpeg(s["fish1"]),
                "fish2": _jpeg(s["fish2"]),
            },
        }
        pub[OBS_TOPIC] = pkt

        # Throttle logging to ~1 Hz — printing every tick at 30 Hz can itself
        # stall the loop on the Orin and drag the effective rate below 30 fps.
        frames += 1
        now = time.time()
        if now - last_log >= 1.0:
            cams_ok     = sum(1 for v in pkt["cameras"].values() if v is not None)
            bytes_total = sum(len(v) for v in pkt["cameras"].values() if v is not None)
            fps         = frames / (now - last_log)
            print(f"[obs] {fps:4.1f} fps  cams={cams_ok}/4  {bytes_total/1024:.0f} KB/frame")
            frames   = 0
            last_log = now

        sleep_s = next_tick - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # Falling behind: encode/publish took longer than one tick.
            # Resync so we don't accumulate unbounded backlog.
            next_tick = time.time()

if __name__ == "__main__":
    main()
