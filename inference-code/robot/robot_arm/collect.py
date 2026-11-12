"""
Data collection server — saves episodes in LeRobot v2.1 format under ./data/.
Episode boundaries are detected from teleop idle/active transitions.
Sync rate: SYNC_HZ (30 Hz, fisheye rate).
"""

import argparse
import av
import json
import queue
import re
import signal
import time
import threading
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import pyzed.sl as sl
from flask import Flask, Response, jsonify
from pathlib import Path
import zmq

from config import PI_IP

WEB_PORT       = 8080
JPEG_QUALITY   = 70
# --- web preview (MJPEG viewer) only; does NOT affect recorded data ---
# The viewer streams over WiFi, so keep its bitrate low or frames back up in the
# socket buffer and the browser lags seconds behind. These downscale/rate/quality
# knobs apply to the preview stream only — recording uses the raw frames.
WEB_FPS          = 15     # preview stream rate (Hz), independent of SYNC_HZ
WEB_MAX_WIDTH    = 640    # downscale wider frames (e.g. the 1280px ZED) for preview
WEB_JPEG_QUALITY = 60     # preview JPEG quality (lower than saved quality)
SYNC_HZ        = 30        # sync / save rate (Hz)
TELEOP_STALE_S = 0.5       # seconds without new teleop frame → episode end
# Episodes are recorded to the T7 SSD, not the Orin's eMMC — the eMMC root fs runs
# ~98% full and a single session can fill what's left. The T7 is exfat, label "T7",
# UUID E05A-0AAC; see _require_t7_mounted() for the not-mounted guard.
T7_MOUNT       = Path("/mnt/t7")
T7_UUID        = "E05A-0AAC"
DATASET_ROOT   = T7_MOUNT / "yor_data"
CHUNK_SIZE     = 100       # episodes per chunk folder
VIDEO_FPS      = SYNC_HZ

# Camera keys (lerobot naming convention)
CAM_ZED   = "observation.images.zed"
CAM_FISH0 = "observation.images.fish0"
CAM_FISH1 = "observation.images.fish1"

# (H, W, C) per camera. Used both for the info.json feature shapes AND to size the
# black placeholder written when a camera frame is momentarily missing, so every
# camera's mp4 stays frame-for-frame aligned with the parquet rows (LeRobot indexes
# video strictly by frame number and assumes len(video) == len(parquet)).
CAM_SHAPES = {
    CAM_ZED:   (720, 1280, 3),
    CAM_FISH0: (480, 640, 3),
    CAM_FISH1: (480, 640, 3),
}
# Per-episode image stats (for meta/episodes_stats.jsonl) are accumulated off the
# sync loop inside each encoder thread. Sampling every Nth encoded frame keeps the
# cost negligible while still giving representative min/max/mean/std per channel.
IMG_STATS_STRIDE = 10

# Cameras are addressed by stable USB-port path, NOT /dev/videoN.
# The two gripper fisheyes are the same model (Innomaker U20CAM) and report the
# same serial (SN0001), so /dev/v4l/by-id is ambiguous and raw video indices can
# silently swap left/right on any replug/reboot. by-path tracks the physical port.
_BYPATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:{port}:1.0-video-index0"
FISH0_DEV = _BYPATH.format(port="4.2.1")   # left-hand fisheye  (Innomaker)
FISH1_DEV = _BYPATH.format(port="4.2.3")   # right-hand fisheye (Innomaker)
# The under-ZED 16MP camera on port 4.2.2 is deliberately NOT opened here: it is
# not part of this dataset's schema, and leaving its sensor idle cuts current draw
# on the shared 1-4.2 hub leg that the gripper fisheyes sit on. deploy_observer.py
# still owns and publishes it. Not opening it does not depower it — the device stays
# enumerated and drawing idle current until it is physically unplugged from 4.2.2.

def _v4l2_set_ctrl(dev, **ctrls):
    """Best-effort UVC control set via v4l2-ctl (OpenCV can't reach these vendor
    controls). Used to disable exposure_dynamic_framerate: with it on, the sensor
    is allowed to HALVE its own frame rate to lengthen exposure in dim light, so a
    camera pointed at a darker scene silently drops to ~20 fps while its neighbours
    run 30 (measured: the under-ZED 16MP fell to 20 fps this way). Pinning it off
    keeps every camera at a solid 30 fps regardless of what it's looking at.

    Also used to pin auto_exposure=3 (Aperture Priority = auto). UVC control state
    lives in the camera's firmware, not in this process: it survives release(),
    script restarts and host reboots, and only resets on USB re-enumeration. A
    stray v4l2-ctl/guvcview session once left the right gripper locked in Manual
    Mode at 58 ms, blowing it out for every recording afterwards with nothing in
    this repo to explain it. Setting it on every open makes both grippers start
    from the same known auto-exposure state."""
    import subprocess
    for name, val in ctrls.items():
        try:
            subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={name}={val}"],
                           check=False, capture_output=True, timeout=2)
        except Exception as e:
            print(f"[v4l2] could not set {name}={val} on {dev}: {e}")

# ---------------------------------------------------------------------------
# Shared latest-frame buffers
# ---------------------------------------------------------------------------
_lock   = threading.Lock()
_latest = {"zed": None, "fish0": None, "fish1": None, "teleop": None, "episode_result": None, "signal": None}

_status = {
    "in_episode":      False,
    "current_episode": 0,
    "current_frame":   0,
    "task":            "",
    "last_episode":    None,
    "last_result":     None,   # "success" / "fail" / "unknown"
    "last_frames":     None,
}

def _set(key, value):
    with _lock:
        _latest[key] = value

def _get(key):
    with _lock:
        return _latest[key]

def _snapshot():
    with _lock:
        return dict(_latest)

def update_status(**kwargs):
    with _lock:
        _status.update(kwargs)

def get_status():
    with _lock:
        return dict(_status)

# ---------------------------------------------------------------------------
# MJPEG web viewer
# ---------------------------------------------------------------------------
_PLACEHOLDER: dict[str, np.ndarray] = {}

def _placeholder(key: str) -> np.ndarray:
    if key not in _PLACEHOLDER:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{key}  — no signal", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        _PLACEHOLDER[key] = img
    return _PLACEHOLDER[key]

def _encode(frame: np.ndarray) -> bytes:
    if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # Downscale wide frames (the ZED) so the WiFi preview bitrate stays low.
    if frame is not None and frame.shape[1] > WEB_MAX_WIDTH:
        scale = WEB_MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (WEB_MAX_WIDTH, int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY])
    return buf.tobytes() if ok else b""

def _mjpeg_stream(key: str):
    # This generator must never raise: an exception ends the multipart/x-mixed-replace
    # response, and the browser <img> then stops updating until a manual reload. A bad
    # or malformed frame must not take the connection down, so per-frame work is guarded
    # and we fall back to the placeholder instead of letting the stream die.
    interval = 1 / WEB_FPS
    while True:
        try:
            frame = _get(key)
            data  = _encode(frame if frame is not None else _placeholder(key))
            if not data:
                data = _encode(_placeholder(key))
            if data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        except Exception as e:
            print(f"[web] stream {key} frame error: {e}")
        time.sleep(interval)

_app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>YOR Teleoperation</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #ccc; font-family: 'Courier New', monospace;
           display: flex; flex-direction: column; min-height: 100vh; }
    header { text-align: center; padding: 10px 0 4px; font-size: 0.9em;
             letter-spacing: 2px; color: #555; text-transform: uppercase; }
    .status-bar { display: flex; gap: 14px; justify-content: center; align-items: center;
                  padding: 4px 0 8px; font-size: 0.78em; letter-spacing: 1px;
                  border-bottom: 1px solid #1a1a1a; }
    .pill { padding: 3px 12px; border-radius: 10px; font-weight: bold; letter-spacing: 1.5px; }
    .pill.idle      { background: #1a1a1a; color: #666; border: 1px solid #2a2a2a; }
    .pill.recording { background: #3a1010; color: #ff6b6b; border: 1px solid #5a1414;
                      animation: blink 1.2s ease-in-out infinite; }
    @keyframes blink { 50% { background: #5a1414; color: #ff9090; } }
    .info     { color: #888; }
    .task     { color: #6a8; }
    .last     { color: #555; }
    .last .pass { color: #6c6; } .last .fail { color: #c66; } .last .unk { color: #aa6; }
    .grid { display: flex; gap: 6px; padding: 6px 6px 0; align-items: stretch;
            justify-content: center; }
    .bottom-row { display: flex; justify-content: center; padding: 6px; }
    /* Sized rather than flex:1 — with the under-ZED tile gone the two grippers
       would otherwise stretch to ~50% of the page each. */
    .cam { display: flex; flex-direction: column; flex: 0 1 40%; }
    .cam.under-zed { flex: none; width: 42%; }
    .label { text-align: center; font-size: 0.72em; letter-spacing: 1.5px;
             color: #444; text-transform: uppercase; padding-bottom: 4px; }
    .cam img { width: 100%; height: auto; display: block;
               border: 1px solid #222; border-radius: 3px; background: #111; }
  </style>
</head>
<body>
  <header>YOR &mdash; Teleoperation Camera View</header>
  <div class="status-bar">
    <span id="state-pill" class="pill idle">IDLE</span>
    <span id="ep-info" class="info">&mdash;</span>
    <span id="task-info" class="task"></span>
    <span id="last-info" class="last">last: &mdash;</span>
  </div>
  <div class="grid">
    <div class="cam"><div class="label">Left Hand</div><img src="/stream/fish0" /></div>
    <div class="cam"><div class="label">Right Hand</div><img src="/stream/fish1" /></div>
  </div>
  <div class="bottom-row">
    <div class="cam under-zed"><div class="label">ZED</div><img src="/stream/zed" /></div>
  </div>
  <script>
    const RESULT = {
      success: {label: 'PASS ✓', cls: 'pass'},
      fail:    {label: 'FAIL ✗', cls: 'fail'},
      unknown: {label: '????',        cls: 'unk'},
    };
    async function tick() {
      try {
        const s = await (await fetch('/status')).json();
        const pill = document.getElementById('state-pill');
        const ep   = document.getElementById('ep-info');
        if (s.in_episode) {
          pill.textContent = 'RECORDING';
          pill.className   = 'pill recording';
          ep.textContent   = `ep ${s.current_episode} · frame ${s.current_frame}`;
        } else {
          pill.textContent = 'IDLE';
          pill.className   = 'pill idle';
          ep.textContent   = `next: ep ${s.current_episode}`;
        }
        document.getElementById('task-info').textContent = s.task ? `task: ${s.task}` : '';
        const last = document.getElementById('last-info');
        if (s.last_result) {
          const r = RESULT[s.last_result];
          last.innerHTML = `last: ep ${s.last_episode} · ` +
                           `<span class="${r.cls}">${r.label}</span> ` +
                           `(${s.last_frames} frames)`;
        } else {
          last.textContent = 'last: —';
        }
      } catch (e) {}
    }
    setInterval(tick, 500); tick();

    // Keep the camera tiles connected: if an MJPEG stream ever drops (server
    // restart, transient network hiccup), the <img> fires onerror and stops for
    // good. Re-point its src with a cache-busting query so the browser reopens the
    // multipart stream automatically instead of showing a broken image.
    document.querySelectorAll('img[src^="/stream/"]').forEach(img => {
      const base = img.getAttribute('src').split('?')[0];
      img.addEventListener('error', () => {
        setTimeout(() => { img.src = base + '?t=' + Date.now(); }, 1000);
      });
    });
  </script>
</body>
</html>
"""

@_app.route("/")
def _index():
    return _HTML

@_app.route("/stream/<cam>")
def _stream(cam):
    if cam not in ("zed", "fish0", "fish1"):
        return "not found", 404
    return Response(_mjpeg_stream(cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@_app.route("/status")
def _status_route():
    return jsonify(get_status())

# ---------------------------------------------------------------------------
# Camera / teleop threads
# ---------------------------------------------------------------------------
def zed_thread():
    # Outer loop reopens the camera after a stall, mirroring fisheye_thread. The ZED 2i
    # is a composite device straddling both buses — USB3 video on 2-3.4, but its USB2
    # HID control interface on 1-4.4.2, which hangs off the SAME physical hub as the
    # fisheyes on the 1-4.2 leg. A disturbance there (unplugging a fisheye, a power
    # transient) can knock the HID offline and make grab() fail forever. Without a
    # reopen the ZED silently stops updating for the rest of the session and every
    # subsequent episode is padded with a frozen frame by add_frame — a dataset that
    # looks structurally perfect but has dead video.
    while True:
        zed  = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps        = 30
        init.depth_mode        = sl.DEPTH_MODE.NONE
        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("[ZED] failed to open, retrying...")
            zed.close()
            time.sleep(1)
            continue
        img = sl.Mat()
        rt  = sl.RuntimeParameters()
        print("[ZED] opened HD720 @ 30 fps")
        fail_count = 0
        while True:
            if zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                fail_count = 0
                zed.retrieve_image(img, sl.VIEW.LEFT)
                _set("zed", img.get_data().copy())
            else:
                fail_count += 1
                time.sleep(0.01)
                if fail_count > 50:      # ~0.5s of dead grabs → reopen
                    print("[ZED] repeated grab failures, reopening...")
                    zed.close()
                    time.sleep(0.2)
                    break

def fisheye_thread(idx, key, flip=False):
    # Outer loop reopens the device after a stall. On the Jetson's UVC/V4L2 stack a
    # CPU-starvation burst (e.g. the finish_episode parquet flush + stats at episode
    # end) can wedge a capture handle so every subsequent read() returns ok=False
    # forever. Without a reopen the tile freezes on its last-good frame and never
    # recovers until collect.py is restarted — the "camera cuts out after one episode"
    # symptom. Detecting repeated failures and reconnecting keeps the stream alive.
    while True:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[{key}] Failed to open {idx}, retrying...")
            time.sleep(1)
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # BUFFERSIZE must be >=2. On this V4L2/OpenCV stack a value of 1 kills
        # double-buffering: while userspace holds the single buffer, the driver has
        # nowhere to capture the next frame, so it drops every other one and the
        # stream runs at a flat ~15 fps instead of 30 (measured). 2 restores the full
        # 30 fps while keeping queue latency to ~1 frame — the tight read loop below
        # drains continuously, so the queue never deepens past that.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # Stop the sensor dropping its own frame rate to lengthen exposure in dim
        # light — otherwise a camera facing a darker scene silently runs <30 fps.
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

_zmq_ctx = zmq.Context.instance()

def _sub_socket(topic: str, rcvhwm: int | None = None) -> "zmq.Socket":
    # The Pi publishes with a raw zmq.PUB sending multipart [topic_bytes, json_bytes]
    # (NOT commlink/pickle), so we subscribe with raw zmq and json.loads the payload.
    sock = _zmq_ctx.socket(zmq.SUB)
    # For high-rate streams we only care about the newest frame, so cap the receive
    # queue tightly (rcvhwm): otherwise a SUB socket buffers up to RCVHWM (default
    # 1000) messages and recv() returns the OLDEST first, so any producer>consumer
    # gap becomes an ever-growing multi-second backlog. Leave rcvhwm=None (default
    # queue) for low-rate discrete-event channels where every message must survive.
    if rcvhwm is not None:
        sock.setsockopt(zmq.RCVHWM, rcvhwm)
    sock.connect(f"tcp://{PI_IP}:5558")
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock

def _recv_latest(sock, want_topic: bytes | None = None):
    """Block for one message, then drain everything already queued and return only
    the newest (optionally filtered to an exact topic). This keeps latency bounded to
    one frame regardless of how far behind the consumer falls — we discard staleness
    instead of accumulating it. Returns None if nothing matched this cycle."""
    parts  = sock.recv_multipart()
    latest = parts if (want_topic is None or parts[0] == want_topic) else None
    while True:
        try:
            parts = sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            break
        if want_topic is None or parts[0] == want_topic:
            latest = parts
    return latest

def teleop_thread():
    # High-rate stream: tight queue + drain-to-newest so latency can't accumulate.
    sock = _sub_socket("teleop/episode", rcvhwm=2)
    last_ts = None
    print(f"[teleop] connected to {PI_IP}:5558")
    while True:
        # SUBSCRIBE is a prefix match, so "teleop/episode" also catches
        # "teleop/episode_result" — keep only the newest exact-topic match.
        parts = _recv_latest(sock, b"teleop/episode")
        if parts is None:
            continue
        frame = json.loads(parts[1])
        if frame["timestamp"] != last_ts:
            last_ts = frame["timestamp"]
            _set("teleop", (frame, time.time()))

def result_thread():
    sock = _sub_socket("teleop/episode_result")
    last_ts = None
    print(f"[result] connected to {PI_IP}:5558")
    while True:
        # Discrete event (pass/fail) — must not be dropped, so no draining.
        parts = sock.recv_multipart()
        frame = json.loads(parts[1])
        if frame["timestamp"] != last_ts:
            last_ts = frame["timestamp"]
            _set("episode_result", frame)

def signal_thread():
    # Explicit episode start/stop signal published by vr_teleop.py (raw zmq.PUB,
    # JSON) on port 5558, topic "episode/signal":
    #   {"timestamp": float, "active": bool, "success": optional bool}
    sock = _sub_socket("episode/signal")
    last_ts = None
    print(f"[signal] connected to {PI_IP}:5558")
    while True:
        # Discrete start/stop event — must not be dropped, so no draining.
        parts = sock.recv_multipart()
        frame = json.loads(parts[1])
        if "timestamp" not in frame:
            continue
        if frame["timestamp"] != last_ts:
            last_ts = frame["timestamp"]
            _set("signal", (frame, time.time()))

# ---------------------------------------------------------------------------
# Dataset directory naming
# ---------------------------------------------------------------------------
def _slugify(task: str) -> str:
    """Turn a task string into a filesystem-safe slug.

    e.g. "[Pick and Place v3]" -> "pick_and_place_v3"
         "pick up the orange cube" -> "pick_up_the_orange_cube"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    return slug or "dataset"

def _new_dataset_dir(task: str, root: Path = DATASET_ROOT) -> Path:
    """Fresh dataset directory named <task-slug>_<timestamp> under `root`."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / f"{_slugify(task)}_{stamp}"

def _require_t7_mounted(out_dir: Path) -> None:
    """Refuse to record into T7_MOUNT when the T7 isn't actually mounted.

    With the drive absent /mnt/t7 is just an empty directory on the Orin's eMMC,
    whose root fs has ~1GB free — recording there would silently fill the root
    filesystem rather than fail, and mkdir/open would happily succeed the whole way.
    Only guards paths that really live under the mount: an explicit --data-root or
    --output-dir elsewhere is the caller's business.
    """
    out_dir = out_dir.resolve()
    if out_dir != T7_MOUNT and T7_MOUNT not in out_dir.parents:
        return
    if not T7_MOUNT.is_mount():
        raise SystemExit(
            f"[collect] refusing to record: {out_dir} is under {T7_MOUNT}, but the T7 is\n"
            f"          not mounted — this would fill the Orin's eMMC instead.\n"
            f"          mount it:       sudo mount -o uid=$(id -u),gid=$(id -g) "
            f"UUID={T7_UUID} {T7_MOUNT}\n"
            f"          or record elsewhere:  --data-root <path>"
        )

# ---------------------------------------------------------------------------
# Threaded per-camera H.264 encoder
# ---------------------------------------------------------------------------
_ENC_STOP = object()   # sentinel; distinct from a None frame (missing camera)

class _VideoEncoder:
    """One background H.264 encoder per camera per episode.

    Moves the CPU-heavy cvtColor + encode + mux off the sync loop: the main thread
    just submit()s a frame onto a bounded queue, and this worker does the encoding.
    The four cameras therefore encode *in parallel* (PyAV releases the GIL during
    encode) instead of serially in the sync loop, so a tick costs ~max(single encode)
    rather than the sum.

    submit() BLOCKS when the queue is full — it applies backpressure to the sync loop
    rather than dropping frames, preserving the 1:1 video-frame <-> parquet-row
    alignment that LeRobot requires.
    """
    def __init__(self, path: str, fps: int, maxsize: int = 90):
        self._container = av.open(path, mode="w")
        self._stream          = self._container.add_stream("h264", rate=fps)
        self._stream.pix_fmt  = "yuv420p"
        self._q: queue.Queue  = queue.Queue(maxsize=maxsize)
        # Running per-channel (RGB) image-stat accumulators, updated on a strided
        # subset of frames inside the worker thread (off the sync loop).
        self._seen  = 0                              # frames encoded (for stride)
        self._n     = 0                              # frames sampled into stats
        self._px    = 0                              # pixels summed (for mean/std)
        self._sum   = np.zeros(3, np.float64)
        self._sumsq = np.zeros(3, np.float64)
        self._min   = np.full(3,  np.inf)
        self._max   = np.full(3, -np.inf)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, img):
        self._q.put(img)          # blocking → backpressure, never drops

    def _run(self):
        stream, container = self._stream, self._container
        while True:
            img = self._q.get()
            if img is _ENC_STOP:
                break
            # ZED is BGRA, fisheye is BGR
            if img.ndim == 3 and img.shape[2] == 4:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if stream.width == 0:
                stream.width  = rgb.shape[1]
                stream.height = rgb.shape[0]
            if self._seen % IMG_STATS_STRIDE == 0:
                flat = rgb.reshape(-1, 3).astype(np.float64)
                self._sum   += flat.sum(0)
                self._sumsq += (flat * flat).sum(0)
                self._min    = np.minimum(self._min, flat.min(0))
                self._max    = np.maximum(self._max, flat.max(0))
                self._px    += flat.shape[0]
                self._n     += 1
            self._seen += 1
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        # flush the encoder and finalize the file
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

    def stats(self):
        """Per-channel image stats in LeRobot v2.1 shape [3,1,1], normalized to
        [0,1]. Call only after join(). Returns None if no frames were sampled."""
        if self._n == 0 or self._px == 0:
            return None
        mean = self._sum / self._px
        var  = np.maximum(self._sumsq / self._px - mean ** 2, 0.0)
        std  = np.sqrt(var)
        chan = lambda v: (v / 255.0).reshape(3, 1, 1).tolist()
        return {
            "min":   chan(self._min),
            "max":   chan(self._max),
            "mean":  chan(mean),
            "std":   chan(std),
            "count": [self._n],
        }

    def request_stop(self):
        """Signal end-of-episode (non-blocking). The worker keeps draining and
        finalizing in the background; call join() to wait for the mp4 to land."""
        self._q.put(_ENC_STOP)

    def join(self):
        self._thread.join()

    def close(self):
        """Signal end-of-episode and block until the queue is fully drained,
        encoded, and the mp4 is finalized on disk."""
        self.request_stop()
        self.join()

# ---------------------------------------------------------------------------
# LeRobot v2.1 writer
# ---------------------------------------------------------------------------
def _array_stats(values) -> dict:
    """min/max/mean/std/count for a tabular feature, in LeRobot stats shape.
    `values` is a list of scalars or of equal-length lists; the returned arrays
    have the feature's per-frame shape (length D), count is [num_frames]."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {
        "min":   a.min(0).tolist(),
        "max":   a.max(0).tolist(),
        "mean":  a.mean(0).tolist(),
        "std":   a.std(0).tolist(),
        "count": [a.shape[0]],
    }


# Tabular (non-image) features whose stats come straight from the in-memory rows.
_VECTOR_FEATURES = (
    "observation.state", "action.left_ee", "action.right_ee",
    "action.gripper", "action.base_vel", "action.lift_cmd",
)


class LeRobotWriter:
    def __init__(self, root: Path, task: str):
        self.root = root
        self.meta_dir = root / "meta"
        self.data_dir = root / "data"
        self.vid_dir  = root / "videos"
        for d in (self.meta_dir, self.data_dir, self.vid_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._task_str     = task
        self._total_tasks  = 0
        self.task_index    = self._register_task(task)   # also sets self._total_tasks
        self.episode_index = self._resume_episode_index()
        self._total_frames = self._resume_total_frames()
        self._rows: list[dict]  = []
        self._encoders: dict = {}   # cam_key → _VideoEncoder (one per episode)
        # Black fallback frame per camera + last-good cache, so a momentarily
        # missing camera frame is padded instead of desyncing that mp4 (see below).
        self._black = {cam: np.zeros(shp, np.uint8) for cam, shp in CAM_SHAPES.items()}
        self._last_good: dict = {}

    def _register_task(self, task: str) -> int:
        tasks_file = self.meta_dir / "tasks.jsonl"
        tasks: list[dict] = []
        if tasks_file.exists():
            with open(tasks_file) as f:
                tasks = [json.loads(l) for l in f if l.strip()]
        for t in tasks:
            if t["task"] == task:
                self._total_tasks = len(tasks)
                return t["task_index"]
        idx = len(tasks)
        with open(tasks_file, "a") as f:
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        self._total_tasks = len(tasks) + 1
        print(f"[writer] task {idx}: {task!r}")
        return idx

    def _resume_episode_index(self) -> int:
        ep_file = self.meta_dir / "episodes.jsonl"
        if not ep_file.exists():
            return 0
        with open(ep_file) as f:
            lines = [l for l in f if l.strip()]
        return len(lines)

    def _resume_total_frames(self) -> int:
        ep_file = self.meta_dir / "episodes.jsonl"
        if not ep_file.exists():
            return 0
        total = 0
        with open(ep_file) as f:
            for l in f:
                if l.strip():
                    total += json.loads(l)["length"]
        return total

    def _chunk_name(self) -> str:
        return f"chunk-{self.episode_index // CHUNK_SIZE:03d}"

    def start_episode(self):
        self._rows = []
        self._last_good = {}
        self._open_video_writers()
        print(f"\n{'='*50}")
        print(f"  EPISODE {self.episode_index} STARTED")
        print(f"  task: {self._task_str!r}")
        print(f"{'='*50}\n")

    def _open_video_writers(self):
        chunk  = self._chunk_name()
        ep_tag = f"episode_{self.episode_index:06d}"
        self._encoders = {}
        for cam in (CAM_ZED, CAM_FISH0, CAM_FISH1):
            cam_dir = self.vid_dir / chunk / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            path    = str(cam_dir / f"{ep_tag}.mp4")
            self._encoders[cam] = _VideoEncoder(path, VIDEO_FPS)

    def add_frame(self, frame_index: int, teleop: dict, zed_img, fish0_img, fish1_img):
        # --- tabular row ---
        f = teleop
        self._rows.append({
            "timestamp":           frame_index / VIDEO_FPS,
            "frame_index":         frame_index,
            "episode_index":       self.episode_index,
            "task_index":          self.task_index,
            "observation.state":   f["left_joints"] + f["right_joints"] + [f["lift"]],
            "action.left_ee":      f["left_ee_target"],
            "action.right_ee":     f["right_ee_target"],
            "action.gripper":      [f["left_gripper"], f["right_gripper"]],
            "action.base_vel":     f["base_vel"],
            "action.lift_cmd":     [float(f["lift_cmd"])],
        })

        # --- video frames (encoded off-thread; cvtColor happens in the worker) ---
        # Every camera gets exactly one frame per row. If a camera is momentarily
        # missing (None), repeat its last good frame — or a black frame before any
        # has arrived — so its mp4 never falls out of frame-for-frame lock with the
        # parquet rows (a dropped frame would desync every subsequent index).
        for cam, img in ((CAM_ZED, zed_img), (CAM_FISH0, fish0_img), (CAM_FISH1, fish1_img)):
            if img is None:
                img = self._last_good.get(cam, self._black[cam])
            else:
                self._last_good[cam] = img
            self._encoders[cam].submit(img)

    def finish_episode(self, success: bool | None):
        if not self._rows:
            print("[writer] empty episode — discarded")
            self._close_video_writers()
            return

        # flush video (and collect per-camera image stats before encoders are cleared)
        img_stats = self._close_video_writers()

        L          = len(self._rows)
        index_base = self._total_frames               # global frame index of row 0
        index_col  = list(range(index_base, index_base + L))

        # write parquet
        chunk  = self._chunk_name()
        ep_tag = f"episode_{self.episode_index:06d}"
        pq_dir = self.data_dir / chunk
        pq_dir.mkdir(parents=True, exist_ok=True)

        table = pa.table({
            "timestamp":         pa.array([r["timestamp"]         for r in self._rows], type=pa.float32()),
            "frame_index":       pa.array([r["frame_index"]       for r in self._rows], type=pa.int64()),
            "episode_index":     pa.array([r["episode_index"]     for r in self._rows], type=pa.int64()),
            "index":             pa.array(index_col,                                    type=pa.int64()),
            "task_index":        pa.array([r["task_index"]        for r in self._rows], type=pa.int64()),
            "observation.state": pa.array([r["observation.state"] for r in self._rows], type=pa.list_(pa.float32())),
            "action.left_ee":    pa.array([r["action.left_ee"]    for r in self._rows], type=pa.list_(pa.float32())),
            "action.right_ee":   pa.array([r["action.right_ee"]   for r in self._rows], type=pa.list_(pa.float32())),
            "action.gripper":    pa.array([r["action.gripper"]    for r in self._rows], type=pa.list_(pa.float32())),
            "action.base_vel":   pa.array([r["action.base_vel"]   for r in self._rows], type=pa.list_(pa.float32())),
            "action.lift_cmd":   pa.array([r["action.lift_cmd"]   for r in self._rows], type=pa.list_(pa.float32())),
        })
        pq.write_table(table, pq_dir / f"{ep_tag}.parquet")

        # per-episode stats (meta/episodes_stats.jsonl) — used for training-time
        # normalization. Tabular features come from the rows; image features from
        # the encoder threads (computed as they encoded, off the sync loop).
        stats: dict = {k: _array_stats([r[k] for r in self._rows]) for k in _VECTOR_FEATURES}
        stats["timestamp"]     = _array_stats([r["timestamp"]     for r in self._rows])
        stats["frame_index"]   = _array_stats([r["frame_index"]   for r in self._rows])
        stats["episode_index"] = _array_stats([r["episode_index"] for r in self._rows])
        stats["index"]         = _array_stats(index_col)
        stats["task_index"]    = _array_stats([r["task_index"]    for r in self._rows])
        for cam, st in img_stats.items():
            if st is not None:
                stats[cam] = st

        # append to episodes.jsonl (LeRobot schema: episode_index, tasks, length;
        # `success` is an extra field LeRobot ignores but we keep for filtering).
        result_str = "success" if success is True else "fail" if success is False else "unknown"
        with open(self.meta_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps({
                "episode_index": self.episode_index,
                "tasks":         [self._task_str],
                "length":        L,
                "success":       result_str,
            }) + "\n")
        with open(self.meta_dir / "episodes_stats.jsonl", "a") as f:
            f.write(json.dumps({"episode_index": self.episode_index, "stats": stats}) + "\n")

        tag, sep = {
            "success": ("PASS ✓", "="),
            "fail":    ("FAIL ✗", "x"),
            "unknown": ("????",   "-"),
        }[result_str]
        print(f"\n{sep*50}")
        print(f"  EPISODE {self.episode_index} {tag}  ({L} frames)")
        print(f"{sep*50}\n")

        # advance counters, then rewrite info.json with the new totals
        self._total_frames += L
        self.episode_index += 1
        self._rows = []
        self._write_info()

    def _close_video_writers(self) -> dict:
        # Blocks until every encoder has drained its queue and finalized its mp4,
        # so the parquet/metadata below is only written once the video is on disk.
        # Signal all four to stop first, then join — so the drain+finalize of the
        # four cameras overlaps instead of running one-camera-at-a-time. Returns
        # each camera's image stats (valid only after join()).
        for enc in self._encoders.values():
            enc.request_stop()
        for enc in self._encoders.values():
            enc.join()
        stats = {cam: enc.stats() for cam, enc in self._encoders.items()}
        self._encoders = {}
        return stats

    def _write_info(self):
        total_episodes = self.episode_index
        total_chunks   = (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE

        def vid_feature(h, w, c):
            return {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(VIDEO_FPS),
                    "video.height": h,
                    "video.width": w,
                    "video.channels": c,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        features = {
            "observation.state": {"dtype": "float32", "shape": [15],
                                  "names": ["lj0","lj1","lj2","lj3","lj4","lj5","lj6",
                                            "rj0","rj1","rj2","rj3","rj4","rj5","rj6",
                                            "lift"]},
            "action.left_ee":    {"dtype": "float32", "shape": [7],
                                  "names": ["qw","qx","qy","qz","x","y","z"]},
            "action.right_ee":   {"dtype": "float32", "shape": [7],
                                  "names": ["qw","qx","qy","qz","x","y","z"]},
            "action.gripper":    {"dtype": "float32", "shape": [2],
                                  "names": ["left_gripper","right_gripper"]},
            "action.base_vel":   {"dtype": "float32", "shape": [3],
                                  "names": ["vx","vy","omega"]},
            "action.lift_cmd":   {"dtype": "float32", "shape": [1],
                                  "names": ["lift_cmd"]},
        }
        for cam, (h, w, c) in CAM_SHAPES.items():
            features[cam] = vid_feature(h, w, c)
        features.update({
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        })

        info = {
            "codebase_version": "v2.1",
            "robot_type":       "yor",
            "total_episodes":   total_episodes,
            "total_frames":     self._total_frames,
            "total_tasks":      self._total_tasks,
            "total_videos":     total_episodes * len(CAM_SHAPES),
            "total_chunks":     total_chunks,
            "chunks_size":      CHUNK_SIZE,
            "fps":              VIDEO_FPS,
            "splits":           {"train": f"0:{total_episodes}"},
            "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features":         features,
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

# ---------------------------------------------------------------------------
# Main sync loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOR data collection")
    parser.add_argument("--task", required=True, help='Task description, e.g. "pick up the orange cube"')
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT,
                        help="Root under which a fresh <task-slug>_<timestamp> dataset dir is created "
                             f"each run (default: {DATASET_ROOT}). Point this at a drive with space, "
                             "e.g. /mnt/t7/yor_data.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Explicit dataset dir (resumes if it exists). Overrides --data-root's "
                             "auto-naming; use it to append to a specific existing dataset.")
    cli = parser.parse_args()

    # By default every run gets its own fresh dataset directory (episodes from 0)
    # under --data-root; --output-dir overrides that with an explicit path.
    output_dir = cli.output_dir if cli.output_dir is not None else _new_dataset_dir(cli.task, cli.data_root)
    _require_t7_mounted(output_dir)
    print(f"[collect] dataset dir: {output_dir}")

    for target, args in [
        (zed_thread,      ()),
        (fisheye_thread,  (FISH0_DEV, "fish0", False)),
        (fisheye_thread,  (FISH1_DEV, "fish1", False)),
        (teleop_thread,   ()),
        (result_thread,   ()),
        (signal_thread,   ()),
    ]:
        threading.Thread(target=target, args=args, daemon=True).start()

    writer      = LeRobotWriter(output_dir, cli.task)
    interval    = 1.0 / SYNC_HZ
    in_episode  = False
    frame_index = 0
    last_signal_ts = None

    update_status(task=cli.task, current_episode=writer.episode_index)

    import socket
    ip = socket.gethostbyname(socket.gethostname())
    threading.Thread(
        target=lambda: _app.run(host="0.0.0.0", port=WEB_PORT, threaded=True),
        daemon=True,
    ).start()
    print(f"\n  Camera viewer: http://{ip}:{WEB_PORT}")
    print(f"  Sync loop at {SYNC_HZ} Hz — waiting for sources...\n")

    # Graceful shutdown: SIGINT/SIGTERM just set a flag (they do NOT raise), so the
    # loop exits cleanly at the next tick and any in-progress episode is finalized
    # below — an unfinalized PyAV container is a corrupt mp4, and the parquet/stats
    # for a killed episode would otherwise be lost.
    stop = threading.Event()
    def _on_stop(signum, _frame):
        if not stop.is_set():
            print("\n[collect] shutdown signal — flushing current episode and exiting...")
        stop.set()
    signal.signal(signal.SIGINT,  _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    while not stop.is_set():
        tick_start = time.time()
        s          = _snapshot()

        teleop_active = False
        teleop_data   = None
        if s["teleop"] is not None:
            f, received_at = s["teleop"]
            if (time.time() - received_at) <= TELEOP_STALE_S:
                teleop_active = True
                teleop_data   = f

        # pick up a fresh, unprocessed episode/signal (start or stop)
        sig_frame = None
        if s["signal"] is not None:
            sf, _ = s["signal"]
            if sf["timestamp"] != last_signal_ts:
                last_signal_ts = sf["timestamp"]
                sig_frame = sf

        # --- episode state machine (episode/signal is authoritative) ---
        if not in_episode:
            # START only on an explicit start signal
            if sig_frame is not None and sig_frame.get("active"):
                in_episode  = True
                frame_index = 0
                _set("episode_result", None)   # discard any stale result
                writer.start_episode()
                update_status(in_episode=True, current_episode=writer.episode_index, current_frame=0)

        else:
            result_msg = s["episode_result"]
            ended_ep   = writer.episode_index
            saved      = frame_index
            end        = False
            success    = None

            if sig_frame is not None and not sig_frame.get("active"):
                # explicit stop signal — success may ride along, else fall back
                # to a pass/fail result message, else "unknown"
                end = True
                if "success" in sig_frame:
                    success = sig_frame["success"]
                elif result_msg is not None:
                    success = result_msg["result"] == "success"
            elif result_msg is not None:
                # pass/fail button ended it without a stop signal
                end = True
                success = result_msg["result"] == "success"
            elif frame_index > 0 and not teleop_active:
                # safety fallback — teleop went silent mid-episode
                end = True

            if end:
                in_episode = False
                _set("episode_result", None)
                writer.finish_episode(success=success)
                if saved > 0:
                    tag = "success" if success is True else "fail" if success is False else "unknown"
                    update_status(in_episode=False, current_episode=writer.episode_index,
                                  last_episode=ended_ep, last_frames=saved, last_result=tag)
                else:
                    update_status(in_episode=False, current_episode=writer.episode_index)

        # --- collect & echo ---
        if in_episode and teleop_data is not None:
            writer.add_frame(frame_index, teleop_data, s["zed"], s["fish0"], s["fish1"])
            frame_index += 1
            update_status(current_frame=frame_index)

        def shape(img):
            return f"{img.shape[1]}x{img.shape[0]}" if img is not None else "---"

        if teleop_active:
            f = teleop_data
            status = (
                f"ep={writer.episode_index}  fr={frame_index:04d}"
                f"  L_grip={f['left_gripper']:.2f}  R_grip={f['right_gripper']:.2f}"
                f"  base={[round(v,2) for v in f['base_vel']]}"
                f"  [RStick=PASS / LStick=FAIL]"
            )
        elif s["teleop"] is not None:
            _, received_at = s["teleop"]
            status = f"teleop=idle ({time.time()-received_at:.1f}s ago)"
        else:
            status = "teleop=waiting..."

        # busy = work done this tick (snapshot + state machine + encode). If this
        # regularly exceeds the tick budget (1000/SYNC_HZ ms) during an episode, the
        # sync loop can't keep realtime and the synchronous H.264 encode in
        # add_frame() is the bottleneck (see notes: move it to a writer thread /
        # NVENC). teleop_age = how stale the freshest teleop frame is; with the ZMQ
        # drain fix this should stay small (<~50 ms) instead of climbing.
        busy_ms = (time.time() - tick_start) * 1000
        teleop_age = f"{time.time()-received_at:.2f}s" if s["teleop"] is not None else "--"
        print(
            f"[{time.time():.3f}]"
            f"  busy={busy_ms:5.1f}ms/{1000/SYNC_HZ:.0f}"
            f"  tele_age={teleop_age}"
            f"  zed={shape(s['zed'])}"
            f"  fish0={shape(s['fish0'])}"
            f"  fish1={shape(s['fish1'])}"
            f"  {status}"
        )

        elapsed = time.time() - tick_start
        time.sleep(max(0.0, interval - elapsed))

    # left the loop via a shutdown signal — finalize any open episode
    if in_episode:
        print("[collect] finalizing in-progress episode before exit...")
        writer.finish_episode(success=None)
    print("[collect] clean shutdown")

if __name__ == "__main__":
    main()
