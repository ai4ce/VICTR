"""
pi0.5 YOR bimanual inference server -- lerobot/PyTorch venv. Companion to
inference_server_openpi.py, same architecture and wire protocol, different model backend: this
one loads a native lerobot checkpoint (config.json + model.safetensors + policy_preprocessor.json
+ policy_postprocessor.json, lerobot 0.6.1's PolicyProcessorPipeline format) and runs inference in
plain PyTorch instead of shelling out to openpi/JAX. See PIPELINE_AUDIT.md for the openpi-side
facts this script's wire protocol and gripper-convention rule are inherited from -- those still
apply unchanged. Facts specific to this checkpoint (verified against
checkpoints/icl-pi05-easy-pnp-15000/config.json, train_config.json, and a live smoke test loading
the real weights + processors, 2026-08-14):

  - Checkpoint format: lerobot 0.6.1's processor-pipeline layout, NOT the openpi/JAX "params" dir
    format used by inference_server_openpi.py's checkpoints. Self-contained: config.json,
    model.safetensors, policy_preprocessor.json (+ its normalizer_processor safetensors state),
    policy_postprocessor.json (+ its unnormalizer_processor safetensors state), stats.json,
    train_config.json. No separate `--config <name>` registry lookup needed like openpi -- the
    checkpoint dir alone fully determines model class, shapes, and normalization stats.
  - Registered policy type is "pi0" (config.json's "type" field) -> resolves to lerobot's
    PI0Policy, even though this run was fine-tuned from `lerobot/pi05_base` weights and its
    preprocessor pipeline includes a pi05-specific state-tokenizer step
    (pi05_prepare_state_tokenizer_processor_step). This is expected: PI0Policy is the shared
    modeling code for both pi0 and pi0.5 checkpoints in this lerobot version; the pi05-flavored
    prompt/state tokenization is just one more step in the serialized pipeline, reconstructed
    faithfully by PolicyProcessorPipeline.from_pretrained() regardless of which config class
    "owns" the registry name. Do not "fix" this to PI05Policy -- it would be wrong.
  - observation.state is 15D here (left_joints(7) + right_joints(7) + lift), NOT the 14D
    (lift-dropped) state the openpi checkpoint used -- confirmed in config.json's
    input_features.observation.state.shape and train_config.json's dataset feature list. build_state()
    below reflects this; the state packet from the Pi (teleop/episode topic) already carries a
    "lift" field (see PIPELINE_AUDIT.md's Pi state transport section), so no new data source is
    needed, just one more field pulled out of the same packet.
  - action is 20D per chunk step: left_ee(7) + right_ee(7) + gripper(2) + base_vel(3) + lift_cmd(1)
    -- confirmed via config.json's action_feature_names, which enumerates exactly these 20 names
    in this order. The wire protocol to the Pi (inference/bimanual_ee_action) only carries
    left_ee/right_ee/left_gripper/right_gripper -- joystick_policy_teleop.py on the Pi has no
    consumer for base_vel or lift_cmd (bimanual arm rig, no mobile base, lift is manual/joystick
    controlled). Those 4 extra dims are computed by the model and logged for visibility but not
    published. If the Pi side ever grows a lift_cmd or base_vel consumer, wire them up then --
    don't invent a payload field with no reader today.
  - chunk_size = n_action_steps = 30 (2x the openpi checkpoint's 10-step horizon). Still run with
    manual receding-horizon replanning via --replan-steps (default 5, same as the openpi script)
    rather than lerobot's own PI0Policy.select_action() action-queue mechanism, which would only
    replan once every 30 steps (~1s at 30Hz) -- too slow for the same reasons the openpi script
    replans early. This means calling policy.predict_action_chunk() directly (bypassing
    select_action()) and slicing/publishing/re-predicting ourselves, mirroring
    inference_server_openpi.py's run_inference()/main loop structure exactly.
  - Gripper convention is UNCHANGED from the openpi checkpoint: dataset/model convention is
    0=open/1=closed, hardware wants 0=closed/1=open (see PIPELINE_AUDIT.md's "Gripper convention"
    section and yor_gripper_convention memory) -- invert (1.0 - value) before publishing, exactly
    like inference_server_openpi.py and replay_episode.py already do. Nothing about the lerobot
    checkpoint changes this; it was trained on the same dataset with the same column convention.
  - Image handling: raw camera frames (480x640x3 uint8) are passed straight through --
    PI0Policy._preprocess_images() resizes-with-pad to config.image_resolution (224x224)
    internally and rescales [0,1] -> [-1,1] for SigLIP; policy_preprocessor.json's own
    normalizer_processor step is IDENTITY for VISUAL features (no separate resize/normalize step
    needed before calling the model). Confirmed empirically: a live smoke test round-tripped a
    dummy 480x640x3 frame through prepare_observation_for_inference() -> preprocessor ->
    predict_action_chunk() -> postprocessor with no shape errors and no manual resize step.
  - No JAX-style GPU memory preallocation issue here -- PyTorch/CUDA doesn't eagerly grab a fixed
    fraction of the card like JAX's default, so this script doesn't need
    inference_server_openpi.py's XLA_PYTHON_CLIENT_MEM_FRACTION workaround.
  - Model load + warmup inference took ~50s + ~0.3s respectively in the live smoke test (weight
    loading dominates; PyTorch eager mode has no JAX-style JIT compile pause). No special
    multi-minute warmup handling needed, unlike the openpi script's JIT-driven "~30-60s" warmup
    note -- kept the warmup call anyway for the same reason (first CUDA call can have kernel
    warmup overhead) but it's not the bottleneck here.

Usage:
  cd /home/lairlab/icl_ws && lerobot_env/bin/python inference_server_lerobot.py \\
      --checkpoint checkpoints/icl-pi05-easy-pnp-15000 [--publish]
"""

import argparse
import io
import json
import logging
import math
import pathlib
import signal
import threading
import time

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CAM_TOPIC = "cameras"
STATE_TOPIC = "teleop/episode"
TASK_TOPIC = b"task"
ACTION_TOPIC = b"inference/bimanual_ee_action"
STALE_TIMEOUT = 2.0


def decode_jpeg(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def build_state(pkt: dict) -> np.ndarray:
    """15D bimanual joint state: [left_joints(7), right_joints(7), lift(1)].

    Unlike the openpi checkpoint's 14D state, this checkpoint's observation.state keeps lift --
    see this file's module docstring for the confirmation source.
    """
    left_joints = np.asarray(pkt["left_joints"], dtype=np.float32)
    right_joints = np.asarray(pkt["right_joints"], dtype=np.float32)
    lift = np.asarray([pkt["lift"]], dtype=np.float32)
    return np.concatenate([left_joints, right_joints, lift]).astype(np.float32)


def load_policy(checkpoint_path: pathlib.Path, device_override: str | None = None, rtc_config=None):
    """rtc_config: an RTCConfig instance (see lerobot.policies.rtc), or None to load the policy
    exactly as before. Must be set on policy_cfg BEFORE policy_cls.from_pretrained() -- confirmed
    2026-08-20 by reading PreTrainedPolicy.from_pretrained(): it only re-derives config from disk
    when the caller passes config=None, otherwise it uses the passed-in config object (including
    any mutated fields) to construct the model directly via cls(config, **kwargs). This is how
    RTCConfig reaches the model's own init_rtc_processor() -> RTCProcessor wiring.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    log.info("Loading lerobot policy from %s ...", checkpoint_path)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    if device_override:
        policy_cfg.device = device_override
    if rtc_config is not None:
        policy_cfg.rtc_config = rtc_config
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(str(checkpoint_path), config=policy_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )
    log.info("Policy loaded (type=%s, device=%s, chunk_size=%d, n_action_steps=%d).",
              policy_cfg.type, policy_cfg.device, policy_cfg.chunk_size, policy_cfg.n_action_steps)
    return policy, preprocessor, postprocessor, policy_cfg


def run_inference(policy, preprocessor, postprocessor, zed_img, fish0_img, fish1_img,
                   state15: np.ndarray, task: str, device):
    """One inference call -> (action_chunk[chunk_size, 20], infer_ms).

    Bypasses PI0Policy.select_action()'s internal action queue (which only replans once every
    n_action_steps) in favor of calling predict_action_chunk() directly, so the caller can
    implement its own receding-horizon replanning -- same reasoning as
    inference_server_openpi.py's run_inference(). See module docstring for the 20D action layout.
    """
    import torch
    from lerobot.policies import prepare_observation_for_inference

    observation = {
        "observation.images.zed": zed_img,
        "observation.images.fish0": fish0_img,
        "observation.images.fish1": fish1_img,
        "observation.state": state15,
    }

    t0 = time.monotonic()
    with torch.inference_mode():
        observation = prepare_observation_for_inference(observation, device, task, robot_type="yor")
        observation = preprocessor(observation)
        action_chunk = policy.predict_action_chunk(observation)
        action_chunk = postprocessor(action_chunk)
    infer_ms = (time.monotonic() - t0) * 1000

    action_chunk = action_chunk.squeeze(0).to(torch.float32).cpu().numpy()  # (chunk_size, 20)
    return action_chunk, infer_ms


# -- Live task-instruction control channel (so inference.py's UI can push a new task without
# restarting this process) --

_task_state = {"task": None, "attempt_id": None, "attempt_index": 0,
               "lock": threading.Lock()}
# Set by main() once the recorder exists, so a live task change can be written into the recording
# as an attempt boundary. Stays None when recording is disabled.
_task_recorder = {"rec": None}


def get_current_task() -> str:
    with _task_state["lock"]:
        return _task_state["task"]


def get_current_attempt() -> tuple[str | None, int]:
    """(attempt_id, attempt_index) of the evaluation attempt currently in progress.

    inference.py's UI opens a new attempt every time the language query changes or an outcome is
    annotated, so one loaded policy spans many attempts. Recorded on every tick, which is what
    makes a multi-query recording separable after the fact -- meta.json's single startup "task"
    cannot express it."""
    with _task_state["lock"]:
        return _task_state["attempt_id"], _task_state["attempt_index"]


def set_task_recorder(rec) -> None:
    _task_recorder["rec"] = rec


def _parse_task_payload(payload: bytes) -> tuple[str, str | None, int | None]:
    """-> (task, attempt_id, attempt_index).

    inference.py sends a JSON object {"task", "attempt_id", "attempt_index"}. Older senders (and
    anything hand-rolled with zmq for a quick test) send the bare instruction string, so a
    non-JSON or non-object payload is treated as just the task with no attempt info -- the live
    task swap keeps working either way."""
    raw = payload.decode()
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw, None, None
    if not isinstance(obj, dict) or "task" not in obj:
        return raw, None, None
    return str(obj["task"]), obj.get("attempt_id"), obj.get("attempt_index")


def start_task_control_listener(port: int | None, initial_task: str,
                                initial_attempt_id: str | None = None,
                                initial_attempt_index: int = 0):
    _task_state["task"] = initial_task
    _task_state["attempt_id"] = initial_attempt_id
    _task_state["attempt_index"] = initial_attempt_index
    if not port:
        return

    def _listener():
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, TASK_TOPIC)
        sock.connect(f"tcp://127.0.0.1:{port}")
        log.info("Task-control listener connected to tcp://127.0.0.1:%d (topic=%r)", port, TASK_TOPIC)
        while True:
            try:
                _, payload = sock.recv_multipart()
                new_task, attempt_id, attempt_index = _parse_task_payload(payload)
            except Exception as e:
                log.warning("Task-control listener error: %s", e)
                continue
            with _task_state["lock"]:
                prev_task = _task_state["task"]
                prev_attempt = _task_state["attempt_id"]
                task_changed = new_task != prev_task
                attempt_changed = attempt_id is not None and attempt_id != prev_attempt
                if task_changed:
                    _task_state["task"] = new_task
                if attempt_changed:
                    _task_state["attempt_id"] = attempt_id
                    if attempt_index is not None:
                        _task_state["attempt_index"] = int(attempt_index)
            if task_changed:
                log.info("Task instruction updated: %r -> %r", prev_task, new_task)
            if attempt_changed:
                log.info("Attempt %s started (index %s)", attempt_id, attempt_index)
            if task_changed or attempt_changed:
                rec = _task_recorder["rec"]
                if rec is not None:
                    rec.record_event(
                        "attempt_boundary", task=new_task, previous_task=prev_task,
                        attempt_id=attempt_id, attempt_index=attempt_index,
                        previous_attempt_id=prev_attempt,
                    )

    threading.Thread(target=_listener, daemon=True).start()


# -- Debug frame viewer: composite JPEG of the latest camera frame + predicted action, served over
# HTTP so it's visible without reading the log --

_debug = {"frame": 0, "state": None, "zed": None, "action": None, "infer_ms": None,
          "lock": threading.Lock()}


def update_debug_frame(frame_n, state, cam_pkt, action, infer_ms):
    try:
        with _debug["lock"]:
            _debug["frame"] = frame_n
            _debug["state"] = state.copy()
            _debug["zed"] = cam_pkt["cameras"]["zed"]
            _debug["action"] = None if action is None else action.copy()
            _debug["infer_ms"] = infer_ms
    except Exception as e:
        log.warning("update_debug_frame error: %s", e)


def _make_composite() -> bytes:
    from PIL import ImageDraw
    with _debug["lock"]:
        frame, state, zed, action, infer_ms = (
            _debug["frame"], _debug["state"], _debug["zed"], _debug["action"], _debug["infer_ms"])
    if zed is None:
        img = Image.new("RGB", (640, 100), (30, 30, 30))
        ImageDraw.Draw(img).text((10, 40), "Waiting for first frame...", fill=(200, 200, 200))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    top = Image.open(io.BytesIO(zed)).convert("RGB")
    bar_h = 40
    bar = Image.new("RGB", (top.width, bar_h), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    line = f"Frame #{frame}"
    if state is not None:
        line += f"  state={np.round(state, 2).tolist()}"
    draw.text((8, 4), line, fill=(220, 220, 60))
    if action is not None:
        line2 = f"pred action[0]={np.round(action[0], 3).tolist()}"
        if infer_ms is not None:
            line2 += f"  infer={infer_ms:.0f}ms"
        draw.text((8, 20), line2, fill=(160, 200, 255))
    combined = Image.new("RGB", (top.width, bar_h + top.height))
    combined.paste(bar, (0, 0))
    combined.paste(top, (0, bar_h))
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def start_debug_server(port: int):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                data = _make_composite()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/":
                html = (b"<html><body style='background:#111'><img src='/frame.jpg' "
                        b"id='f' style='max-width:100%'><script>"
                        b"setInterval(()=>{document.getElementById('f').src='/frame.jpg?t='+Date.now()},800)"
                        b"</script></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *_): pass

    try:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info("Frame viewer -> http://localhost:%d", port)
    except OSError as e:
        log.warning("Could not start frame viewer on port %d: %s", port, e)


def _run_rtc_loop(policy, preprocessor, postprocessor, torch_device, get_task,
                   latest_cam_pkt, cam_lock, latest_state_pkt, state_lock, pub_sock, dt, rtc_config,
                   recorder):
    """RTC producer/consumer loop -- replaces the synchronous predict-then-publish loop in main()
    when --rtc is enabled. Adapted from lerobot.rollout.inference.rtc.RTCInferenceEngine (the
    library's own reference implementation for driving predict_action_chunk() with RTC, confirmed
    present in this venv's lerobot==0.6.1), simplified to read from this file's raw commlink/zmq
    camera+state threads instead of lerobot's Robot/build_dataset_frame machinery. This project's
    checkpoint has no enabled RelativeActionsProcessorStep (confirmed via
    checkpoints/icl-pi05-easy-pnp-15000/policy_preprocessor.json, "relative_actions_processor":
    {"enabled": false}), so the reference engine's relative-action re-anchoring branch is dropped
    entirely -- it would be a silent no-op here anyway.

    IMPORTANT: unlike run_inference() above, this function must NOT wrap predict_action_chunk() in
    torch.inference_mode(). RTC's guidance term needs autograd (a torch.enable_grad() block nested
    inside predict_action_chunk's own @torch.no_grad() decorator) to compute a correction gradient
    against the previous chunk's leftover tail. no_grad() is locally reversible by a nested
    enable_grad(); inference_mode() is not -- empirically confirmed 2026-08-20 against this
    checkpoint: wrapping the guided call in inference_mode() raises "element 0 of tensors does not
    require grad and does not have a grad_fn", while no context manager (relying on the method's
    own no_grad decorator, exactly like the upstream RTCInferenceEngine) works and measurably
    applies guidance (~0.03 mean abs action delta vs the unguided baseline in that same test).
    """
    import torch
    from lerobot.policies import prepare_observation_for_inference
    from lerobot.policies.rtc import ActionQueue, LatencyTracker

    action_queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()
    time_per_chunk = dt  # predict_action_chunk's inference_delay/real_delay are in units of
                          # control-loop steps, i.e. steps of length dt -- see ActionQueue.merge().
    frame_state = {"n": 0}

    def _producer():
        while True:
            with cam_lock:
                cam_pkt, cam_t = latest_cam_pkt["pkt"], latest_cam_pkt["t"]
                cam_receipt_timestamp = latest_cam_pkt.get("receipt_timestamp")
                cam_recording = latest_cam_pkt.get("recording")
            with state_lock:
                state_pkt, state_t = latest_state_pkt["pkt"], latest_state_pkt["t"]
                state_receipt_timestamp = latest_state_pkt.get("receipt_timestamp")
                state_event_id = latest_state_pkt.get("event_id")

            if cam_pkt is None or state_pkt is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if (now - cam_t) > STALE_TIMEOUT or (now - state_t) > STALE_TIMEOUT:
                time.sleep(0.05)
                continue

            state15 = build_state(state_pkt)
            cameras = cam_pkt["cameras"]
            zed_img = decode_jpeg(cameras["zed"])
            fish0_img = decode_jpeg(cameras["fish0"])
            fish1_img = decode_jpeg(cameras["fish1"])

            idx_before = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()
            latency = latency_tracker.max()
            delay = math.ceil(latency / time_per_chunk) if latency else 0

            observation = {
                "observation.images.zed": zed_img,
                "observation.images.fish0": fish0_img,
                "observation.images.fish1": fish1_img,
                "observation.state": state15,
            }
            t0 = time.monotonic()
            observation = prepare_observation_for_inference(observation, torch_device, get_task(),
                                                              robot_type="yor")
            observation = preprocessor(observation)
            # Deliberately no torch.inference_mode()/no_grad() here -- see docstring above.
            actions = policy.predict_action_chunk(
                observation, inference_delay=delay, prev_chunk_left_over=prev_actions)
            original = actions.squeeze(0).detach().clone()
            processed = postprocessor(actions).squeeze(0).detach()
            new_latency = time.monotonic() - t0
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)
            action_queue.merge(original, processed, new_delay, idx_before)

            processed_np = processed.to(torch.float32).cpu().numpy()
            _tick_task = get_current_task()
            _tick_attempt_id, _tick_attempt_index = get_current_attempt()
            update_debug_frame(frame_state["n"], state15, cam_pkt, processed_np, new_latency * 1000)
            log.info("[rtc %5d] infer=%.0fms  delay=%d->%d steps (est->actual)  state=%s",
                      frame_state["n"], new_latency * 1000, delay, new_delay, np.round(state15, 3))
            log.info("            pred action[0]=%s", np.round(processed_np[0], 3))
            recorder.record_tick(
                frame_state["n"], state15, processed_np, new_latency * 1000,
                None if cam_recording else cameras,
                executed_steps=rtc_config.execution_horizon, state_event_id=state_event_id,
                camera_event_id=cam_recording["event_id"] if cam_recording else None,
                frame_names=cam_recording["frames"] if cam_recording else None,
                state_receipt_timestamp=state_receipt_timestamp,
                camera_receipt_timestamp=cam_receipt_timestamp,
                # Per-tick, not per-run: the UI can swap the language query and open a new
                # attempt mid-run without restarting this process.
                task=_tick_task, attempt_id=_tick_attempt_id, attempt_index=_tick_attempt_index,
            )
            frame_state["n"] += 1

    threading.Thread(target=_producer, daemon=True, name="rtc-producer").start()
    log.info("RTC producer thread started; publishing at %.1fHz from the action queue.", 1.0 / dt)

    next_t = time.monotonic()
    empty_warned_at = 0.0
    command_index = 0
    try:
        while True:
            action = action_queue.get()
            if action is None:
                now = time.monotonic()
                if now - empty_warned_at > 2.0:
                    log.warning("RTC action queue empty -- waiting for first/next chunk...")
                    empty_warned_at = now
                time.sleep(0.01)
                next_t = time.monotonic()
                continue

            action = action.to(torch.float32).cpu().numpy()
            if pub_sock is not None:
                # dict, serialized only for the wire -- record_command() takes the dict.
                payload = {
                    "timestamp": time.time(),
                    "left_ee": action[0:7].tolist(),
                    "right_ee": action[7:14].tolist(),
                    "left_gripper": 1.0 - float(action[14]),
                    "right_gripper": 1.0 - float(action[15]),
                    "infer_ms": 0.0,  # per-chunk latency is logged by the producer thread instead
                }
                pub_sock.send_multipart([ACTION_TOPIC, json.dumps(payload).encode()])
                published_wire = np.concatenate([
                    action[0:14], [payload["left_gripper"], payload["right_gripper"]]
                ])
                recorder.record_command(-1, command_index, action, published_wire, payload,
                                        ACTION_TOPIC.decode(), time.time())
                command_index += 1

            next_t += dt
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        recorder.close()
        log.info("Stopped (RTC mode). %d chunks predicted.", frame_state["n"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True,
                         help="Path to a self-contained lerobot checkpoint dir (config.json, "
                              "model.safetensors, policy_preprocessor.json, "
                              "policy_postprocessor.json, ...), e.g. "
                              "checkpoints/icl-pi05-easy-pnp-15000.")
    parser.add_argument("--device", default=None,
                         help="Override the checkpoint's own config.json device (e.g. 'cpu' for a "
                              "GPU-less smoke test). Default: use whatever the checkpoint specifies "
                              "(normally 'cuda').")
    parser.add_argument("--task", default="pick up the cube with the left hand and place it on the plate")
    parser.add_argument("--cam-host", default="192.168.123.116")
    parser.add_argument("--cam-port", type=int, default=5550)
    parser.add_argument("--state-host", default="192.168.123.55")
    parser.add_argument("--state-port", type=int, default=5568)
    parser.add_argument("--action-host", default="0.0.0.0")
    parser.add_argument("--action-port", type=int, default=5572)
    parser.add_argument("--publish-hz", type=float, default=30.0)
    parser.add_argument("--replan-steps", type=int, default=5,
                         help="Publish only the first N steps of each freshly-predicted action "
                              "chunk, then immediately replan with fresh camera/state instead of "
                              "continuing to the end of the chunk (receding-horizon execution). "
                              "This checkpoint's own chunk_size/n_action_steps is 30; clamped to "
                              "the chunk length if --replan-steps is larger.")
    parser.add_argument("--task-control-port", type=int, default=5573,
                         help="ZMQ port for live task-instruction updates. 0 disables.")
    parser.add_argument("--rtc", action="store_true",
                         help="Enable Real-Time Chunking (Physical Intelligence's inference-time "
                              "algorithm, https://www.physicalintelligence.company/research/real_time_chunking, "
                              "as implemented in lerobot.policies.rtc, confirmed present in this "
                              "venv's lerobot==0.6.1 and confirmed working against this checkpoint "
                              "2026-08-20). A background thread continuously predicts the next "
                              "chunk while the current one is still publishing, and flow-matching "
                              "guidance blends the new chunk's start against the executing tail of "
                              "the old one instead of pausing at each replan boundary. Replaces the "
                              "--replan-steps receding-horizon scheme entirely (that flag is "
                              "ignored when this is set) -- RTC always operates one published step "
                              "at a time, continuously replanned. openpi checkpoints are NOT "
                              "supported (no RTC implementation in the openpi repo as of this "
                              "checkout, 2026-08-20) -- lerobot checkpoints only.")
    parser.add_argument("--rtc-execution-horizon", type=int, default=10,
                         help="How many trailing steps of the previous chunk RTC's guidance term "
                              "blends the new chunk's start against. Clamped to this checkpoint's "
                              "chunk_size if larger. Docs recommend 8-12.")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0,
                         help="How strongly RTC enforces consistency with the previous chunk's "
                              "unexecuted tail. 10.0 is the documented default for 10-step "
                              "flow-matching policies (pi0/pi0.5/SmolVLA).")
    parser.add_argument("--rtc-prefix-schedule", choices=["linear", "exp", "ones", "zeros"],
                         default="exp",
                         help="How to weight consistency across the overlap region. 'exp' "
                              "(exponential decay) is the lerobot docs' recommended starting "
                              "point; RTCConfig's own dataclass default is 'linear'.")
    parser.add_argument("--viewer-port", type=int, default=8082,
                         help="Different default from inference_server_openpi.py's 8081 so both "
                              "servers can run side by side for comparison.")
    parser.add_argument("--publish", action="store_true",
                         help="Publish predicted actions to the Pi. Without this flag, actions are "
                              "only logged. The Pi's joystick interlock (control loop + policy "
                              "rollout must be physically enabled) is the actual safety boundary.")
    parser.add_argument("--attempt-id", default=None,
                         help="Id of the first evaluation attempt, minted by inference.py before "
                              "this process is spawned. Passed on the command line rather than "
                              "over the task-control channel because a just-bound zmq PUB drops "
                              "messages sent before its subscribers have propagated, which would "
                              "leave every tick of attempt 1 recorded under the default index 0 "
                              "and unseparable from attempt 2.")
    parser.add_argument("--attempt-index", type=int, default=0,
                         help="1-based index of the first attempt (see --attempt-id).")
    parser.add_argument("--rollout-id", default=None,
                         help="inference.py's Backend.start() passes this -- recorded in "
                              "RolloutRecorder's meta.json for correlation with rollout_logs/"
                              "rollouts.jsonl and /annotate, but no longer used to name the "
                              "recording directory itself (see --recording-dir).")
    parser.add_argument("--recording-dir", default=None,
                         help="Directory RolloutRecorder persists full-precision state/action/"
                              "camera-JPEG data per tick into -- see rollout_recorder.py. "
                              "inference.py's Backend.start() computes this as "
                              "rollout_recordings/<policy>/<task>/<date_ran>/. Omit for a manual/ "
                              "non-UI invocation to disable recording entirely.")
    args = parser.parse_args()

    # PyTorch doesn't eagerly preallocate a fixed fraction of the card like JAX does (see module
    # docstring), but on this shared GPU we still want a hard cap rather than letting the
    # allocator grow unbounded -- measured empirically 2026-08-14: the fine-tuned YOR checkpoints
    # peak at a flat 9.07GB (torch.cuda.max_memory_allocated(), steady across repeated inference
    # calls, no growth). 2026-08-27: the pi05_base-derived zero-shot checkpoint (same architecture,
    # untruncated base weights) peaks higher at ~16.9GB. 0.2 of this card's ~97887MiB total is
    # ~19.6GiB, comfortable headroom over both measured peaks while staying well clear of the
    # ~75GB+ a JAX process on the same card could otherwise be left to grab. Must be set before any
    # CUDA allocation, i.e. before load_policy().
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.2)

    rtc_config = None
    if args.rtc:
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc import RTCConfig
        schedule_map = {
            "linear": RTCAttentionSchedule.LINEAR, "exp": RTCAttentionSchedule.EXP,
            "ones": RTCAttentionSchedule.ONES, "zeros": RTCAttentionSchedule.ZEROS,
        }
        rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=schedule_map[args.rtc_prefix_schedule],
        )
        if args.replan_steps != 5:  # 5 is the argparse default -- only warn on an explicit override
            log.warning("--rtc is set: ignoring --replan-steps=%d (RTC always replans "
                        "continuously, one published step at a time).", args.replan_steps)

    policy, preprocessor, postprocessor, policy_cfg = load_policy(args.checkpoint, args.device, rtc_config)
    device = policy_cfg.device
    torch_device = torch.device(device)

    from rollout_recorder import RolloutRecorder
    recorder = RolloutRecorder(args.recording_dir, rollout_id=args.rollout_id, meta={
        "family": "lerobot", "checkpoint": str(args.checkpoint), "task": args.task,
        "publish": args.publish, "rtc": args.rtc, "publish_hz": args.publish_hz,
        "replan_steps": args.replan_steps,
    })

    def _stop_for_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_for_signal)

    if rtc_config is not None:
        if rtc_config.execution_horizon > policy_cfg.chunk_size:
            log.warning("--rtc-execution-horizon=%d > this checkpoint's chunk_size=%d; clamping.",
                        rtc_config.execution_horizon, policy_cfg.chunk_size)
            rtc_config.execution_horizon = policy_cfg.chunk_size
        if not (getattr(policy, "supports_rtc", None) and policy.supports_rtc()):
            raise RuntimeError(
                f"--rtc was requested but policy.supports_rtc() is False for checkpoint type "
                f"{policy_cfg.type!r} -- this policy class doesn't implement RTC in this lerobot "
                f"version. Confirmed working for this project's pi0.5 checkpoints; if you're "
                f"loading a different checkpoint type, drop --rtc.")
        log.info("RTC enabled: execution_horizon=%d max_guidance_weight=%.1f schedule=%s",
                  rtc_config.execution_horizon, rtc_config.max_guidance_weight, args.rtc_prefix_schedule)

    start_debug_server(args.viewer_port)
    # Before the listener starts, so the very first live task change is recorded too.
    set_task_recorder(recorder)
    start_task_control_listener(args.task_control_port or None, args.task,
                                args.attempt_id, args.attempt_index)

    log.info("Running warmup inference...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_state = np.zeros(15, dtype=np.float32)
    t0 = time.monotonic()
    run_inference(policy, preprocessor, postprocessor, dummy_img, dummy_img, dummy_img,
                  dummy_state, args.task, torch_device)
    log.info("Warmup done in %.1fs.", time.monotonic() - t0)

    import zmq

    latest_cam_pkt = {"pkt": None, "t": 0.0, "receipt_timestamp": None, "recording": None}
    cam_lock = threading.Lock()

    def _cam_reader():
        from commlink import Subscriber
        sub = Subscriber(args.cam_host, port=args.cam_port, topics=[CAM_TOPIC], buffer=True, queue_size=2)
        log.info("Camera SUB -> tcp://%s:%d topic=%r", args.cam_host, args.cam_port, CAM_TOPIC)
        while True:
            pkt = sub.get(CAM_TOPIC)  # blocks until the next frame
            receipt_timestamp = time.time()
            recording = recorder.record_camera(pkt, receipt_timestamp)
            with cam_lock:
                latest_cam_pkt["pkt"] = pkt
                latest_cam_pkt["t"] = time.monotonic()
                latest_cam_pkt["receipt_timestamp"] = receipt_timestamp
                latest_cam_pkt["recording"] = recording

    def _cam_watchdog():
        while True:
            time.sleep(2)
            with cam_lock:
                t = latest_cam_pkt["t"]
            age = None if t == 0.0 else time.monotonic() - t
            if age is None or age > 2.0:
                log.warning("Camera: no frames in >%.0fs -- is camera_proxy.py running on %s:%d?",
                            age or 0.0, args.cam_host, args.cam_port)

    threading.Thread(target=_cam_reader, daemon=True).start()
    threading.Thread(target=_cam_watchdog, daemon=True).start()

    latest_state_pkt = {"pkt": None, "t": 0.0, "receipt_timestamp": None, "event_id": None}
    state_lock = threading.Lock()

    def _state_reader():
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        sock.connect(f"tcp://{args.state_host}:{args.state_port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        log.info("State SUB -> tcp://%s:%d topic=%r", args.state_host, args.state_port, STATE_TOPIC)
        while True:
            ready = dict(poller.poll(timeout=2000))
            if sock not in ready:
                continue
            received = []
            try:
                received.append(sock.recv_multipart(flags=zmq.NOBLOCK))
                while True:  # drain to the latest
                    try:
                        received.append(sock.recv_multipart(flags=zmq.NOBLOCK))
                    except zmq.Again:
                        break
            except zmq.Again:
                continue
            for frames in received:
                try:
                    if len(frames) < 2 or frames[0] != STATE_TOPIC.encode():
                        continue
                    pkt = json.loads(frames[1])
                    receipt_timestamp = time.time()
                    event_id = recorder.record_observation(pkt, receipt_timestamp)
                    with state_lock:
                        latest_state_pkt["pkt"] = pkt
                        latest_state_pkt["t"] = time.monotonic()
                        latest_state_pkt["receipt_timestamp"] = receipt_timestamp
                        latest_state_pkt["event_id"] = event_id
                except Exception as e:
                    log.warning("State parse error: %s", e)

    threading.Thread(target=_state_reader, daemon=True).start()

    pub_sock = None
    if args.publish:
        pub_sock = zmq.Context.instance().socket(zmq.PUB)
        pub_sock.bind(f"tcp://{args.action_host}:{args.action_port}")
        time.sleep(0.1)
        log.info("PUBLISHING: predicted EE-space actions -> inference/bimanual_ee_action (%s:%d). "
                  "Gripper is inverted to hardware convention (0=closed/1=open); base_vel/lift_cmd "
                  "are computed but not published (no Pi-side consumer); no other filtering is "
                  "applied.", args.action_host, args.action_port)
    else:
        log.info("DRY RUN: predicted actions will be logged only. Pass --publish to drive the robot.")

    dt = 1.0 / args.publish_hz
    frame_count = 0
    infer_times: list[float] = []

    if rtc_config is not None:
        _run_rtc_loop(policy, preprocessor, postprocessor, torch_device, get_current_task,
                      latest_cam_pkt, cam_lock, latest_state_pkt, state_lock, pub_sock, dt, rtc_config,
                      recorder)
        return

    try:
        while True:
            with cam_lock:
                cam_pkt, cam_t = latest_cam_pkt["pkt"], latest_cam_pkt["t"]
                cam_receipt_timestamp = latest_cam_pkt["receipt_timestamp"]
                cam_recording = latest_cam_pkt["recording"]
            with state_lock:
                state_pkt, state_t = latest_state_pkt["pkt"], latest_state_pkt["t"]
                state_receipt_timestamp = latest_state_pkt["receipt_timestamp"]
                state_event_id = latest_state_pkt["event_id"]

            if cam_pkt is None or state_pkt is None:
                log.warning("Waiting for camera/state data...")
                time.sleep(0.1)
                continue

            now = time.monotonic()
            if (now - cam_t) > STALE_TIMEOUT or (now - state_t) > STALE_TIMEOUT:
                log.warning("Stream stale (cam=%.1fs ago, state=%.1fs ago) -- pausing.",
                            now - cam_t, now - state_t)
                time.sleep(0.1)
                continue

            state15 = build_state(state_pkt)

            cameras = cam_pkt["cameras"]
            zed_img = decode_jpeg(cameras["zed"])
            fish0_img = decode_jpeg(cameras["fish0"])
            fish1_img = decode_jpeg(cameras["fish1"])

            # One snapshot per tick, so the instruction fed to the model and the instruction
            # recorded against the tick can never disagree -- the UI can change it at any moment.
            _tick_task = get_current_task()
            _tick_attempt_id, _tick_attempt_index = get_current_attempt()

            t0 = time.monotonic()
            action_chunk, model_infer_ms = run_inference(
                policy, preprocessor, postprocessor, zed_img, fish0_img, fish1_img, state15,
                _tick_task, torch_device)
            infer_ms = (time.monotonic() - t0) * 1000
            infer_times.append(infer_ms)

            # Receding-horizon execution: only ever publish the first --replan-steps steps of a
            # freshly-predicted chunk, then immediately replan with fresh camera/state instead of
            # continuing to the end of the chunk.
            steps_to_run = action_chunk[: args.replan_steps]

            update_debug_frame(frame_count, state15, cam_pkt, action_chunk, infer_ms)
            log.info("[%5d] infer=%.0fms (model=%.0fms)  state=%s", frame_count, infer_ms,
                      model_infer_ms, np.round(state15, 3))
            log.info("         pred action[0]=%s  action[-1]=%s  (running %d/%d steps)",
                      np.round(action_chunk[0], 3), np.round(action_chunk[-1], 3),
                      len(steps_to_run), len(action_chunk))
            # Full-precision state/action + the actual camera JPEGs for this tick -- see
            # rollout_recorder.py.
            recorder.record_tick(
                frame_count, state15, action_chunk, infer_ms,
                None if cam_recording else cameras,
                executed_steps=len(steps_to_run), state_event_id=state_event_id,
                camera_event_id=cam_recording["event_id"] if cam_recording else None,
                frame_names=cam_recording["frames"] if cam_recording else None,
                state_receipt_timestamp=state_receipt_timestamp,
                camera_receipt_timestamp=cam_receipt_timestamp,
                # Per-tick, not per-run: the UI can swap the language query and open a new
                # attempt mid-run without restarting this process.
                task=_tick_task, attempt_id=_tick_attempt_id, attempt_index=_tick_attempt_index,
            )

            if pub_sock is not None:
                t_chunk = time.monotonic()
                for i, action in enumerate(steps_to_run):
                    # Only transform applied: invert gripper to hardware convention
                    # (0=closed, 1=open). base_vel(16:19)/lift_cmd(19:20) are computed but have no
                    # wire-protocol field / Pi-side consumer today, so they're dropped here, not
                    # published as unused payload keys.
                    # dict, serialized only for the wire -- record_command() takes the dict.
                    payload = {
                        "timestamp": time.time(),
                        "left_ee": action[0:7].tolist(),
                        "right_ee": action[7:14].tolist(),
                        "left_gripper": 1.0 - float(action[14]),
                        "right_gripper": 1.0 - float(action[15]),
                        "infer_ms": infer_ms,
                    }
                    pub_sock.send_multipart([ACTION_TOPIC, json.dumps(payload).encode()])
                    published_wire = np.concatenate([
                        action[0:14], [payload["left_gripper"], payload["right_gripper"]]
                    ])
                    recorder.record_command(frame_count, i, action, published_wire, payload,
                                            ACTION_TOPIC.decode(), time.time())
                    next_t = t_chunk + (i + 1) * dt
                    slack = next_t - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
            else:
                time.sleep(0.03)

            frame_count += 1

    except KeyboardInterrupt:
        recorder.close()
        if infer_times:
            log.info("Stopped. %d inferences, mean=%.0fms, min=%.0fms, max=%.0fms",
                      len(infer_times), np.mean(infer_times), np.min(infer_times), np.max(infer_times))
        else:
            log.info("Stopped after %d frames (no inference ran).", frame_count)


if __name__ == "__main__":
    main()
