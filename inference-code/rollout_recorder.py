"""RolloutRecorder: persists per-tick state/action/camera-frame data for a single
inference rollout to disk. inference.py's Backend.start() computes a directory per
rollout -- rollout_recordings/<policy>/<task>/<date_ran>/ (per user direction
2026-09-07: grouped by policy then task then when it ran, not a flat pile of uuid
directories) -- and passes it straight through via --recording-dir; this module
just writes into whatever directory it's given. A backend subprocess's entire
lifetime is exactly one rollout (Backend.start() always calls self.stop() first),
so recording spans the whole process with no separate start/stop signal needed.

Shared between inference_server_openpi.py and inference_server_lerobot.py (both
loops, including the RTC producer loop) since this is generic (state, action,
camera bytes) with no backend-specific pipeline logic that would need to diverge
-- unlike build_state(), which each script deliberately keeps as its own copy.

Before this existed, the only persisted record of a rollout was rollout_logs/
rollouts.jsonl (a one-line-per-rollout summary: id/label/task/duration/
annotation) and inference_ui_logs/ (raw stdout text -- rounded-to-3-decimals
state/action printed as log lines, no images at all). Neither is replayable.
This fixes that: full-precision state/action per tick, plus the actual camera
JPEGs, not just a text description of them.

Layout, under the given recording_dir:
  meta.json     -- policy/task/date_ran plus rollout_id, config/checkpoint, family, publish
  observations.jsonl -- every received teleop/episode packet, independently timestamped
                   at receipt and retaining the Pi's source timestamp and full packet
  cameras.jsonl -- every received camera packet with receipt/source timestamps and paths into
                   camera_frames/ (camera acquisition rate, independent of inference)
  commands.jsonl -- every command actually published, including send time, chunk/step index,
                   raw action, converted published action, wire payload, and topic
  inferences.jsonl -- one line per model invocation: tick, timestamp, full-precision
                   state and action_chunk, infer_ms, and the frame filenames below.
                   Delta/rot6D policies additionally store current_left_ee,
                   current_right_ee, commanded_left_ee_chunk, and
                   commanded_right_ee_chunk, all in wxyz_xyz pose layout.
  steps.jsonl   -- compatibility mirror of inferences.jsonl for older plotting tools
  camera_frames/event_<N>_<camera>.jpg -- every raw JPEG received from the camera stream
  recorder_stats.json -- event counts and any queue/write drops, emitted at clean shutdown

All writes run on one bounded background-writer queue. Receiver and command-publisher threads
only enqueue immutable event data, so diagnostic disk I/O does not delay the control loop.

Recording is on whenever a recording_dir is given -- covers dry runs and
--publish runs alike, matching inference.py's own rollout lifecycle exactly (a
recording_dir is minted on every /start regardless of --publish). Passing
recording_dir=None (e.g. a manual, non-UI invocation of a backend script)
disables recording entirely rather than writing to some default location.
"""

import atexit
import json
import logging
import pathlib
import queue
import threading
import time
from collections import Counter

import numpy as np

log = logging.getLogger("rollout_recorder")


class RolloutRecorder:
    def __init__(self, recording_dir: pathlib.Path | str | None, meta: dict, rollout_id: str | None = None):
        self.enabled = recording_dir is not None
        self._files = {}
        self._closing = False
        self._drops = Counter()
        self._ids = Counter()
        self._id_lock = threading.Lock()
        self._enqueue_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=2048)
        self._writer_thread = None
        if not self.enabled:
            return
        self.dir = pathlib.Path(recording_dir)
        self.frames_dir = self.dir / "frames"
        try:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.camera_frames_dir = self.dir / "camera_frames"
            self.camera_frames_dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            (self.dir / "meta.json").write_text(json.dumps({
                **meta,
                "recorder_schema_version": 2,
                "rollout_id": rollout_id,
                "start_time": now,
                "date_ran": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            }, indent=2))
            for stream in ("steps", "inferences", "observations", "cameras", "commands", "events"):
                self._files[stream] = open(self.dir / f"{stream}.jsonl", "a", buffering=1)
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="rollout-recorder-writer"
            )
            self._writer_thread.start()
            atexit.register(self.close)
        except Exception as e:
            # A recording failure should never take down a real rollout -- log loudly and
            # keep going with recording effectively disabled for the rest of this process.
            log.warning("RolloutRecorder: failed to initialize at %s: %s -- recording disabled "
                        "for this run.", self.dir, e)
            self.enabled = False

    def _next_id(self, stream: str) -> int:
        with self._id_lock:
            event_id = self._ids[stream]
            self._ids[stream] += 1
        return event_id

    def _enqueue(self, stream: str, record: dict, blobs: dict[pathlib.Path, bytes] | None = None) -> bool:
        with self._enqueue_lock:
            if not self.enabled or self._closing:
                return False
            try:
                self._queue.put_nowait((stream, record, blobs or {}))
                return True
            except queue.Full:
                self._drops[stream] += 1
                if self._drops[stream] == 1 or self._drops[stream] % 100 == 0:
                    log.warning("RolloutRecorder: writer queue full; dropped %s event (%d total)",
                                stream, self._drops[stream])
                return False

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            stream, record, blobs = item
            try:
                for path, contents in blobs.items():
                    path.write_bytes(contents)
                line = json.dumps(record) + "\n"
                self._files[stream].write(line)
                # steps.jsonl remains the backward-compatible inference stream consumed by
                # existing plotting tools; inferences.jsonl is its explicit event-stream name.
                if stream == "inferences":
                    self._files["steps"].write(line)
            except Exception as e:
                self._drops[f"{stream}_write_error"] += 1
                log.warning("RolloutRecorder: failed writing %s event: %s", stream, e)
            finally:
                self._queue.task_done()

    def record_event(self, kind: str, **fields) -> int | None:
        """Append a free-form, low-rate marker to events.jsonl.

        Used for things that happen BETWEEN ticks and would otherwise be invisible in the
        recording -- above all the live task-instruction changes that inference.py's UI can push
        mid-run (see inference_server_openpi.py's task-control listener). Without these markers a
        recording made across several language queries looks like one undifferentiated run, and
        meta.json's "task" -- captured once at startup -- silently misattributes every tick after
        the first change. Each event carries the wall-clock time, so a tick can be assigned to the
        attempt that was active when it ran by comparing timestamps.

        Deliberately schema-free past `kind`: this is a diagnostic side-channel, not a data
        contract, and it must never be worth failing a rollout over.
        """
        if not self.enabled:
            return None
        event_id = self._next_id("events")
        self._enqueue("events", {
            "event_id": event_id,
            "kind": kind,
            "timestamp": time.time(),
            **fields,
        })
        return event_id

    def record_observation(self, packet: dict, receipt_timestamp: float | None = None) -> int | None:
        """Queue every robot-state packet received from teleop/episode."""
        event_id = self._next_id("observation")
        record = {
            "event_id": event_id,
            "receipt_timestamp": time.time() if receipt_timestamp is None else receipt_timestamp,
            "source_timestamp": packet.get("timestamp"),
            "packet": packet,
        }
        return event_id if self._enqueue("observations", record) else None

    def record_camera(self, packet: dict, receipt_timestamp: float | None = None) -> dict | None:
        """Queue every camera packet and return stable paths that inference events can reference."""
        event_id = self._next_id("camera")
        receipt_timestamp = time.time() if receipt_timestamp is None else receipt_timestamp
        frame_names, blobs = {}, {}
        for camera, contents in packet.get("cameras", {}).items():
            if contents is None:
                continue
            relative = pathlib.Path("camera_frames") / f"event_{event_id:07d}_{camera}.jpg"
            frame_names[camera] = str(relative)
            blobs[self.dir / relative] = contents
        record = {
            "event_id": event_id,
            "receipt_timestamp": receipt_timestamp,
            "source_timestamp": packet.get("timestamp"),
            "frames": frame_names,
        }
        if not self._enqueue("cameras", record, blobs):
            return None
        return record

    def record_command(self, chunk_tick: int, step_index: int, raw_action: np.ndarray,
                       published_action: np.ndarray, payload: dict, topic: str,
                       sent_timestamp: float | None = None) -> None:
        """Record one command immediately after it is published to the robot."""
        self._enqueue("commands", {
            "event_id": self._next_id("command"),
            "sent_timestamp": time.time() if sent_timestamp is None else sent_timestamp,
            "chunk_tick": chunk_tick,
            "step_index": step_index,
            "topic": topic,
            "raw_action": np.asarray(raw_action).tolist(),
            "published_action": np.asarray(published_action).tolist(),
            "payload": payload,
        })

    def record_tick(self, tick: int, state: np.ndarray, action_chunk: np.ndarray, infer_ms: float,
                    cameras: dict | None, *, current_left_ee: np.ndarray | None = None,
                    current_right_ee: np.ndarray | None = None,
                    commanded_left_ee_chunk: np.ndarray | None = None,
                    commanded_right_ee_chunk: np.ndarray | None = None,
                    executed_steps: int | None = None, state_event_id: int | None = None,
                    camera_event_id: int | None = None, frame_names: dict | None = None,
                    state_receipt_timestamp: float | None = None,
                    camera_receipt_timestamp: float | None = None,
                    anchor_left_ee: np.ndarray | None = None,
                    anchor_right_ee: np.ndarray | None = None,
                    task: str | None = None, attempt_id: str | None = None,
                    attempt_index: int | None = None, retrieval: dict | None = None) -> None:
        """cameras: the raw {camera_name: jpeg_bytes} dict straight off the wire (e.g.
        cam_pkt["cameras"]) -- written to disk unmodified, not the decoded numpy arrays.

        The optional EE values are emitted together or omitted together. Current poses and
        commanded chunk poses use [qw, qx, qy, qz, x, y, z]. They are used by delta/rot6D
        policies because their raw action_chunk alone cannot be compared with measured pose.

        task/attempt_id/attempt_index describe the evaluation attempt this tick belongs to. They
        are per-tick rather than per-recording because inference.py's UI can change the language
        query and open a new attempt without restarting the policy, so meta.json's single
        startup "task" is only correct until the first change. See events.jsonl's
        "attempt_boundary" records for the transitions themselves.

        current_left_ee/current_right_ee are always the MEASURED pose this tick. anchor_*_ee
        (optional, same layout, recorded independently of the group above) is the pose the
        predicted deltas were actually added to, which under the default command anchoring is
        the server's own last commanded pose rather than the measured one -- see
        inference_server_openpi.select_delta_anchor(). Recording both keeps the anchor offset
        recoverable from the rollout alone.
        """
        if not self.enabled:
            return
        try:
            blobs = {}
            frame_names = dict(frame_names or {})
            for cam_name, jpeg_bytes in (cameras or {}).items():
                if jpeg_bytes is not None:
                    fname = f"tick_{tick:05d}_{cam_name}.jpg"
                    frame_names[cam_name] = f"frames/{fname}"
                    blobs[self.frames_dir / fname] = jpeg_bytes
            record = {
                "event_id": self._next_id("inference"),
                "tick": tick,
                "timestamp": time.time(),
                "state": np.asarray(state).tolist(),
                "action_chunk": np.asarray(action_chunk).tolist(),
                "executed_steps": len(action_chunk) if executed_steps is None else executed_steps,
                "infer_ms": infer_ms,
                "frames": frame_names,
                "state_event_id": state_event_id,
                "camera_event_id": camera_event_id,
                "state_receipt_timestamp": state_receipt_timestamp,
                "camera_receipt_timestamp": camera_receipt_timestamp,
                "task": task,
                "attempt_id": attempt_id,
                "attempt_index": attempt_index,
            }
            if retrieval is not None:
                # inference_server_openpi.summarize_retrieval: which demo chunk this tick's context came
                # from, its score, the query progress value and timings.
                record["retrieval"] = retrieval
            ee_values = (current_left_ee, current_right_ee,
                         commanded_left_ee_chunk, commanded_right_ee_chunk)
            if any(value is not None for value in ee_values):
                if not all(value is not None for value in ee_values):
                    raise ValueError("EE recording fields must be supplied together")
                left_current = np.asarray(current_left_ee)
                right_current = np.asarray(current_right_ee)
                left_chunk = np.asarray(commanded_left_ee_chunk)
                right_chunk = np.asarray(commanded_right_ee_chunk)
                if left_current.shape != (7,) or right_current.shape != (7,):
                    raise ValueError("Current EE poses must each have shape (7,)")
                expected_chunk_shape = (np.asarray(action_chunk).shape[0], 7)
                if left_chunk.shape != expected_chunk_shape or right_chunk.shape != expected_chunk_shape:
                    raise ValueError(f"Commanded EE chunks must each have shape {expected_chunk_shape}")
                if not all(np.isfinite(value).all()
                           for value in (left_current, right_current, left_chunk, right_chunk)):
                    raise ValueError("EE recording fields must be finite")
                record.update({
                    "current_left_ee": left_current.tolist(),
                    "current_right_ee": right_current.tolist(),
                    "commanded_left_ee_chunk": left_chunk.tolist(),
                    "commanded_right_ee_chunk": right_chunk.tolist(),
                })
            for name, value in (("anchor_left_ee", anchor_left_ee),
                                ("anchor_right_ee", anchor_right_ee)):
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.shape != (7,):
                    raise ValueError(f"{name} must have shape (7,)")
                if not np.isfinite(arr).all():
                    raise ValueError(f"{name} must be finite")
                record[name] = arr.tolist()
            self._enqueue("inferences", record, blobs)
        except Exception as e:
            log.warning("RolloutRecorder.record_tick failed (tick=%d): %s", tick, e)

    def close(self) -> None:
        if not self.enabled or self._closing:
            return
        with self._enqueue_lock:
            self._closing = True
            self._queue.put(None)
        try:
            if self._writer_thread is not None:
                self._writer_thread.join()
            for handle in self._files.values():
                handle.close()
            (self.dir / "recorder_stats.json").write_text(json.dumps({
                "event_counts": dict(self._ids),
                "dropped_events": dict(self._drops),
                "closed_timestamp": time.time(),
            }, indent=2) + "\n")
        except Exception as e:
            log.warning("RolloutRecorder.close failed: %s", e)
