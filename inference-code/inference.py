"""
YOR inference UI -- orchestrates inference_server_openpi.py (Start/Stop/live task push/rollout
annotation) and serves a small web UI with a live camera preview. Rebuilt from scratch 2026-08-07
alongside inference_server_openpi.py; simplified to openpi/pi0.5 only (see PIPELINE_AUDIT.md for
why the other backends were dropped from scope).

Run:
  cd /home/lairlab/icl_ws && .venv/bin/python inference.py --ui-port 8090
"""

import argparse
import datetime
import io
import json
import logging
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

from PIL import Image, ImageDraw

import experiments
import onion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OPENPI_PYTHON = PROJECT_ROOT / "openpi" / ".venv" / "bin" / "python"
OPENPI_SCRIPT = PROJECT_ROOT / "inference_server_openpi.py"
LEROBOT_PYTHON = PROJECT_ROOT / "lerobot_env" / "bin" / "python"
LEROBOT_SCRIPT = PROJECT_ROOT / "inference_server_lerobot.py"
LOG_DIR = PROJECT_ROOT / "inference_ui_logs"
ROLLOUT_LOG_PATH = PROJECT_ROOT / "rollout_logs" / "rollouts.jsonl"
ROLLOUT_RECORDINGS_DIR = PROJECT_ROOT / "rollout_recordings"
DEFAULT_TASK = "pick up the cube with the left hand and place it on the plate"


def _slug(text: str) -> str:
    """Filesystem-safe slug -- same alnum+underscore convention task_slug() uses
    elsewhere in this codebase (yor_retrieval.py, build_victr_pool.py), reused here
    so directory names stay consistent with how tasks are already identified
    throughout this project."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:80]


def _policy_slug(family: str, config_name: str | None, checkpoint_path: str) -> str:
    """A stable, human-readable directory name identifying a specific checkpoint --
    NOT the step-only basename (collides across different runs at the same step
    number), and not the full free-text UI label (can be user-overridden, less
    predictable). openpi checkpoints live at .../checkpoints/<config>/<run_dir>/<step>
    (see inference.py's _discover_openpi_checkpoints) -- <run_dir>_<step> is already
    unique and matches how this project names checkpoint downloads on disk.
    lerobot checkpoints are self-contained dirs -- just use that dir's own name."""
    ckpt = pathlib.Path(checkpoint_path)
    if family == "openpi":
        run_dir, step = ckpt.parent.name, ckpt.name
        return _slug(f"{run_dir}_{step}")
    return _slug(ckpt.name)


def _lan_ip() -> str:
    """Best-effort real LAN IP for the startup log (doesn't actually send anything)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.123.1", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# -- Checkpoint discovery --
# openpi/JAX:  <PROJECT_ROOT>/openpi/checkpoints/<config_name>/<run_name>/<step>/params
# lerobot/PyTorch: <PROJECT_ROOT>/checkpoints/<run_name>/ (self-contained: config.json +
#   model.safetensors directly in the dir, no --config registry lookup needed -- see
#   inference_server_lerobot.py's module docstring)

# config_name -> reason, for openpi checkpoints that load/discover fine but are known broken
# at inference time for a reason discovery itself can't detect (e.g. a tokenizer mismatch that
# only shows up as a runtime decode failure). Disables the UI dropdown option entirely rather
# than leaving it silently selectable -- see this project's history for the investigation.
_OPENPI_BROKEN_CONFIGS: dict[str, str] = {
    "yor_icl_pi0_fast_easy_pnp_v2_sanity15k_annealing": (
        "FAST tokenizer mismatch (2026-08-30, retested 2026-09-01): the model's own emitted "
        "action tokens don't decode -- every decode attempt produces a garbage token count, "
        "not a multiple of action_dim, falling back to a constant action every time. Tried "
        "openpi's stock tokenizer, a from-scratch fit on the real training data, AND (2026-09-01) "
        "lair-nyu/yor-icl-pi05-easy-pnp-v2-fast-tokenizer -- a real tokenizer artifact published "
        "and explicitly named for this exact checkpoint -- and all three fail identically on a "
        "real dry-run test (live camera frames, 5 independent calls, same reshape error and same "
        "constant fallback action every time). Since even this checkpoint's own real, "
        "purpose-published tokenizer doesn't fix decoding, the problem is not just 'wrong "
        "tokenizer file' -- something else about this checkpoint's generation is broken, not yet "
        "diagnosed."
    ),
}

# config_name -> True for openpi checkpoints that run openpi.policies.yor_retrieval.
# RetrievalContextInputs (VICTR context-chunk conditioning) -- these are the only
# checkpoints whose backend server actually populates get_last_retrieval(), so the
# retrieved-context panel (see the frontend's contextRow/contextFrame) is only shown
# for them.
# config name -> the retrieval metric that config retrieves with when the UI sends no override
# (openpi model.retrieval_metric, or the data config's own override). Membership = VICTR-capable.
# The metric decides whether a progress estimator is meaningful: vision-only retrieval never
# builds one, so the UI offers "none" there instead of a Robometer/VFE choice that does nothing.
_OPENPI_VICTR_CONFIGS: dict[str, str] = {
    "yor_icl_fast_victr_vision_expanded": "vision",
    "yor_icl_victr_vision_expanded": "vision",
    "yor_icl_victr_value_expanded": "value",
    "yor_icl_victr_vision_value_expanded": "vision_value",
    "yor_icl_victr_vision_expanded_serve": "vision",
    "yor_icl_victr_value_expanded_serve": "value",
    "yor_icl_victr_vision_value_expanded_serve": "vision_value",
    "yor_icl_pi05_expanded_with_context": "vision",  # experiments.py EXP B2: base pi0.5 given retrieved context
    # ctx3cam VICTR arms (canonical space, 3-camera context, 40k+5k).
    "yor_icl_victr_vision_canonical_ctx3cam_serve": "vision",
    "yor_icl_victr_value_canonical_ctx3cam_serve": "value",
    "yor_icl_victr_vision_value_canonical_ctx3cam_serve": "vision_value",
    # EXP B2: warmstart baseline pi0.5 with a vision-retrieved demonstration injected.
    "yor_icl_pi05_canonical_warmstart_baseline_with_context": "vision",
    # RICL-pi0.5 (vision retrieval + action interpolation toward the top-1 neighbor).
    "yor_icl_victr_vision_interp_canonical_ctx3cam_serve": "vision",
}

# openpi configs with RICL action interpolation (use_continuous_action_interpolation): the only ones
# that accept --interp-lamda, so the UI shows that field for them alone.
_OPENPI_INTERP_CONFIGS = {"yor_icl_victr_vision_interp_canonical_ctx3cam_serve"}

# Run-dir aliases under openpi/checkpoints/<config>/ (symlinks to an existing checkpoint) that start
# an experiment arm instead of the plain checkpoint, so a commonly-run arm gets its own dropdown entry
# and its runs still record as that arm (same rollout_recordings folder and meta).
_OPENPI_RUN_ARM_ALIASES = {"victr_value_dino_retrieval": "A1"}


def _date_added(path: pathlib.Path) -> str:
    """Best-effort 'when was this checkpoint added' signal, for grouping the UI's
    checkpoint dropdown by date. Resolves symlinks first -- openpi entries are
    discovered via a symlink tree (see module docstring above) whose own mtime can be
    much later than the real checkpoint's (e.g. rebuilt during a later cleanup pass),
    so the REAL checkpoint directory's mtime (set once, at download time, and not
    touched again) is the meaningful signal, not the symlink's. Falls back to the
    symlink's own mtime if the resolved target is unreadable for any reason (broken
    symlink, permissions) rather than raising -- discovery should never fail outright
    over a missing timestamp."""
    try:
        return datetime.date.fromtimestamp(path.resolve().stat().st_mtime).isoformat()
    except OSError:
        try:
            return datetime.date.fromtimestamp(path.lstat().st_mtime).isoformat()
        except OSError:
            return "unknown"


def _discover_openpi_checkpoints() -> list[dict]:
    root = PROJECT_ROOT / "openpi" / "checkpoints"
    entries = []
    if not root.exists():
        return entries
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        broken_reason = _OPENPI_BROKEN_CONFIGS.get(config_name)
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for step_dir in sorted(run_dir.iterdir(),
                                    key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name)):
                if not (step_dir.is_dir() and step_dir.name.isdigit()):
                    continue
                if not (step_dir / "params").exists():
                    continue
                alias_arm = _OPENPI_RUN_ARM_ALIASES.get(run_dir.name)
                entries.append({
                    "family": "openpi",
                    "label": f"[pi0.5] {config_name} / {run_dir.name} / {step_dir.name}",
                    "path": str(step_dir),
                    "config_name": config_name,
                    "runnable": broken_reason is None,
                    "reason": broken_reason,
                    "rtc_capable": False,  # openpi RTC support was removed 2026-08-28 (safety incidents
                                            # during live testing) -- no openpi Pi0RtcConfig exists anymore.
                    "victr_capable": config_name in _OPENPI_VICTR_CONFIGS,
                    "victr_metric": (experiments.find_arm(alias_arm)[2]["retrieval_metric"] if alias_arm
                                     else _OPENPI_VICTR_CONFIGS.get(config_name)),
                    "interp_capable": config_name in _OPENPI_INTERP_CONFIGS,
                    "date_added": _date_added(step_dir),
                })
    return entries


def _discover_lerobot_checkpoints() -> list[dict]:
    root = PROJECT_ROOT / "checkpoints"
    entries = []
    if not root.exists():
        return entries
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not ((run_dir / "config.json").exists() and (run_dir / "model.safetensors").exists()):
            continue  # not a self-contained lerobot checkpoint (e.g. an openpi-style tree)
        entries.append({
            "family": "lerobot",
            "label": f"[lerobot pi0.5] {run_dir.name}",
            "path": str(run_dir),
            "config_name": None,
            "runnable": True,
            "reason": None,
            "rtc_capable": True,  # inference_server_lerobot.py's --rtc works for any lerobot checkpoint
            "victr_capable": False,  # VICTR retrieval-context conditioning is openpi-only
            "date_added": _date_added(run_dir),
        })
    return entries


def discover_checkpoints() -> list[dict]:
    return _discover_openpi_checkpoints() + _discover_lerobot_checkpoints()


# -- Camera preview (decoupled from whichever backend subprocess is running) --

_frame_state = {"frame": 0, "zed": None, "fish0": None, "fish1": None, "lock": threading.Lock()}


def _camera_reader_thread(cam_host: str, cam_port: int):
    # camera_proxy.py publishes via commlink.Publisher (ROUTER/DEALER, custom protocol), not
    # plain PUB/SUB -- see PIPELINE_AUDIT.md. Must use commlink.Subscriber.
    from commlink import Subscriber

    sub = Subscriber(cam_host, port=cam_port, topics=["cameras"], buffer=True, queue_size=2)
    log.info("Camera preview SUB -> tcp://%s:%d topic='cameras'", cam_host, cam_port)
    frame_n = 0
    while True:
        try:
            pkt = sub.get("cameras")
            cameras = pkt["cameras"]
            frame_n += 1
            with _frame_state["lock"]:
                _frame_state["frame"] = frame_n
                _frame_state["zed"] = cameras["zed"]
                _frame_state["fish0"] = cameras["fish0"]
                _frame_state["fish1"] = cameras.get("fish1")
        except Exception as e:
            log.warning("Camera preview decode error: %s", e)
            time.sleep(0.5)


def make_composite_jpeg(policy_view: bool = False) -> bytes:
    """The live camera preview. Layout comes from onion.compose(), which the dataset ghost also
    uses, so the two layers register exactly when the onion overlay stacks them.

    policy_view=True renders the ZED at the geometry the policy is actually fed (squashed to the
    dataset's 640x480 by inference_server_openpi.match_dataset_zed_geometry) rather than its raw
    1280x720. Aligning the robot against a dataset frame only means anything in that space."""
    try:
        with _frame_state["lock"]:
            frame_n = _frame_state["frame"]
            zed = _frame_state["zed"]
            fish0 = _frame_state["fish0"]
            fish1 = _frame_state["fish1"]

        if zed is None:
            img = Image.new("RGB", (640, 100), (30, 30, 30))
            ImageDraw.Draw(img).text((10, 40), "Waiting for first camera frame...", fill=(200, 200, 200))
            buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

        top = Image.open(io.BytesIO(zed)).convert("RGB")
        if policy_view:
            top = onion.to_policy_view(top)
        left = Image.open(io.BytesIO(fish0)).convert("RGB") if fish0 is not None else None
        right = Image.open(io.BytesIO(fish1)).convert("RGB") if fish1 is not None else None

        combined = onion.compose(top, left, right, bar_text=f"frame #{frame_n}", labels=True)
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        img = Image.new("RGB", (640, 80), (80, 20, 20))
        ImageDraw.Draw(img).text((10, 30), f"Error: {e}", fill=(255, 100, 100))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()


# -- Rollout log (append-only JSONL, one line per Start->Stop session) --

# Outcome vocabulary for an attempt. Ordered as the UI renders them, worst-to-best reading
# left-to-right is deliberately NOT used -- "success" sits first because it is the common case.
#   success         -- completed the task
#   partial_success -- got materially through it (e.g. grasped and lifted) then failed before
#                      finishing. The distinction that matters for reporting: the policy showed
#                      the right behavior but did not close it out. For tasks with a plan in
#                      subtask_plans.json the verdict also records HOW FAR it got -- see
#                      subtask_plan_for() and Backend.annotate().
#   failure         -- failed the task entirely
#   skip            -- do not count this attempt (setup error, operator mistake, robot fault)
ANNOTATION_LABELS = ("success", "partial_success", "failure", "skip")

# Attempts shorter than this are discarded instead of queued for annotation -- see
# Backend._close_attempt_locked.
MIN_ATTEMPT_S = 3.0

# Per-task ordered subtask lists (from the icl-sarm-annotations VLM decomposition), used to turn
# a bare partial_success into "completed subtasks 1..k, then failed". Read with an mtime cache so
# a renamed subtask applies on the next poll, without restarting the UI.
SUBTASK_PLANS_PATH = PROJECT_ROOT / "subtask_plans.json"
_subtask_plans_cache = {"mtime": None, "index": {}}


def _norm_task(task: str | None) -> str:
    return " ".join((task or "").lower().replace(".", " ").split())


def subtask_plan_for(task: str | None) -> dict | None:
    """The plan whose task or aliases match this query, or None.

    Exact match after lowercasing and collapsing whitespace -- deliberately not fuzzy: a near-miss
    that silently attached the wrong task's subtasks would put confident, wrong labels in the log.
    An unmatched query just gets the plain Partial success button."""
    try:
        mtime = SUBTASK_PLANS_PATH.stat().st_mtime
    except OSError:
        return None
    if mtime != _subtask_plans_cache["mtime"]:
        index = {}
        try:
            for plan in json.loads(SUBTASK_PLANS_PATH.read_text()).get("tasks", []):
                if len(plan.get("subtasks") or []) < 2:
                    continue  # one step leaves nothing between failure and success
                for name in [plan["task"], *plan.get("aliases", [])]:
                    index[_norm_task(name)] = plan
        except Exception as e:
            log.warning("Could not load %s: %s -- subtask partials disabled", SUBTASK_PLANS_PATH, e)
            index = {}
        _subtask_plans_cache.update(mtime=mtime, index=index)
    plan = _subtask_plans_cache["index"].get(_norm_task(task))
    if plan is None:
        return None
    return {"task": plan["task"], "domain": plan.get("domain"), "subtasks": list(plan["subtasks"])}


def _append_rollout(entry: dict):
    ROLLOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROLLOUT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def experiments_view() -> dict:
    """experiments.py registry for the UI, with each arm's checkpoint availability and its tally
    of annotated attempts (from the rollout log)."""
    tallies: dict[str, dict[str, int]] = {}
    if ROLLOUT_LOG_PATH.exists():
        for line in ROLLOUT_LOG_PATH.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            arm = (entry.get("experiment") or {}).get("arm")
            if arm:
                counts = tallies.setdefault(arm, dict.fromkeys(ANNOTATION_LABELS, 0))
                if entry.get("annotation") in counts:
                    counts[entry["annotation"]] += 1
    out = {}
    for exp_id, exp in experiments.EXPERIMENTS.items():
        arms = {}
        for arm_id, arm in exp["arms"].items():
            ckpt = PROJECT_ROOT / arm["checkpoint"]
            arms[arm_id] = {**arm, "label": experiments.arm_label(arm_id),
                            "available": (ckpt / "params").is_dir(),
                            "tally": tallies.get(arm_id, dict.fromkeys(ANNOTATION_LABELS, 0))}
        out[exp_id] = {**exp, "arms": arms}
    return out


def _recent_rollouts(n: int = 10) -> list[dict]:
    if not ROLLOUT_LOG_PATH.exists():
        return []
    try:
        lines = ROLLOUT_LOG_PATH.read_text().splitlines()[-n:]
        return [json.loads(line) for line in reversed(lines)]
    except Exception:
        return []


# -- Backend subprocess lifecycle --

class Backend:
    def __init__(self, task_control_port: int, action_port: int, cam_host: str, cam_port: int):
        self.task_control_port = task_control_port
        self.action_port = action_port
        # Forwarded to every backend so the Orin's address lives in one place (this process's
        # --cam-host) -- the Orin's DHCP lease moved .116 -> .117 on 2026-10-08.
        self.cam_host = cam_host
        self.cam_port = cam_port
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.family: str | None = None
        self.label: str | None = None
        self.config_name: str | None = None
        self.checkpoint_path: str | None = None
        self.task: str = DEFAULT_TASK
        self.start_time: float | None = None
        self.log_path: pathlib.Path | None = None
        self.publish_requested: bool = False
        self.rollout_id: str | None = None
        self.recording_dir: str | None = None
        self.experiment: dict | None = None  # experiments.arm_record() of the loaded arm, if any
        # -- Attempts --------------------------------------------------------------------------
        # An "attempt" is one evaluation trial: the robot tries the current language query once
        # and gets one outcome. A loaded policy spans MANY attempts, because the query can be
        # changed live (push_task) without paying the JAX warmup again, which is the whole point
        # of the task-control channel. So the annotatable unit is the attempt, not the backend
        # process -- annotating only on stop (the pre-2026-09-12 behavior) collapsed an entire
        # multi-query session into a single record labelled with whichever query happened to be
        # set at startup.
        self.attempt: dict | None = None          # the attempt currently in progress
        self.attempts_opened: int = 0             # 1-based index counter, reset per policy load
        self.pending_annotations: list[dict] = [] # closed but not yet labelled, oldest first
        self._task_pub = None

    def _get_task_pub(self):
        if self._task_pub is None:
            import zmq
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUB)
            sock.bind(f"tcp://127.0.0.1:{self.task_control_port}")
            time.sleep(0.1)
            self._task_pub = sock
        return self._task_pub

    def _publish_mode(self) -> str | None:
        if self.label is None:
            return None
        script_name = "inference_server_openpi.py" if self.family == "openpi" else "inference_server_lerobot.py"
        if not self.publish_requested:
            return f"DRY RUN -- not driving the arm (see {script_name})"
        # See inference_server_openpi.py's is_absolute_joint / BIMANUAL_JOINT_ACTION_TOPIC --
        # this checkpoint publishes raw joint targets (no IK), not an EE pose.
        if self.config_name == "yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k":
            return "LIVE -- publishing joint-space actions to the Pi (inference/bimanual_joints_action)"
        return "LIVE -- publishing EE-space actions to the Pi (inference/bimanual_ee_action)"

    @staticmethod
    def _attempt_view(attempt: dict | None, running: bool) -> dict | None:
        """JSON-safe copy for the UI: drops the private monotonic stamp and adds a live elapsed."""
        if attempt is None:
            return None
        view = {k: v for k, v in attempt.items() if not k.startswith("_")}
        if "_started_monotonic" in attempt:
            view["elapsed_s"] = time.monotonic() - attempt["_started_monotonic"]
        view["running"] = running
        view["subtask_plan"] = subtask_plan_for(view.get("task"))
        return view

    def status(self) -> dict:
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            exit_code = None if self.proc is None else self.proc.poll()
            uptime = (time.monotonic() - self.start_time) if (running and self.start_time) else None
            log_tail = []
            if self.log_path and self.log_path.exists():
                try:
                    log_tail = self.log_path.read_text(errors="replace").splitlines()[-40:]
                except Exception:
                    pass
            # Self-heal: a loaded, running policy must ALWAYS have an attempt to pass a verdict
            # on. Any path that closes one without reopening (a race, a future edit, an annotate
            # that landed while the process was briefly unpollable) would otherwise leave the
            # operator watching the robot with no button to press, and the only recovery would be
            # a full policy reload -- exactly the cost this whole feature exists to avoid.
            #
            # Gated on rollout_id as well as the process being alive: stop() clears rollout_id
            # (and the attempt) under this lock, then releases it while the process takes up to
            # 8s to exit. A poll landing in that window saw "running, no attempt" and reopened
            # one for a policy that was shutting down, leaving a stale verdict card after unload
            # whose annotation would log with rollout_id None. Observed 2026-09-13.
            if running and self.rollout_id is not None and self.attempt is None:
                self._open_attempt_locked(self.task)
                self._publish_attempt_locked()
            return {
                "running": running,
                "family": self.family,
                "label": self.label,
                "checkpoint_path": self.checkpoint_path,
                "pid": self.proc.pid if self.proc else None,
                "exit_code": exit_code,
                "uptime_s": uptime,
                "task": self.task,
                "publish_mode": self._publish_mode(),
                "log_tail": log_tail,
                "attempt": self._attempt_view(self.attempt, running),
                "pending_annotations": [self._attempt_view(a, False)
                                         for a in self.pending_annotations],
                "annotation_labels": list(ANNOTATION_LABELS),
                "recent_rollouts": _recent_rollouts(),
                "rollout_id": self.rollout_id,
                "recording_dir": self.recording_dir,
                "experiment": self.experiment,
            }

    def stop(self):
        with self.lock:
            proc = self.proc
            self._close_attempt_locked()
            self.rollout_id = None
            self.recording_dir = None
            self.start_time = None
        if proc is None or proc.poll() is not None:
            return
        log.info("Stopping backend pid=%d ...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            log.warning("Backend pid=%d did not exit in time, killing.", proc.pid)
            proc.kill()
            proc.wait(timeout=5)

    def start(self, checkpoint_path: str, label: str, config_name: str | None, task: str,
              family: str = "openpi", publish: bool = False,
              replan_steps: int | None = None, action_horizon: int | None = None,
              rtc: bool = False, gripper_sigmoid_gain: float | None = None,
              gripper_sigmoid_center: float | None = None,
              retrieval_lambda: float | None = None, retrieval_metric: str | None = None,
              retrieval_key_cameras: str | None = None, experiment_arm: str | None = None,
              value_estimator: str | None = None, interp_lamda: float | None = None):
        experiment = None
        context_mode = None
        pool_demos = None
        if not experiment_arm and family == "openpi":
            experiment_arm = _OPENPI_RUN_ARM_ALIASES.get(pathlib.Path(checkpoint_path).parent.name)
        if experiment_arm:
            # The arm, not the request, decides what is loaded and how context is retrieved --
            # otherwise a stray UI value could make a run silently not be the arm it is labelled.
            _, _, arm = experiments.find_arm(experiment_arm)
            if family != "openpi" or rtc:
                raise ValueError("Experiment arms run on the openpi backend without RTC")
            if retrieval_metric and retrieval_metric != arm["retrieval_metric"]:
                raise ValueError(f"Experiment arm {experiment_arm} fixes the retrieval metric; clear the override")
            ckpt = PROJECT_ROOT / arm["checkpoint"]
            if not (ckpt / "params").is_dir():
                raise ValueError(f"Checkpoint for {experiment_arm} is not on disk: {ckpt} "
                                 f"(download {arm['hf_repo']} from Hugging Face)")
            checkpoint_path, config_name = str(ckpt), arm["config"]
            label = experiments.arm_label(experiment_arm)
            retrieval_metric, context_mode = arm["retrieval_metric"], arm["context_mode"]
            pool_demos = arm.get("pool_demos")  # EXP E: only the N shortest demos are retrieval candidates
            fixed_estimator = arm.get("value_estimator")
            if fixed_estimator:
                if value_estimator not in (None, "", fixed_estimator):
                    raise ValueError(f"Experiment arm {experiment_arm} fixes the progress estimator to "
                                     f"{fixed_estimator!r}; clear the override")
                value_estimator = fixed_estimator
            if context_mode != "retrieved":
                retrieval_lambda = retrieval_key_cameras = value_estimator = None  # nothing is retrieved in this arm
            experiment = experiments.arm_record(experiment_arm)
        if value_estimator == "none":
            # "No progress estimator": valid only when the metric in force never reads progress
            # values -- vision-only retrieval. Sent to the server as no --value-estimator at all.
            effective_metric = retrieval_metric or _OPENPI_VICTR_CONFIGS.get(config_name)
            if effective_metric in ("value", "vision_value"):
                raise ValueError(f"{effective_metric} retrieval needs a progress estimator; choose Robometer or VFE")
            value_estimator = None
        if retrieval_lambda is not None:
            import math
            if not math.isfinite(retrieval_lambda) or retrieval_lambda < 0:
                raise ValueError("Retrieval lambda must be finite and nonnegative")
        if retrieval_metric and retrieval_metric not in ("vision", "value", "vision_value"):
            raise ValueError("Unknown retrieval metric")
        if retrieval_key_cameras and retrieval_key_cameras not in ("zed", "zed,fish0,fish1"):
            raise ValueError("Unknown retrieval key cameras")
        if value_estimator and value_estimator not in ("robometer", "robometer_ft", "vfe", "vfe_recap"):
            raise ValueError("Unknown value estimator")
        if (retrieval_lambda is not None or retrieval_metric or retrieval_key_cameras or value_estimator) \
                and config_name not in _OPENPI_VICTR_CONFIGS:
            raise ValueError("Retrieval settings require a VICTR checkpoint")
        if interp_lamda is not None:
            import math
            if config_name not in _OPENPI_INTERP_CONFIGS:
                raise ValueError("Interpolation lambda requires a RICL (action interpolation) checkpoint")
            if not math.isfinite(interp_lamda) or interp_lamda < 0:
                raise ValueError("Interpolation lambda must be finite and nonnegative")
            # In the label, so rollouts.jsonl annotations from different lambdas stay separable.
            label = f"{label} [interp lamda={interp_lamda:g}]"
        self.stop()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{family}_{int(time.time())}.log"
        logfile = open(log_path, "w")

        # Minted before building args (not after Popen, as before) so it can be passed to the
        # backend subprocess itself -- see rollout_recorder.py. One backend subprocess's whole
        # lifetime is exactly one rollout (this method always calls self.stop() first), so the
        # backend can safely record for its entire run under this same id with no separate
        # start/stop signal needed.
        import uuid
        rollout_id = uuid.uuid4().hex
        # Attempt 1's id is minted HERE, before the subprocess is spawned, and handed over as a
        # command-line argument rather than broadcast on the task-control channel. The channel is
        # a zmq PUB/SUB pair, and a PUB that has only just bound silently drops what it sends
        # before its subscribers' subscriptions have propagated -- the classic slow-joiner. Every
        # tick of the first attempt would then be recorded under the server's default index 0,
        # making attempt 1 (and any attempt annotated before the first query change) impossible
        # to separate in the recording. Observed exactly that way on 2026-09-12. Subsequent
        # attempts are fine over the channel: by then the socket is long settled.
        first_attempt_id = uuid.uuid4().hex

        # rollout_recordings/<policy>/<task>/<date_ran>/ -- one directory per rollout, grouped
        # by policy then task so browsing/comparing across many rollouts is actually practical
        # (per user direction 2026-09-07), not a flat pile of uuid directories. date_ran carries
        # time-of-day too, not just the date, since re-running the same (policy, task) more than
        # once on the same day -- the normal case during a real test session -- would otherwise
        # collide.
        policy_slug = _policy_slug(family, config_name, checkpoint_path)
        if experiment:
            # Same directory depth as every other recording, so the rollout browsers still find it.
            policy_slug = _slug(f"{experiment['experiment']}-{experiment['arm']}__{pathlib.Path(checkpoint_path).name}")
        date_ran = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        recording_dir = ROLLOUT_RECORDINGS_DIR / policy_slug / _slug(task) / date_ran

        if family == "openpi":
            if not config_name:
                raise ValueError("openpi checkpoints require config_name")
            if rtc:
                # openpi RTC support (Pi0Rtc/Pi0RtcConfig, --rtc in inference_server_openpi.py)
                # was removed 2026-08-28 after real, repeated safety incidents during live
                # testing (runaway arm behavior) -- see this project's history for the
                # investigation. No openpi backend currently implements RTC.
                raise ValueError("Real-Time Chunking is not available for the openpi backend.")
            args = [str(OPENPI_PYTHON), str(OPENPI_SCRIPT),
                    "--config", config_name,
                    "--checkpoint", checkpoint_path,
                    "--task", task,
                    "--task-control-port", str(self.task_control_port),
                    "--action-port", str(self.action_port),
                    "--rollout-id", rollout_id,
                    "--attempt-id", first_attempt_id,
                    "--attempt-index", "1",
                    "--recording-dir", str(recording_dir)]
            if action_horizon is not None:
                args += ["--action-horizon", str(action_horizon)]
            if retrieval_lambda is not None:
                args += ["--retrieval-lambda", str(retrieval_lambda)]
            if retrieval_metric:
                args += ["--retrieval-metric", retrieval_metric]
            if retrieval_key_cameras:
                args += ["--retrieval-key-cameras", retrieval_key_cameras]
            if value_estimator:
                args += ["--value-estimator", value_estimator]
            if interp_lamda is not None:
                args += ["--interp-lamda", repr(interp_lamda)]
            if context_mode:
                args += ["--context-mode", context_mode]
            if pool_demos is not None:
                args += ["--pool-demos", str(pool_demos)]
            if experiment:
                args += ["--experiment-arm", experiment["arm"]]
            # Blank fields leave these off entirely so the server's own defaults apply -- the
            # UI does not restate them, which would make the default silently live in two places.
            # DISABLED 2026-09-11 with the sigmoid itself -- inference_server_openpi.py's
            # --gripper-sigmoid-* arguments are commented out, so passing them would make every
            # Start fail with "unrecognized arguments".
            # if gripper_sigmoid_gain is not None:
            #     args += ["--gripper-sigmoid-gain", str(gripper_sigmoid_gain)]
            # if gripper_sigmoid_center is not None:
            #     args += ["--gripper-sigmoid-center", str(gripper_sigmoid_center)]
            cwd = PROJECT_ROOT / "openpi"
        elif family == "lerobot":
            # Self-contained checkpoint dir -- no --config registry lookup, no --action-horizon
            # override support (see inference_server_lerobot.py's module docstring).
            args = [str(LEROBOT_PYTHON), str(LEROBOT_SCRIPT),
                    "--checkpoint", checkpoint_path,
                    "--task", task,
                    "--task-control-port", str(self.task_control_port),
                    "--action-port", str(self.action_port),
                    "--rollout-id", rollout_id,
                    "--attempt-id", first_attempt_id,
                    "--attempt-index", "1",
                    "--recording-dir", str(recording_dir)]
            cwd = PROJECT_ROOT
        else:
            raise ValueError(f"Unknown family: {family!r}")

        args += ["--cam-host", self.cam_host, "--cam-port", str(self.cam_port)]
        if publish:
            args.append("--publish")
        if rtc:
            # RTC replaces the receding-horizon replan-steps scheme entirely (see
            # inference_server_lerobot.py's --rtc help text), so --replan-steps is intentionally
            # not passed through when RTC is on.
            args.append("--rtc")
        elif replan_steps is not None:
            args += ["--replan-steps", str(replan_steps)]

        # Bind the task-control PUB before spawning, so the child's SUB connects to a socket
        # that already exists and its subscription settles during the (many-second) model load.
        # Created lazily otherwise, which put the very first live publish -- typically the
        # attempt roll on the first annotation -- right at the bind and into the slow-joiner
        # window. See the --attempt-id note above.
        self._get_task_pub()

        proc = subprocess.Popen(args, cwd=str(cwd), stdout=logfile, stderr=subprocess.STDOUT)

        with self.lock:
            self.proc = proc
            self.family = family
            self.label = label
            self.config_name = config_name
            self.checkpoint_path = checkpoint_path
            self.task = task
            self.start_time = time.monotonic()
            self.log_path = log_path
            self.publish_requested = publish
            self.rollout_id = rollout_id
            self.recording_dir = str(recording_dir)
            self.experiment = experiment
            # Fresh policy load -- restart attempt numbering and open attempt 1. Any attempt
            # still awaiting a verdict from the previous load stays in pending_annotations: it
            # was really run and really watched, so it is still the operator's to label.
            self.attempts_opened = 0
            self.attempt = None
            self._open_attempt_locked(task)
            # Must match the --attempt-id handed to the subprocess above, or the UI and the
            # recording would disagree about what attempt 1 was.
            self.attempt["attempt_id"] = first_attempt_id
        log.info("Started backend pid=%d family=%s label=%r publish=%s log=%s rollout_id=%s "
                  "recording_dir=%s", proc.pid, family, label, publish, log_path, rollout_id,
                  recording_dir)

    # -- Attempt bookkeeping. Callers must hold self.lock. -------------------------------------

    def _open_attempt_locked(self, task: str) -> dict:
        import uuid
        self.attempts_opened += 1
        self.attempt = {
            "attempt_id": uuid.uuid4().hex,
            "attempt_index": self.attempts_opened,
            "rollout_id": self.rollout_id,
            "label": self.label,
            "task": task,
            "recording_dir": self.recording_dir,
            "experiment": self.experiment,
            "started_at": time.time(),
            "_started_monotonic": time.monotonic(),
        }
        return self.attempt

    def _close_attempt_locked(self) -> dict | None:
        """Close the in-progress attempt and queue it for annotation.

        Attempts shorter than MIN_ATTEMPT_S are dropped rather than queued: changing the query
        twice while thinking, or a mis-click, should not leave phantom rows demanding a verdict
        the operator has no memory of. Long enough to have been watched, and it queues."""
        attempt = self.attempt
        self.attempt = None
        if attempt is None:
            return None
        attempt = dict(attempt)
        attempt["ended_at"] = time.time()
        attempt["duration_s"] = time.monotonic() - attempt.pop("_started_monotonic")
        if attempt["duration_s"] < MIN_ATTEMPT_S:
            self.attempts_opened -= 1
            log.info("Discarding attempt %s (%.1fs < %.1fs, too short to judge)",
                      attempt["attempt_id"], attempt["duration_s"], MIN_ATTEMPT_S)
            return None
        self.pending_annotations.append(attempt)
        return attempt

    def push_task(self, task: str):
        """Change the language query live. Opens a new attempt if the query actually changed.

        The old attempt is queued for annotation rather than discarded or force-labelled -- the
        operator may well change the query first and record the verdict a moment later, and
        silently dropping what they just watched is the one behavior that loses real data."""
        with self.lock:
            changed = task != self.task
            self.task = task
            running = self.proc is not None and self.proc.poll() is None
            if changed and running:
                self._close_attempt_locked()
                self._open_attempt_locked(task)
            elif running and self.attempt is not None:
                self.attempt["task"] = task
            payload = {
                "task": task,
                "attempt_id": (self.attempt or {}).get("attempt_id"),
                "attempt_index": (self.attempt or {}).get("attempt_index"),
            }
        try:
            self._get_task_pub().send_multipart([b"task", json.dumps(payload).encode()])
        except Exception as e:
            log.warning("push_task failed: %s", e)

    def annotate(self, attempt_id: str, label: str, partial_subtask_index: int | None = None) -> bool:
        """Record an outcome for one attempt, running or already closed.

        Annotating the IN-PROGRESS attempt is the normal path: the operator watches the robot
        finish, hits a verdict, and a fresh attempt opens immediately for the same query so they
        can just run it again. No stop, no reload, no warmup.

        partial_subtask_index (partial_success only): how many of the task's subtasks were
        completed, in order, before it failed -- 1..len(subtasks)-1, since completing all of
        them is a success. Validated BEFORE anything is closed, so a bad index is rejected
        without consuming the attempt."""
        if label not in ANNOTATION_LABELS:
            log.warning("Rejected unknown annotation label %r", label)
            return False
        with self.lock:
            if self.attempt is not None and self.attempt["attempt_id"] == attempt_id:
                target = self.attempt
            else:
                target = next((a for a in self.pending_annotations
                               if a["attempt_id"] == attempt_id), None)
            if target is None:
                return False
            plan = None
            if partial_subtask_index is not None:
                plan = subtask_plan_for(target.get("task"))
                try:
                    partial_subtask_index = int(partial_subtask_index)
                except (TypeError, ValueError):
                    partial_subtask_index = -1
                if (label != "partial_success" or plan is None
                        or not 1 <= partial_subtask_index < len(plan["subtasks"])):
                    log.warning("Rejected subtask index %r for label %r on task %r",
                                partial_subtask_index, label, target.get("task"))
                    return False
            entry = None
            if self.attempt is not None and self.attempt["attempt_id"] == attempt_id:
                entry = self._close_attempt_locked()
                # Queued by _close_attempt_locked; this call labels it, so unqueue it again.
                if entry is not None:
                    self.pending_annotations = [
                        a for a in self.pending_annotations
                        if a["attempt_id"] != attempt_id
                    ]
                running = self.proc is not None and self.proc.poll() is None
                if running:
                    self._open_attempt_locked(self.task)
                    self._publish_attempt_locked()
            else:
                for i, a in enumerate(self.pending_annotations):
                    if a["attempt_id"] == attempt_id:
                        entry = self.pending_annotations.pop(i)
                        break
            if entry is None:
                return False
            entry = dict(entry)
            entry["annotation"] = label
            entry["timestamp"] = time.time()
            if plan is not None:
                # annotation stays "partial_success" so every existing tally keeps counting it.
                entry["partial_subtask_index"] = partial_subtask_index
                entry["partial_subtask_name"] = plan["subtasks"][partial_subtask_index - 1]
                entry["subtasks_completed"] = plan["subtasks"][:partial_subtask_index]
                entry["subtask_plan"] = plan
        _append_rollout(entry)
        return True

    def _publish_attempt_locked(self):
        """Tell the backend which attempt is now current, so its recording is separable.

        Same channel as a query change; the instruction text is simply unchanged."""
        payload = {
            "task": self.task,
            "attempt_id": (self.attempt or {}).get("attempt_id"),
            "attempt_index": (self.attempt or {}).get("attempt_index"),
        }
        try:
            self._get_task_pub().send_multipart([b"task", json.dumps(payload).encode()])
        except Exception as e:
            log.warning("attempt publish failed: %s", e)


# -- HTML/JS page --

_PAGE = r"""<!DOCTYPE html>
<html>
<head>
  <title>YOR Inference</title>
  <style>
    body { background: #0d0d0d; color: #eee; font-family: -apple-system, sans-serif;
           max-width: 900px; margin: 0 auto; padding: 20px; }
    h1 { font-size: 1.3em; color: #ddd; }
    h2 { font-size: 1.05em; color: #bbb; margin-top: 0; }
    .card { background: #171717; border: 1px solid #2a2a2a; border-radius: 10px;
            padding: 16px 18px; margin-bottom: 16px; }
    .row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .row label { min-width: 100px; color: #999; font-size: 0.9em; }
    select, input[type=text] { flex: 1; min-width: 0; background: #0d0d0d; color: #eee; border: 1px solid #333;
                                 border-radius: 6px; padding: 7px 10px; font-size: 0.95em; }
    button { background: #2a2a2a; color: #eee; border: 1px solid #3a3a3a; border-radius: 6px;
             padding: 8px 16px; cursor: pointer; font-size: 0.95em; }
    button:hover { background: #333; }
    #loadBtn { background: #1e4f2a; border-color: #2a6b3a; }
    #unloadBtn { background: #5f1e1e; border-color: #7a2a2a; }
    #queryBtn { background: #1e3a4f; border-color: #2a556b; flex-shrink: 0; }
    .muted { color: #777; font-size: 0.85em; }
    .dirty { color: #ffcf7e; }
    #task.dirtyField { border-color: #7a5f20; }
    #statusBody { font-size: 0.9em; color: #ccc; line-height: 1.6; }
    #statusEmpty { color: #666; font-style: italic; }
    img#frame, img#contextFrame { max-width: 100%; border-radius: 6px; display: block; }
    /* isolation:isolate keeps 'difference' blending between the two layers only, instead of
       letting the ghost blend through to the page background behind them. */
    #camWrap { position: relative; line-height: 0; isolation: isolate; }
    #camWrap img#frame { width: 100%; }
    #ghost { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
             pointer-events: none; display: none; border-radius: 6px; }
    #onionToggleBox { display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
      border-radius:8px; border:2px solid #2a3f5f; background:#0c1220; width:100%; }
    input[type=range] { flex: 1; min-width: 0; accent-color: #6aa9ff; }
    .rangeVal { min-width: 62px; text-align: right; color: #9ab; font-size: 0.85em;
      font-variant-numeric: tabular-nums; }
    #onionWarn { color: #ffcf7e; font-size: 0.82em; margin-top: 6px; display: none; }
    #onionStatus { color: #888; font-size: 0.82em; }
    table#rolloutTable { width: 100%; border-collapse: collapse; font-size: 0.85em; }
    table#rolloutTable th { text-align: left; color: #888; font-weight: 500; padding: 4px 8px;
      border-bottom: 1px solid #2a2a2a; }
    table#rolloutTable td { padding: 5px 8px; border-bottom: 1px solid #1f1f1f; color: #ccc; }
    #rolloutEmpty { color: #666; font-style: italic; font-size: 0.85em; }
    details > summary { cursor: pointer; color: #888; font-size: 0.85em; user-select: none; }
    #log { background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 6px; height: 200px;
      overflow-y: auto; padding: 10px; font-size: 0.8em; white-space: pre-wrap; margin-top: 8px; color: #999; }
    #expInfo { margin: -2px 0 10px 110px; padding: 10px 12px; border-radius: 8px; border: 1px solid #2a3f5f;
      background: #0c1220; font-size: 0.85em; color: #c8d3e0; line-height: 1.5; display: none; }
    #expInfo b { color: #eee; }
    #expInfo .warn { color: #ffcf7e; }
    .expTag { display: inline-block; font-size: 0.75em; font-weight: 600; padding: 1px 6px; border-radius: 4px;
      border: 1px solid #2a556b; color: #9fd0ff; margin-right: 6px; }
    .expTag.control { border-color: #444; color: #aaa; }
    table.expTable { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-bottom: 12px; }
    table.expTable th { text-align: left; color: #888; font-weight: 500; padding: 4px 8px; border-bottom: 1px solid #2a2a2a; }
    table.expTable td { padding: 5px 8px; border-bottom: 1px solid #1f1f1f; color: #ccc; vertical-align: top; }
    table.expTable td.num { text-align: right; font-variant-numeric: tabular-nums; }
    #contextCard .rcHead { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
    #contextCard .rcHead h2 { margin:0; }
    .rcPill { font-size:0.78em; font-weight:600; padding:2px 9px; border-radius:999px; border:1px solid #444; color:#bbb; }
    .rcPill.ok { border-color:#2a6b3a; color:#7eff9e; background:#0c1a0f; }
    .rcPill.warn { border-color:#7a5f20; color:#ffcf7e; background:#1a1608; }
    .rcPill.bad { border-color:#7a2a2a; color:#ff9e9e; background:#1f0c0c; }
    .rcGrid { display:grid; grid-template-columns: 150px 1fr; gap:4px 12px; font-size:0.85em; margin-bottom:10px; }
    .rcGrid .k { color:#888; }
    .rcGrid .v { color:#ddd; word-break:break-word; }
    .rcLive { font-size:0.85em; color:#ccc; background:#101010; border:1px solid #242424; border-radius:8px; padding:8px 10px; margin-bottom:10px; line-height:1.6; font-variant-numeric: tabular-nums; }
    .rcChecks { list-style:none; padding:0; margin:10px 0 0; font-size:0.85em; }
    .rcChecks li { padding:3px 0; color:#ccc; }
    .rcChecks li .i { display:inline-block; width:18px; font-weight:700; }
    .rcChecks li.ok .i { color:#7eff9e; } .rcChecks li.warn .i { color:#ffcf7e; } .rcChecks li.bad .i { color:#ff9e9e; } .rcChecks li.info .i { color:#9fd0ff; }
    #rcTrend { width:100%; height:92px; display:block; margin-top:6px; }
    #annotationPrompt { margin-top: 12px; padding: 10px 12px; background: #1a1a12; border: 1px solid #4a4020;
      border-radius: 8px; display: none; }
    .attemptCard { margin-top: 12px; padding: 10px 12px; border-radius: 8px; display: none; }
    #currentAttempt { background: #0f1a12; border: 1px solid #2a4a32; }
    #pendingAttempts { background: #1a1a12; border: 1px solid #4a4020; }
    .attemptHead { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
    .attemptIdx { font-weight:600; color:#8fd6a3; }
    #pendingAttempts .attemptIdx { color:#d6c68f; }
    .attemptTask { color:#bbb; font-size:0.9em; word-break:break-word; }
    .attemptMeta { color:#777; font-size:0.8em; }
    .verdicts { display:flex; gap:6px; flex-wrap:wrap; }
    .verdicts button { padding:6px 12px; border-radius:6px; border:1px solid #3a3a3a;
      background:#1e1e1e; color:#ddd; cursor:pointer; font-size:0.85em; }
    .verdicts button:hover { border-color:#666; }
    .verdicts button[data-label="success"]:hover { border-color:#3f8f5a; color:#8fd6a3; }
    .verdicts button[data-label="partial_success"]:hover { border-color:#8f7a3f; color:#d6c68f; }
    .verdicts button[data-label="failure"]:hover { border-color:#8f3f3f; color:#d68f8f; }
    .pendingRow { border-top:1px solid #2a2a1a; padding-top:8px; margin-top:8px; }
    .partialGroup { flex-basis:100%; display:flex; flex-wrap:wrap; align-items:center; gap:6px;
      padding:6px 8px; border:1px dashed #4a4020; border-radius:6px; }
    .partialCaption { color:#d6c68f; font-size:0.8em; flex-basis:100%; }
    .partialGroup button { border-color:#5a4a20; }
    .partialGroup button:hover { border-color:#8f7a3f; color:#d6c68f; }
    #publishToggleBox { display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
      border-radius:8px; border:2px solid #5f4a1e; background:#1a1608; width:100%; }
  </style>
</head>
<body>
  <h1>YOR Inference</h1>

  <div class="card">
    <h2>Checkpoint &amp; Controls</h2>
    <div class="row">
      <label>Experiment</label>
      <select id="expArm" title="EXP A / EXP B arms fix the checkpoint, retrieval metric and context mode"></select>
    </div>
    <div id="expInfo"></div>
    <div class="row" id="ckptRow">
      <label>Checkpoint</label>
      <select id="ckpt"></select>
    </div>
    <div class="row">
      <div id="publishToggleBox">
        <input type="checkbox" id="publish" style="width:18px; height:18px; margin-top:2px; cursor:pointer; flex-shrink:0;">
        <label for="publish" style="cursor:pointer; margin:0; flex:1; min-width:auto;">
          <strong id="publishStateText" style="color:#ffcf7e;">DRY RUN &mdash; actions will NOT be sent to the robot</strong><br>
          <span style="font-size:0.8em; color:#999;">Check to publish live. Requires the Pi's joystick control loop + policy rollout enabled first.</span>
        </label>
      </div>
    </div>
    <div class="row" id="rtcRow">
      <input id="rtc" type="checkbox" style="width:18px; height:18px; margin-top:2px; cursor:pointer; flex-shrink:0;">
      <label for="rtc" style="cursor:pointer; margin:0; flex:1; min-width:auto;">
        Real-time chunking (RTC)<br>
        <span id="rtcHint" style="font-size:0.8em; color:#999;">A background thread continuously predicts the next chunk while the current one is still publishing, blended smoothly against it instead of pausing at each replan. Replaces "Replan steps" entirely. lerobot checkpoints only -- no openpi implementation.</span>
      </label>
    </div>
    <div class="row" id="replanStepsRow">
      <label>Replan steps</label>
      <input id="replanSteps" type="text" value="30" placeholder="5"
             title="Publish only the first N steps of each predicted chunk, then replan. Blank = backend default (5). Ignored when Real-time chunking is enabled.">
    </div>
    <div class="row" id="actionHorizonRow">
      <label>Action chunk</label>
      <input id="actionHorizon" type="text" value="" placeholder="use checkpoint's own default"
             title="EXPERIMENTAL: override the loaded checkpoint's action_horizon (chunk length) at inference time. Blank = use whatever the checkpoint/config was trained with. Quality beyond what the checkpoint actually saw in training is unverified. openpi checkpoints only -- inference_server_lerobot.py has no override support.">
    </div>
    <div class="row" id="retrievalRow" style="display:none;">
      <label>Context retrieval</label>
      <select id="retrievalMetric">
        <option value="">Checkpoint default</option>
        <option value="vision">Vision</option>
        <option value="value">Value (progress)</option>
        <option value="vision_value">Vision + value</option>
      </select>
      <select id="valueEstimator" title="Progress estimator for value / vision+value retrieval, used for both the live query and the pool. Unused by vision-only retrieval.">
        <option value="">Progress: Robometer (default)</option>
        <option value="vfe">Progress: IC-VFE (step 30000, demo context)</option>
        <option value="vfe_recap">Progress: RECAP VFE (step 20000, no context)</option>
        <option value="robometer_ft">Progress: Robometer finetuned (icl-dataset)</option>
        <option value="none">Progress: none (vision retrieval)</option>
      </select>
      <select id="retrievalKeyCameras" title="Cameras whose DINOv2 keys are used to rank pool chunks">
        <option value="">Keys: checkpoint default (head)</option>
        <option value="zed">Keys: head camera</option>
        <option value="zed,fish0,fish1">Keys: head + both wrists</option>
      </select>
      <input id="retrievalLambda" type="number" min="0" step="0.1" placeholder="lambda: 1"
             title="Value weight in fused retrieval. 1 matches training; larger values favor progress similarity.">
      <input id="interpLamda" type="number" min="0" step="0.5" placeholder="interp lamda: 3 (trained)" style="display:none;"
             title="RICL only: lamda in the blend weight w = exp(-lamda * d / d_max). Blank = 3, as trained. 1000 = no blend (w = 0); 0 = execute the retrieved neighbor's action verbatim (w = 1, dry-run first). Only w changes: same weights, neighbor and context.">
    </div>
    <div class="row" id="gripperSigmoidRow" style="display:none;">  <!-- DISABLED 2026-09-11 with the server-side sigmoid -->
      <label>Gripper shaping</label>
      <input id="gripperGain" type="text" value="" placeholder="gain (blank = 12)"
             title="Steepness of the sigmoid applied to the predicted gripper before publishing. Higher is more decisive. 0 disables it and publishes the raw prediction. Blank uses the server default.">
      <input id="gripperCenter" type="text" value="" placeholder="centre (blank = 0.4)"
             title="Predicted gripper value (0=open, 1=closed) that maps to half-closed. Below 0.5 makes the gripper close earlier and release later. 0.5 is a pure sharpener with no timing shift. Blank uses the server default.">
    </div>
    <!-- Everything above this line is fixed for the life of a load: they become command-line
         arguments to the backend process, so changing one means reloading and paying the JAX
         warmup again. Everything below is live. -->
    <div class="row">
      <button id="loadBtn">Load policy</button>
      <button id="unloadBtn">Unload</button>
      <span id="loadState" class="muted"></span>
    </div>
  </div>

  <!-- Separate card on purpose: the language query is NOT a load-time setting. It can be
       changed as often as you like against one loaded policy, which is the entire point of the
       task-control channel -- a reload costs a full model load plus JIT warmup. -->
  <div class="card">
    <h2>Language query</h2>
    <div class="row">
      <label>Query</label>
      <input id="task" type="text" value="pick up the cube with the left hand and place it on the plate"
             title="Sent to the running policy on Enter. Changing it opens a new attempt; the previous one stays annotatable.">
      <button id="queryBtn">Enter</button>
    </div>
    <div class="row">
      <span id="queryState" class="muted"></span>
    </div>
    <!-- The attempt in progress. Verdict buttons are live: recording one closes this attempt
         and immediately opens the next for the same query, so a whole eval session runs without
         ever stopping the policy (and re-paying JAX warmup). -->
    <div id="currentAttempt" class="attemptCard">
      <div class="attemptHead">
        <span class="attemptIdx" id="curAttemptIdx"></span>
        <span class="attemptTask" id="curAttemptTask"></span>
        <span class="attemptMeta" id="curAttemptMeta"></span>
      </div>
      <div class="verdicts" id="curVerdicts"></div>
    </div>
    <!-- Attempts closed by a query change before a verdict was recorded. Queued rather than
         dropped, because they really ran and the operator really watched them. -->
    <div id="pendingAttempts" class="attemptCard">
      <div class="attemptHead"><span class="attemptIdx">Awaiting a verdict</span></div>
      <div id="pendingList"></div>
    </div>
  </div>

  <div class="card">
    <h2>Status</h2>
    <div id="statusBody"><span id="statusEmpty">Not running.</span></div>
  </div>

  <div class="card">
    <h2>Camera preview</h2>
    <div id="camWrap">
      <img id="frame" src="/frame.jpg">
      <img id="ghost">
    </div>
  </div>

  <div class="card" id="onionCard">
    <h2>Onion overlay &mdash; start-pose alignment</h2>
    <div class="row">
      <div id="onionToggleBox">
        <input type="checkbox" id="onionOn" style="width:18px; height:18px; margin-top:2px; cursor:pointer; flex-shrink:0;">
        <label for="onionOn" style="cursor:pointer; margin:0; flex:1; min-width:auto;">
          <strong style="color:#8ec5ff;">Ghost a dataset episode over the live cameras</strong><br>
          <span style="font-size:0.8em; color:#999;">Move the base and arms until the live view lines up
          with the ghost, so the rollout starts from an initial condition the policy actually trained on.
          Both layers are drawn at the geometry the policy is fed (ZED squashed to the dataset's 640x480),
          not the raw camera frame.</span>
        </label>
      </div>
    </div>
    <div id="onionControls" style="display:none;">
      <div class="row">
        <label>Task</label>
        <select id="onionTask"></select>
      </div>
      <div id="onionWarn"></div>
      <div class="row">
        <label>Episode</label>
        <select id="onionEp"></select>
      </div>
      <div class="row">
        <label>Opacity</label>
        <input type="range" id="onionOpacity" min="5" max="100" value="35">
        <span class="rangeVal" id="onionOpacityVal">35%</span>
      </div>
      <div class="row">
        <label>Blend</label>
        <select id="onionBlend">
          <option value="normal">Normal &mdash; ghost fades over the live view</option>
          <option value="difference">Difference &mdash; goes black where the two match</option>
        </select>
      </div>
      <div class="row">
        <label>Frame</label>
        <input type="range" id="onionFrame" min="0" max="0" value="0">
        <span class="rangeVal" id="onionFrameVal">0</span>
      </div>
      <div class="row">
        <button id="onionFirst">First frame</button>
        <button id="onionPlay">Play</button>
        <span id="onionStatus"></span>
      </div>
    </div>
  </div>

  <div class="card" id="contextCard" style="display:none;">
    <div class="rcHead"><h2>Retrieval check (VICTR)</h2><span id="rcPill" class="rcPill">waiting</span></div>
    <div class="rcGrid" id="rcSettings"></div>
    <div class="rcLive" id="rcLive">Waiting for the first retrieval…</div>
    <img id="contextFrame" alt="Live query frame next to the frames of the retrieved demonstration chunk">
    <svg id="rcTrend" viewBox="0 0 600 92" preserveAspectRatio="none" role="img" aria-label="Retrieval score and query progress over recent ticks"></svg>
    <div class="muted" id="rcTrendLegend"></div>
    <ul class="rcChecks" id="rcChecks"></ul>
  </div>

  <div class="card" id="expCard">
    <h2>Experiments &mdash; annotated attempts per arm</h2>
    <div id="expResults" class="muted">Loading…</div>
  </div>

  <div class="card">
    <h2>Recent rollouts</h2>
    <table id="rolloutTable"><thead><tr><th>Time</th><th>Arm</th><th>Task</th><th>Annotation</th><th>Duration</th></tr></thead>
      <tbody id="rolloutBody"></tbody></table>
    <div id="rolloutEmpty" style="display:none;">No rollouts yet.</div>
  </div>

  <div class="card">
    <details>
      <summary>Backend log (last 40 lines)</summary>
      <div id="log"></div>
    </details>
  </div>

<script>
let checkpoints = [];

let experimentsData = {};

function selectedArm() {
  const id = document.getElementById('expArm').value;
  for (const exp of Object.values(experimentsData)) if (exp.arms[id]) return { id, exp, arm: exp.arms[id] };
  return null;
}

async function loadExperiments() {
  experimentsData = await (await fetch('/experiments')).json();
  const sel = document.getElementById('expArm');
  const prev = sel.value;
  sel.innerHTML = '<option value="">None &mdash; choose a checkpoint below</option>';
  for (const exp of Object.values(experimentsData)) {
    const grp = document.createElement('optgroup');
    grp.label = `${exp.name}: ${exp.title}`;
    for (const [id, arm] of Object.entries(exp.arms)) {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `${exp.name} · ${id} ${arm.role === 'control' ? '(control)' : ''} — ${arm.summary}${arm.available ? '' : '  [checkpoint missing]'}`;
      grp.appendChild(opt);
    }
    sel.appendChild(grp);
  }
  if (prev) sel.value = prev;
  renderExperimentResults();
  refreshFieldVisibility();
}

function renderExperimentResults() {
  const box = document.getElementById('expResults');
  const labels = ['success', 'partial_success', 'failure', 'skip'];
  box.className = '';
  box.innerHTML = Object.values(experimentsData).map(exp => `
    <div style="color:#bbb; font-size:0.9em; margin-bottom:4px;"><b>${exp.name}</b> — ${exp.question}</div>
    <table class="expTable"><thead><tr><th>Arm</th><th>Trained with → deployed with</th>
      <th class="num">success</th><th class="num">partial</th><th class="num">failure</th><th class="num">skip</th><th class="num">success rate</th></tr></thead><tbody>
    ${Object.entries(exp.arms).map(([id, a]) => {
      const t = a.tally, judged = t.success + t.partial_success + t.failure;
      return `<tr><td><span class="expTag ${a.role === 'control' ? 'control' : ''}">${id}</span>${a.role}${a.control ? ` <span class="muted">vs ${a.control}</span>` : ''}</td>
        <td>${a.trained} → <b>${a.deployed}</b><div class="muted">${a.config}${a.available ? '' : ' · checkpoint missing'}</div></td>
        ${labels.map(l => `<td class="num">${t[l]}</td>`).join('')}
        <td class="num">${judged ? `${Math.round(100 * t.success / judged)}% of ${judged}` : '—'}</td></tr>`;
    }).join('')}</tbody></table>`).join('') +
    '<div class="muted">Counts come from annotated attempts in rollout_logs/rollouts.jsonl; skips are excluded from the rate. Compare each treatment only with its own control, on the same tasks and start poses.</div>';
}

function selectedEntry() {
  const sel = document.getElementById('ckpt');
  return checkpoints.find(c => c.path === sel.value);
}

function updatePublishToggleStyle() {
  const checked = document.getElementById('publish').checked;
  const box = document.getElementById('publishToggleBox');
  const text = document.getElementById('publishStateText');
  if (checked) {
    box.style.borderColor = '#1e4f2a'; box.style.background = '#0c1a0f';
    text.style.color = '#7eff9e'; text.innerHTML = 'LIVE &mdash; actions WILL be published to the robot';
  } else {
    box.style.borderColor = '#5f4a1e'; box.style.background = '#1a1608';
    text.style.color = '#ffcf7e'; text.innerHTML = 'DRY RUN &mdash; actions will NOT be sent to the robot';
  }
}
document.getElementById('publish').addEventListener('change', updatePublishToggleStyle);

function refreshFieldVisibility() {
  const sel = selectedArm();
  const info = document.getElementById('expInfo');
  document.getElementById('ckptRow').style.display = sel ? 'none' : 'flex';
  if (sel) {
    const a = sel.arm;
    info.style.display = 'block';
    info.innerHTML = `<b>${sel.exp.name} · ${sel.id}</b> (${a.role}${a.control ? `, compare with ${a.control}` : ''}) — ${a.summary}<br>
      Checkpoint <code>${a.checkpoint}</code> · config <code>${a.config}</code><br>
      Trained with <b>${a.trained}</b>; this run deploys with <b>${a.deployed}</b>.
      ${a.retrieval_metric === 'value' ? '<br>Value retrieval starts the progress-estimator worker on first load (Robometer ~1.5 min; IC-VFE ~1 min, up to ~6 min on a cold compile cache; RECAP VFE seconds).' : ''}
      ${a.value_estimator ? `<br>Progress estimator fixed: <b>${ESTIMATOR_NAME[a.value_estimator] || a.value_estimator}</b>.` : ''}
      ${a.pool_demos ? `<br>Demo buffer fixed: the <b>${a.pool_demos} shortest</b> successful (keep=true) demonstration${a.pool_demos === 1 ? '' : 's'} of the task pool; a pool with fewer keeps all of them. Keep the progress estimator the same for every ${sel.exp.name} arm.` : ''}
      ${a.available ? '' : `<br><span class="warn">Checkpoint not on disk — download ${a.hf_repo} into ${a.checkpoint} before loading.</span>`}`;
  } else {
    info.style.display = 'none';
  }
  const entry = sel ? { family: 'openpi', victr_capable: sel.arm.context_mode === 'retrieved', rtc_capable: false } : selectedEntry();
  const isLerobot = entry && entry.family === 'lerobot';
  document.getElementById('actionHorizonRow').style.display = isLerobot ? 'none' : 'flex';
  // DISABLED 2026-09-11: row stays hidden while the server-side sigmoid is commented out.
  // document.getElementById('gripperSigmoidRow').style.display = isLerobot ? 'none' : 'flex';

  // The retrieved-context panel lives on the backend subprocess's own debug server
  // (default port 8082 -- see inference_server_openpi.py's --viewer-port), not this
  // process, so it needs an absolute URL rather than a path this server itself
  // handles (unlike /frame.jpg, which IS served by this process's own independent
  // camera subscriber -- see module docstring). NOT port 8081: confirmed 2026-08-30
  // that port is permanently occupied by an unrelated pre-existing service on this
  // machine, so the debug server never actually bound there -- this panel was
  // silently hitting that other app's own HTML instead of ever reaching our server.
  document.getElementById('retrievalRow').style.display = (entry && entry.victr_capable) ? 'flex' : 'none';
  document.getElementById('interpLamda').style.display = (!sel && entry && entry.interp_capable) ? '' : 'none';
  // An arm fixes its metric; key cameras and lambda stay adjustable (apply to arm and control alike).
  const metricSel = document.getElementById('retrievalMetric');
  metricSel.disabled = !!sel;
  if (sel) metricSel.value = '';

  // Progress estimator: only value and vision+value retrieval read progress values. For a
  // vision-only run (the checkpoint's own metric, an arm's fixed metric, or a Vision override)
  // "none" is selected and the choice locked, since Robometer/VFE would do nothing there.
  const estSel = document.getElementById('valueEstimator');
  const noneOpt = estSel.querySelector('option[value="none"]');
  const effectiveMetric = sel ? (sel.arm.retrieval_metric || '') : (metricSel.value || (entry && entry.victr_metric) || '');
  const visionOnly = !!(entry && entry.victr_capable) && effectiveMetric === 'vision';
  noneOpt.hidden = noneOpt.disabled = !visionOnly;
  estSel.disabled = visionOnly;
  if (visionOnly) estSel.value = 'none';
  else if (estSel.value === 'none') estSel.value = '';
  estSel.title = visionOnly
    ? 'Vision retrieval does not use a progress estimator.'
    : 'Progress estimator for value / vision+value retrieval, used for both the live query and the pool.';
  // An experiment arm that fixes its estimator (EXP A2: IC-VFE) locks the dropdown to it.
  const fixedEst = sel && sel.arm.value_estimator;
  if (fixedEst) {
    estSel.value = fixedEst;
    estSel.disabled = true;
    estSel.title = `${sel.id} fixes the progress estimator to ${ESTIMATOR_NAME[fixedEst] || fixedEst}.`;
  }

  const rtcCapable = !!(entry && entry.rtc_capable);
  const rtcBox = document.getElementById('rtc');
  const rtcRow = document.getElementById('rtcRow');
  const rtcHint = document.getElementById('rtcHint');
  rtcBox.disabled = !rtcCapable;
  rtcRow.style.opacity = rtcCapable ? '1' : '0.5';
  if (!rtcCapable) {
    rtcBox.checked = false;
    rtcHint.textContent = isLerobot
      ? 'Not available for this checkpoint.'
      : 'lerobot checkpoints only -- no openpi implementation.';
  } else {
    rtcHint.textContent = 'A background thread continuously predicts the next chunk while the current one is still publishing, blended smoothly against it instead of pausing at each replan. Replaces "Replan steps" entirely.';
  }
  updateReplanStepsEnabled();
}

// replanSteps/actionHorizon are free-text fields that otherwise never reset -- without this
// a stale value left over from a PREVIOUS, unrelated checkpoint selection would silently carry
// over and get sent as --replan-steps/--action-horizon for the newly selected one too, which
// directly affects real actuation. Reset only on an actual checkpoint CHANGE (the dropdown's
// own 'change' event) -- NOT from refreshFieldVisibility(), which loadCheckpoints() also calls
// on every 5s poll regardless of whether the selection changed; wiring the reset through there
// used to silently overwrite anything typed in (e.g. 30) back to the default within 5 seconds,
// even with the same checkpoint still selected.
function resetOverridesForNewCheckpoint() {
  document.getElementById('replanSteps').value = '30';
  document.getElementById('actionHorizon').value = '';
  document.getElementById('retrievalMetric').value = '';
  document.getElementById('retrievalLambda').value = '';
  document.getElementById('interpLamda').value = '';
  document.getElementById('retrievalKeyCameras').value = '';
  document.getElementById('valueEstimator').value = '';
  document.getElementById('gripperGain').value = '';
  document.getElementById('gripperCenter').value = '';
}

function updateReplanStepsEnabled() {
  const rtcOn = document.getElementById('rtc').checked;
  const replanInput = document.getElementById('replanSteps');
  replanInput.disabled = rtcOn;
  document.getElementById('replanStepsRow').style.opacity = rtcOn ? '0.5' : '1';
}
document.getElementById('rtc').addEventListener('change', updateReplanStepsEnabled);

async function loadCheckpoints() {
  const res = await fetch('/checkpoints');
  checkpoints = await res.json();
  const sel = document.getElementById('ckpt');
  const prevValue = sel.value;
  sel.innerHTML = '';

  // Group by date_added (newest first) -- flat, undated lists get confusing fast
  // once checkpoints accumulate across many sessions. "unknown" (no readable
  // mtime) sorts last rather than breaking the date ordering.
  const groups = new Map();
  for (const c of checkpoints) {
    const key = c.date_added || 'unknown';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  }
  const dateKeys = [...groups.keys()].sort((a, b) => {
    if (a === 'unknown') return 1;
    if (b === 'unknown') return -1;
    return b.localeCompare(a);  // newest first
  });
  for (const dateKey of dateKeys) {
    const grp = document.createElement('optgroup');
    grp.label = dateKey === 'unknown' ? 'Date unknown' : dateKey;
    for (const c of groups.get(dateKey)) {
      const opt = document.createElement('option');
      opt.value = c.path;
      // Reason strings can run to a paragraph (e.g. a tokenizer-mismatch writeup) --
      // a single very long <option> forces some browsers to render the whole <select>
      // at that content width, ignoring its own flex/width CSS entirely (confirmed
      // 2026-08-30: this, not the retrieval-context <img>, was the real cause of a
      // long horizontal scrollbar). Truncate what's shown; the full text is still one
      // hover away via the title tooltip.
      const shortReason = c.reason && c.reason.length > 60 ? c.reason.slice(0, 60) + '...' : c.reason;
      opt.textContent = c.label + (c.runnable ? '' : '  (disabled: ' + shortReason + ')');
      if (c.reason) opt.title = c.reason;
      opt.disabled = !c.runnable;
      grp.appendChild(opt);
    }
    sel.appendChild(grp);
  }
  if (prevValue && checkpoints.some(c => c.path === prevValue)) sel.value = prevValue;
  refreshFieldVisibility();
}
document.getElementById('ckpt').addEventListener('change', () => {
  resetOverridesForNewCheckpoint();
  refreshFieldVisibility();
});

document.getElementById('expArm').addEventListener('change', () => {
  resetOverridesForNewCheckpoint();
  refreshFieldVisibility();
});

document.getElementById('retrievalMetric').addEventListener('change', refreshFieldVisibility);

document.getElementById('loadBtn').addEventListener('click', async () => {
  const arm = selectedArm();
  if (arm) {
    const res = await fetch('/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        experiment_arm: arm.id, family: 'openpi',
        task: document.getElementById('task').value,
        publish: document.getElementById('publish').checked,
        replan_steps: document.getElementById('replanSteps').value,
        action_horizon: document.getElementById('actionHorizon').value,
        retrieval_lambda: arm.arm.context_mode === 'retrieved' ? document.getElementById('retrievalLambda').value : '',
        retrieval_key_cameras: arm.arm.context_mode === 'retrieved' ? document.getElementById('retrievalKeyCameras').value : '',
        value_estimator: arm.arm.context_mode === 'retrieved' ? document.getElementById('valueEstimator').value : '',
      }),
    });
    if (!res.ok) alert('Start failed: ' + (await res.json()).error);
    refreshStatus();
    return;
  }
  const entry = selectedEntry();
  if (!entry) return;
  const res = await fetch('/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      path: entry.path,
      label: entry.label,
      config_name: entry.config_name,
      family: entry.family,
      task: document.getElementById('task').value,
      publish: document.getElementById('publish').checked,
      replan_steps: document.getElementById('replanSteps').value,
      action_horizon: entry.family === 'lerobot' ? '' : document.getElementById('actionHorizon').value,
      retrieval_metric: entry.victr_capable ? document.getElementById('retrievalMetric').value : '',
      retrieval_lambda: entry.victr_capable ? document.getElementById('retrievalLambda').value : '',
      retrieval_key_cameras: entry.victr_capable ? document.getElementById('retrievalKeyCameras').value : '',
      value_estimator: entry.victr_capable ? document.getElementById('valueEstimator').value : '',
      interp_lamda: entry.interp_capable ? document.getElementById('interpLamda').value : '',
      rtc: !!entry.rtc_capable && document.getElementById('rtc').checked,
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    alert('Start failed: ' + err.error);
  }
  refreshStatus();
});

document.getElementById('unloadBtn').addEventListener('click', async () => {
  await fetch('/stop', {method: 'POST'});
  refreshStatus();
});

// The query is committed explicitly, never on a timer. The previous build pushed it on a 500ms
// debounce, so half-typed instructions ("pick up the red") reached the model and each pause
// opened an attempt -- the query the policy was actually given became impossible to know from
// the UI alone. Committed is now exactly what you pressed Enter on.
let committedTask = null;   // last value the server acknowledged, for the dirty indicator
let taskAdopted = false;    // has this page pulled the server's in-force query into the field yet

async function commitQuery() {
  const value = document.getElementById('task').value;
  await fetch('/task', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task: value}),
  });
  committedTask = value;
  refreshStatus();
}

document.getElementById('queryBtn').addEventListener('click', commitQuery);
document.getElementById('task').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); commitQuery(); }
});
document.getElementById('task').addEventListener('input', renderQueryState);

// Typing alone changes nothing on the robot, so say so plainly -- an operator who types a new
// query and watches the old behavior continue would otherwise have no way to tell why.
function renderQueryState() {
  const input = document.getElementById('task');
  const label = document.getElementById('queryState');
  const typed = input.value;
  if (committedTask === null) {
    label.textContent = 'Not sent yet — loading a policy sends this as the first query.';
    label.className = 'muted';
    input.classList.remove('dirtyField');
  } else if (typed !== committedTask) {
    label.textContent = 'Edited — press Enter to send it to the running policy.';
    label.className = 'muted dirty';
    input.classList.add('dirtyField');
  } else {
    label.textContent = 'Sent. Changing it opens a new attempt.';
    label.className = 'muted';
    input.classList.remove('dirtyField');
  }
}

const ESTIMATOR_NAME = {robometer: 'Robometer', robometer_ft: 'Robometer finetuned', vfe: 'IC-VFE', vfe_recap: 'RECAP VFE'};
const LABEL_TEXT = {
  success: 'Success',
  partial_success: 'Partial success',
  failure: 'Failure',
  skip: 'Skip',
};
const LABEL_HINT = {
  success: 'Completed the task',
  partial_success: 'Got partway (e.g. grasped and lifted) then failed before finishing',
  failure: 'Failed the task entirely',
  skip: "Don't count this attempt (setup error, operator mistake, robot fault)",
};

async function annotateAttempt(attemptId, label, subtaskIndex) {
  const payload = {attempt_id: attemptId, label: label};
  if (subtaskIndex !== undefined) payload.partial_subtask_index = subtaskIndex;
  const res = await fetch('/annotate', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const out = await res.json().catch(() => ({}));
  if (!out.ok) alert('Annotation was not recorded (the attempt may already be labelled).');
  refreshStatus();
}

function planKey(plan) {
  return plan ? plan.subtasks.join('|') : '';
}

// Built fresh rather than bound once: the attempt being judged changes as the session runs, and
// stale attempt ids in a handler closure would silently annotate the wrong trial.
function verdictButtons(attemptId, labels, plan) {
  const wrap = document.createElement('div');
  wrap.className = 'verdicts';
  for (const label of labels) {
    // With a plan, partial success becomes one button per subtask: the one pressed is the LAST
    // subtask the robot completed. The final subtask gets no button -- finishing it is Success.
    if (label === 'partial_success' && plan && plan.subtasks.length >= 2) {
      const group = document.createElement('div');
      group.className = 'partialGroup';
      const cap = document.createElement('span');
      cap.className = 'partialCaption';
      cap.textContent = 'Partial success — last subtask completed:';
      group.appendChild(cap);
      plan.subtasks.slice(0, -1).forEach((name, i) => {
        const k = i + 1;
        const b = document.createElement('button');
        b.dataset.label = label;
        b.textContent = `${k}. ${name}`;
        b.title = `Completed: ${plan.subtasks.slice(0, k).join(' → ')}\nFailed at: ${plan.subtasks[k]}`;
        b.addEventListener('click', () => annotateAttempt(attemptId, label, k));
        group.appendChild(b);
      });
      wrap.appendChild(group);
      continue;
    }
    const b = document.createElement('button');
    b.dataset.label = label;
    b.textContent = LABEL_TEXT[label] || label;
    b.title = LABEL_HINT[label] || '';
    b.addEventListener('click', () => annotateAttempt(attemptId, label));
    wrap.appendChild(b);
  }
  return wrap;
}

async function refreshStatus() {
  const res = await fetch('/status');
  const s = await res.json();
  lastStatus = s;
  const body = document.getElementById('statusBody');
  if (!s.running && !s.label) {
    body.innerHTML = '<span id="statusEmpty">Not running.</span>';
  } else {
    body.innerHTML = `
      <div><b>Running:</b> ${s.running} ${s.pid ? '(pid ' + s.pid + ')' : ''}</div>
      <div><b>Label:</b> ${s.label || '-'}</div>
      ${s.experiment ? `<div><b>Experiment:</b> ${s.experiment.experiment_name} ${s.experiment.arm} (${s.experiment.role}) — trained with ${s.experiment.trained}, deployed with ${s.experiment.deployed}</div>` : ''}
      <div><b>Mode:</b> ${s.publish_mode || '-'}</div>
      <div><b>Uptime:</b> ${s.uptime_s ? s.uptime_s.toFixed(0) + 's' : '-'}</div>
      <div><b>Exit code:</b> ${s.exit_code === null ? '-' : s.exit_code}</div>
    `;
  }
  document.getElementById('log').textContent = (s.log_tail || []).join('\n');
  document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;

  // Load state, and the query the SERVER believes is in force. Taken from /status rather than
  // remembered locally so a reloaded page, or a second browser, shows the truth instead of its
  // own stale idea of it.
  const loadState = document.getElementById('loadState');
  if (s.running) {
    loadState.textContent = `loaded — ${s.label || ''}`;
    loadState.className = 'muted';
  } else if (s.label) {
    loadState.textContent = `unloaded (last: ${s.label}${s.exit_code ? ', exit ' + s.exit_code : ''})`;
    loadState.className = 'muted';
  } else {
    loadState.textContent = 'no policy loaded';
    loadState.className = 'muted';
  }

  if (s.label && s.task != null) {
    // Adopt the server's query into the field ONCE, on the first poll after this page loads, so
    // a reload mid-session shows the query actually in force rather than the HTML default (which
    // would read as "Edited" and invite an accidental change). After that the field is the
    // operator's: a 1s poll must never overwrite what they are typing.
    if (!taskAdopted && document.activeElement !== document.getElementById('task')) {
      document.getElementById('task').value = s.task;
      taskAdopted = true;
    }
    committedTask = s.task;
  }
  renderQueryState();

  const labels = s.annotation_labels || ['success', 'partial_success', 'failure', 'skip'];

  const curCard = document.getElementById('currentAttempt');
  const cur = s.attempt;
  if (cur) {
    curCard.style.display = 'block';
    document.getElementById('curAttemptIdx').textContent = `Attempt ${cur.attempt_index}`;
    document.getElementById('curAttemptTask').textContent = cur.task || '';
    document.getElementById('curAttemptMeta').textContent =
      `${(cur.elapsed_s || 0).toFixed(0)}s elapsed`;
    const box = document.getElementById('curVerdicts');
    // Rebuild only when the attempt actually changes, so a mid-click 1s poll cannot swap the
    // button out from under the pointer.
    const curKey = cur.attempt_id + '#' + planKey(cur.subtask_plan);
    if (box.dataset.attemptId !== curKey) {
      box.dataset.attemptId = curKey;
      box.replaceChildren(...verdictButtons(cur.attempt_id, labels, cur.subtask_plan).childNodes);
    }
  } else {
    curCard.style.display = 'none';
    document.getElementById('curVerdicts').dataset.attemptId = '';
  }

  const pendCard = document.getElementById('pendingAttempts');
  const pending = s.pending_annotations || [];
  const pendKey = pending.map(a => a.attempt_id + '#' + planKey(a.subtask_plan)).join(',');
  pendCard.style.display = pending.length ? 'block' : 'none';
  if (pendCard.dataset.key !== pendKey) {
    pendCard.dataset.key = pendKey;
    const list = document.getElementById('pendingList');
    list.replaceChildren();
    for (const a of pending) {
      const row = document.createElement('div');
      row.className = 'pendingRow';
      const head = document.createElement('div');
      head.className = 'attemptHead';
      head.innerHTML =
        `<span class="attemptIdx">Attempt ${a.attempt_index}</span>` +
        `<span class="attemptTask"></span>` +
        `<span class="attemptMeta">${(a.duration_s || 0).toFixed(0)}s</span>`;
      head.querySelector('.attemptTask').textContent = a.task || '';
      row.appendChild(head);
      row.appendChild(verdictButtons(a.attempt_id, labels, a.subtask_plan));
      list.appendChild(row);
    }
  }

  const rolloutBody = document.getElementById('rolloutBody');
  const rolloutEmpty = document.getElementById('rolloutEmpty');
  const rollouts = s.recent_rollouts || [];
  rolloutBody.innerHTML = '';
  rolloutEmpty.style.display = rollouts.length ? 'none' : 'block';
  for (const r of rollouts) {
    const tr = document.createElement('tr');
    const t = new Date(r.timestamp * 1000).toLocaleTimeString();
    let verdict = LABEL_TEXT[r.annotation] || r.annotation || '';
    if (r.partial_subtask_name && r.subtask_plan) {
      verdict += ` (${r.partial_subtask_index}/${r.subtask_plan.subtasks.length}: ${r.partial_subtask_name})`;
    }
    tr.innerHTML = `<td>${t}</td><td></td><td></td><td></td><td>${(r.duration_s||0).toFixed(0)}s</td>`;
    tr.children[3].textContent = verdict;
    tr.children[1].textContent = r.experiment ? `${r.experiment.experiment_name} ${r.experiment.arm}` : '';
    tr.children[2].textContent = r.task || '';
    rolloutBody.appendChild(tr);
  }
}

function reloadFrame() {
  // In policy view the ZED is squashed to the dataset's 640x480 -- the ghost is only
  // comparable against that, so the live layer switches geometry with the overlay.
  const policy = document.getElementById('onionOn').checked ? '&policy=1' : '';
  const img = new Image();
  img.onload = function() { document.getElementById('frame').src = this.src; };
  img.src = '/frame.jpg?t=' + Date.now() + policy;
}

// -- Onion overlay --
let onionTasks = [];
let onionEpisodes = [];
let onionFps = 30;
let onionPlayTimer = null;
let onionIndexRetry = null;
let onionSelectSeq = 0;

function onionEls() {
  return {
    on: document.getElementById('onionOn'),
    controls: document.getElementById('onionControls'),
    task: document.getElementById('onionTask'),
    ep: document.getElementById('onionEp'),
    opacity: document.getElementById('onionOpacity'),
    blend: document.getElementById('onionBlend'),
    frame: document.getElementById('onionFrame'),
    frameVal: document.getElementById('onionFrameVal'),
    opacityVal: document.getElementById('onionOpacityVal'),
    status: document.getElementById('onionStatus'),
    warn: document.getElementById('onionWarn'),
    ghost: document.getElementById('ghost'),
    play: document.getElementById('onionPlay'),
  };
}

function applyGhostStyle() {
  const e = onionEls();
  e.ghost.style.opacity = (e.opacity.value / 100).toFixed(2);
  e.ghost.style.mixBlendMode = e.blend.value;
  e.opacityVal.textContent = e.opacity.value + '%';
}

function showGhostFrame() {
  const e = onionEls();
  const f = parseInt(e.frame.value, 10) || 0;
  e.frameVal.textContent = f;
  // No cache-buster: a given episode+frame never changes, and letting the browser keep it is
  // what makes scrubbing and looping playback smooth after the first pass.
  e.ghost.src = '/onion/frame.jpg?frame=' + f;
  // Only reveal the layer once it has a source -- shown earlier it is an empty <img>, which
  // some browsers draw as a broken-image icon over the live view.
  if (e.on.checked) e.ghost.style.display = 'block';
}

function checkTaskMatch() {
  const e = onionEls();
  const dsTask = e.task.options[e.task.selectedIndex] ? e.task.options[e.task.selectedIndex].dataset.name : '';
  const uiTask = document.getElementById('task').value.trim();
  if (dsTask && uiTask && dsTask !== uiTask) {
    e.warn.style.display = 'block';
    e.warn.textContent = 'Heads up: the policy is being prompted with "' + uiTask +
      '", which is not the task of the episode you are aligning to.';
  } else {
    e.warn.style.display = 'none';
  }
}

// The task box ships with a hardcoded default, and refreshStatus() deliberately never writes
// into it (that would fight anything being typed, the same way it would for replanSteps). So on
// a fresh page load it disagrees with whatever task the backend is actually carrying, which the
// onion overlay then auto-matches against and warns about. Seed it once at startup instead, and
// only while it still holds its own default -- if anything has been typed, leave it alone.
async function initTaskFromBackend() {
  const el = document.getElementById('task');
  try {
    const s = await (await fetch('/status')).json();
    if (s.task && el.value === el.defaultValue) el.value = s.task;
  } catch (e) { /* status not up yet -- the default is a fine starting point */ }
}

async function loadOnionTasks() {
  const res = await fetch('/onion/tasks');
  const data = await res.json();
  const e = onionEls();
  if (!data.ready) {
    e.status.textContent = 'Building the episode index from the dataset, one moment...';
    // One timer at a time -- toggling the overlay off and on while the index builds would
    // otherwise leave a retry running for every toggle.
    clearTimeout(onionIndexRetry);
    onionIndexRetry = setTimeout(loadOnionTasks, 1500);
    return;
  }
  onionFps = data.fps || 30;
  onionTasks = data.tasks.filter(t => t.n_episodes > 0);
  e.task.innerHTML = '';
  for (const t of onionTasks) {
    const opt = document.createElement('option');
    opt.value = t.task_index;
    opt.dataset.name = t.task;
    opt.textContent = t.task + '  (' + t.n_episodes + ' episodes)';
    e.task.appendChild(opt);
  }
  // Prefer the task the policy is actually being prompted with, if the dataset has it.
  const uiTask = document.getElementById('task').value.trim();
  const match = onionTasks.find(t => t.task === uiTask);
  if (match) e.task.value = match.task_index;
  e.status.textContent = '';
  await loadOnionEpisodes();
}

async function loadOnionEpisodes() {
  const e = onionEls();
  const res = await fetch('/onion/episodes?task_index=' + e.task.value);
  onionEpisodes = await res.json();
  e.ep.innerHTML = '';
  for (const ep of onionEpisodes) {
    const opt = document.createElement('option');
    opt.value = ep.episode_index;
    const secs = (ep.length / onionFps).toFixed(1);
    opt.textContent = 'episode ' + ep.episode_index + '  (' + secs + 's' +
      (ep.in_training_set ? ', in training set' : ', held out of training set') + ')';
    e.ep.appendChild(opt);
  }
  checkTaskMatch();
  if (onionEpisodes.length) await selectOnionEpisode();
}

async function selectOnionEpisode() {
  const e = onionEls();
  stopOnionPlayback();
  e.status.textContent = 'Loading episode frames...';
  // Clicking through episodes faster than they load would otherwise let an earlier response
  // land last and describe an episode the server no longer holds.
  const seq = ++onionSelectSeq;
  const res = await fetch('/onion/select', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({episode_index: parseInt(e.ep.value, 10)}),
  });
  const st = await res.json();
  if (seq !== onionSelectSeq) return;
  if (st.error) { e.status.textContent = 'Failed: ' + st.error; return; }
  e.frame.max = Math.max(0, st.length - 1);
  e.frame.value = 0;
  const missing = (st.cameras || []).filter(n => n === 0).length;
  e.status.textContent = st.length + ' frames' +
    (missing ? '  (' + missing + ' camera(s) unavailable for this episode)' : '');
  showGhostFrame();
}

function stopOnionPlayback() {
  if (onionPlayTimer) { clearInterval(onionPlayTimer); onionPlayTimer = null; }
  onionEls().play.textContent = 'Play';
}

function toggleOnionPlayback() {
  const e = onionEls();
  if (onionPlayTimer) { stopOnionPlayback(); return; }
  e.play.textContent = 'Pause';
  onionPlayTimer = setInterval(() => {
    const max = parseInt(e.frame.max, 10) || 0;
    let f = parseInt(e.frame.value, 10) || 0;
    f = f >= max ? 0 : f + 1;
    e.frame.value = f;
    showGhostFrame();
  }, 1000 / onionFps);
}

function setOnionEnabled(enabled) {
  const e = onionEls();
  e.controls.style.display = enabled ? 'block' : 'none';
  if (enabled) {
    applyGhostStyle();
    if (!onionTasks.length) loadOnionTasks(); else showGhostFrame();
  } else {
    stopOnionPlayback();
    e.ghost.style.display = 'none';
  }
  reloadFrame();  // switch the live layer into/out of policy geometry immediately
}

document.getElementById('onionOn').addEventListener('change', ev => setOnionEnabled(ev.target.checked));
document.getElementById('onionTask').addEventListener('change', loadOnionEpisodes);
document.getElementById('onionEp').addEventListener('change', selectOnionEpisode);
document.getElementById('onionOpacity').addEventListener('input', applyGhostStyle);
document.getElementById('onionFrame').addEventListener('input', () => { stopOnionPlayback(); showGhostFrame(); });
document.getElementById('onionFirst').addEventListener('click', () => {
  stopOnionPlayback();
  document.getElementById('onionFrame').value = 0;
  showGhostFrame();
});
document.getElementById('onionPlay').addEventListener('click', toggleOnionPlayback);
document.getElementById('onionBlend').addEventListener('change', ev => {
  // Difference only reads correctly at full strength -- a faded difference is just a dim image.
  const o = document.getElementById('onionOpacity');
  o.value = ev.target.value === 'difference' ? 100 : 35;
  applyGhostStyle();
});
document.getElementById('task').addEventListener('input', checkTaskMatch);

// -- Retrieval check: settings the loaded backend actually resolved, the latest retrieval, a short
// trend and pass/warn checks. Everything comes from the backend's own /retrieval.json (debug server,
// port 8082), so it describes what is running -- not what is selected in the dropdown above. --
let lastStatus = null;
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const CAM_NAME = { 'observation.images.zed': 'head', 'observation.images.fish0': 'left wrist', 'observation.images.fish1': 'right wrist',
                   zed: 'head', fish0: 'left wrist', fish1: 'right wrist' };

async function refreshRetrievalCheck() {
  const card = document.getElementById('contextCard');
  if (!lastStatus || !lastStatus.running) { card.style.display = 'none'; return; }
  let data;
  try {
    const res = await fetch(`http://${location.hostname}:8082/retrieval.json`, { cache: 'no-store' });
    data = await res.json();
  } catch (e) {
    return;  // backend still loading (debug server not up yet) -- keep the last view
  }
  if (!data.config) { card.style.display = 'none'; return; }  // loaded policy does no retrieval
  card.style.display = 'block';
  renderRetrievalCheck(data);
}

function renderRetrievalCheck(d) {
  const c = d.config, L = d.latest, exp = d.experiment;
  const f = (v, n = 3) => (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(n);
  const random = c.context_mode === 'random';
  const encDesc = c.context_encoding === 'ctx3cam'
    ? `ctx3cam: ${c.context_cameras.map(x => CAM_NAME[x] || x).join(' + ')} frames + ${c.context_text_max_length}-token task/state/action span`
    : `head frame + ${c.context_text_max_length}-token text`;
  const ctxDesc = c.context_mode === 'none' ? 'NONE — the policy gets no retrieved context'
    : random ? `RANDOM chunk of the task's pool, new draw every tick · ${encDesc}` : encDesc;
  const metricDesc = c.context_mode === 'none' ? '— (no retrieval)'
    : random ? `— (random draw, nothing ranked) · checkpoint trained with ${c.trained_metric}`
    : `${c.metric}${c.metric === 'vision_value' ? ` (λ = ${f(c.fusion_lambda, 2)})` : ''} · checkpoint trained with ${c.trained_metric}`;
  const rows = [
    ['Loaded config', c.config],
    ['Checkpoint', (c.checkpoint || '').split('/').slice(-3).join('/')],
    exp ? ['Experiment arm', `${exp.experiment_name} · ${exp.arm} (${exp.role}): ${exp.summary}`] : null,
    ['Context', ctxDesc],
    ['Retrieval metric', metricDesc],
    c.context_mode !== 'retrieved' || !/value/.test(c.metric) ? null : ['Progress estimator', c.value_estimator_label || c.value_estimator],
    c.action_interpolation ? ['Action interpolation', `w = exp(−λ · d / d_max), λ = ${c.interp_lamda} · checkpoint trained with λ = ${c.trained_interp_lamda}`] : null,
    c.context_mode !== 'retrieved' ? null : ['Vision keys · query', `${(c.key_cameras || []).map(x => CAM_NAME[x] || x).join(' + ')} · ${c.query_image_mode === 'auto' ? "each pool's own key" : c.query_image_mode} preprocessing`],
    c.context_mode === 'none' ? null : ['Pool directories', String(c.pool_dir || '').split(':').join(' · ')],
    c.pool_demos ? ['Demo buffer', `the ${c.pool_demos} shortest keep=true demonstration${c.pool_demos === 1 ? '' : 's'} of each task pool`] : null,
  ].filter(Boolean);
  document.getElementById('rcSettings').innerHTML = rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join('');

  const checks = [];
  const add = (level, text) => checks.push({ level, text });
  const age = L && L.time ? d.now - L.time : null;
  if (!L) add('warn', 'No retrieval yet — waiting for camera and state streams (the backend retrieves every tick, also in dry run).');
  else if (age > 3) add('bad', `No retrieval for ${age.toFixed(0)} s — streams stale or the backend is stuck; do not publish.`);
  else add('ok', `Retrieving every tick (last ${age.toFixed(1)} s ago, tick ${L.tick}).`);
  if (L && L.context_mode === 'none') add('info', 'Context disabled for this run: the policy runs on the live observation only.');
  if (L && L.context_mode !== 'none') {
    if (!L.hit) add('bad', `No retrieval pool for the current query “${esc(L.prompt)}” (looked for ${esc((L.pool_path || '').split('/').pop())}). Use the pool task's wording or the trained prompt.`);
    else {
      const poolKind = /indomain/.test(L.pool_path || '') ? 'in-domain pool (training split)' : /demo/.test(L.pool_path || '') ? 'demo pool' : 'pool';
      add('ok', `${poolKind[0].toUpperCase() + poolKind.slice(1)} found for “${esc(L.pool_task)}”: ${L.num_candidates} candidate chunks.`);
      if (L.pool_key_image_mode && L.query_image_mode !== L.pool_key_image_mode) add('bad', `Query frames are ${L.query_image_mode}-preprocessed but this pool's keys are ${L.pool_key_image_mode}: vision distances are not comparable.`);
      if (L.chunk_task && L.chunk_task !== L.pool_task) add('bad', `Retrieved chunk is from a different task (“${esc(L.chunk_task)}”).`);
      if (L.prompt !== d.task) add('warn', `Latest retrieval used an older query (“${esc(L.prompt)}”); waiting for the next tick.`);
      if (L.context_encoding !== c.context_encoding) add('bad', `Context encoding ${L.context_encoding} does not match the checkpoint's ${c.context_encoding}.`);
      else add('ok', `Context encoding matches the checkpoint (${c.context_encoding === 'ctx3cam' ? '3 camera frames + FAST span' : 'head frame + text'}).`);
      if (L.context_tokens_used != null) {
        if (c.context_encoding === 'ctx3cam' && L.context_tokens_used >= L.context_text_max_length) add('warn', `Context span hit the ${L.context_text_max_length}-token limit (truncated).`);
        else if (c.context_encoding === 'ctx3cam') add('ok', `Context span uses ${L.context_tokens_used}/${L.context_text_max_length} tokens (not truncated).`);
      }
      if (L.context_mode === 'random') {
        const drawn = (d.history || []).filter(x => x.hit && x.context_mode === 'random');
        const distinct = new Set(drawn.map(x => `${x.episode}/${x.start_frame}`)).size;
        add('info', `Random context: each tick draws a chunk uniformly from the ${L.num_candidates} candidates of this task's pool${exp && exp.context_mode === 'random' ? ` — intended by ${exp.experiment_name} ${exp.arm}` : ''}. ${distinct} different chunks in the last ${drawn.length} ticks.`);
      } else if (L.metric !== c.trained_metric) {
        if (exp && exp.retrieval_metric === L.metric) add('info', `Retrieval metric ${L.metric} differs from training (${c.trained_metric}) — intended by ${exp.experiment_name} ${exp.arm}.`);
        else add('warn', `Retrieval metric ${L.metric} differs from what this checkpoint was trained with (${c.trained_metric}).`);
      } else add('ok', `Retrieval metric ${L.metric} matches training.`);
      if (L.metric === 'value' || L.metric === 'vision_value') {
        const est = ESTIMATOR_NAME[L.value_estimator] || 'Robometer';
        if (L.query_value == null) add('bad', `Value retrieval but no progress estimate (${est}).`);
        else add('ok', `${est} progress estimate ${f(L.query_value)}` +
          (L.value_estimator === 'vfe' ? ` (conditioned on demo episode ${L.value_context_episode}, ${f(L.value_ms, 0)} ms).`
            : L.value_estimator === 'vfe_recap' ? ` (no context, ${f(L.value_ms, 0)} ms).` : '.'));
      } else if (L.context_mode === 'retrieved' && ['vfe', 'vfe_recap', 'robometer_ft'].includes(c.value_estimator)) add('info', 'A progress estimator is selected, but vision-only retrieval does not use progress estimates.');
      if (c.action_interpolation) {
        if (c.interp_lamda !== c.trained_interp_lamda) add('warn', `Interpolation λ = ${c.interp_lamda} overrides the trained λ = ${c.trained_interp_lamda} (diagnostic run).`);
        if (L.interp) add('info', `Interpolation weight w = ${f(L.interp.exp_lamda_distance)} (rank-1 distance ${f(L.interp.top1_distance, 1)} / pool max ${f(L.interp.max_distance, 1)} = ${f(L.interp.normalized_distance)}; neighbor episode ${L.interp.nearest_episode}, frame ${L.interp.nearest_start_frame}).`);
        else add('bad', 'This is an interpolation checkpoint but the latest retrieval carries no interpolation weight.');
      }
      if (c.pool_demos) {
        const P = L.pool_demos;
        if (!P) add('bad', `This run should keep only ${c.pool_demos} demonstrations, but the latest retrieval used the whole pool.`);
        else {
          const eps = P.episodes.map(x => `${x.episode} (${x.length} frames)`).join(', ');
          add('ok', `Demo buffer: ${P.episodes.length} of the pool's ${P.available} demonstrations — episode${P.episodes.length === 1 ? '' : 's'} ${eps}; ${L.num_candidates} candidate chunks.`);
          if (P.available < P.requested) add('warn', `This pool has only ${P.available} demonstrations, fewer than the ${P.requested} this arm asks for: the buffer is the whole pool, the same as any arm with ${P.available} or more.`);
          if (P.dropped_keep_false && P.dropped_keep_false.length) add('info', `Failed (keep=false) pool episodes ${P.dropped_keep_false.join(', ')} are never counted as demonstrations.`);
          if (L.value_estimator === 'vfe' && L.value_context_episode != null && !P.episodes.some(x => x.episode === L.value_context_episode))
            add('warn', `The IC-VFE progress estimate is conditioned on demo episode ${L.value_context_episode}, which is outside this ${P.episodes.length}-demo buffer (the estimator's context is not restricted). Robometer or the RECAP VFE use no demonstration.`);
        }
      } else if (L.pool_demos) add('bad', 'The latest retrieval restricted the demo buffer, but this run was launched with the whole pool.');
      if (L.retrieval_ms > 1500) add('warn', `Retrieval is slow (${L.retrieval_ms.toFixed(0)} ms per tick).`);
    }
    const trained = Object.entries(c.prompt_to_task_overrides || {}).find(([prompt, task]) => task === d.task);
    if (trained) add('warn', `This checkpoint was trained with the prompt “${esc(trained[0])}” for this task; consider using it.`);
  }
  document.getElementById('rcChecks').innerHTML = checks.map(x => `<li class="${x.level}"><span class="i">${{ok: '✓', warn: '!', bad: '✗', info: 'i'}[x.level]}</span>${x.text}</li>`).join('');
  const worst = checks.some(x => x.level === 'bad') ? 'bad' : checks.some(x => x.level === 'warn') ? 'warn' : 'ok';
  const pill = document.getElementById('rcPill');
  pill.className = `rcPill ${L ? worst : 'warn'}`;
  pill.textContent = !L ? 'waiting' : worst === 'ok' ? 'all checks pass' : worst === 'warn' ? 'check warnings' : 'problem';

  const live = document.getElementById('rcLive');
  if (!L) live.textContent = 'Waiting for the first retrieval…';
  else if (L.context_mode === 'none') live.innerHTML = `Tick ${L.tick} · inference ${f(L.infer_ms, 0)} ms · no context`;
  else if (!L.hit) live.innerHTML = `Tick ${L.tick} · <b>no pool for this query</b>`;
  else live.innerHTML = `Tick ${L.tick} · retrieval ${f(L.retrieval_ms, 0)} ms · inference ${f(L.infer_ms, 0)} ms<br>
    ${L.context_mode === 'random' ? 'Randomly drew' : 'Retrieved'} <b>episode ${L.episode}</b>, frames ${L.start_frame}–${L.end_frame} · ${L.context_mode === 'random' ? 'no score (random)' : `score ${f(L.score)}`}` +
    `${L.query_value != null ? ` · query progress <b>${f(L.query_value)}</b>` : ''}` +
    `${L.context_tokens_used != null ? ` · context ${L.context_tokens_used}/${L.context_text_max_length} tokens` : ''}`;
  renderRetrievalTrend(d.history || []);
}

function renderRetrievalTrend(history) {
  const svg = document.getElementById('rcTrend'), legend = document.getElementById('rcTrendLegend');
  const h = history.filter(x => x.hit);
  if (h.length < 2) { svg.innerHTML = ''; legend.textContent = ''; return; }
  const W = 600, H = 92, n = h.length, X = i => i / (n - 1) * W;
  const line = (vals, y0, y1, color) => {
    const fin = vals.filter(v => v != null && isFinite(v));
    if (!fin.length) return '';
    let lo = Math.min(...fin), hi = Math.max(...fin); if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pts = vals.map((v, i) => v == null ? null : `${X(i).toFixed(1)},${(y1 - (v - lo) / (hi - lo) * (y1 - y0)).toFixed(1)}`).filter(Boolean);
    return `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
  };
  let changes = 0, marks = '';
  for (let i = 1; i < n; i++) if (h[i].episode !== h[i - 1].episode || h[i].start_frame !== h[i - 1].start_frame) {
    changes++; marks += `<line x1="${X(i)}" x2="${X(i)}" y1="0" y2="${H}" stroke="#333" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
  }
  const hasValue = h.some(x => x.query_value != null);
  svg.innerHTML = marks + line(h.map(x => x.score), 4, hasValue ? 44 : 88, '#6aa9ff') + (hasValue ? line(h.map(x => x.query_value), 50, 88, '#7ee0b0') : '');
  legend.innerHTML = `Last ${n} ticks: <span style="color:#6aa9ff">retrieval score</span>${hasValue ? ' · <span style="color:#7ee0b0">query progress</span>' : ''} · grey lines = retrieved chunk changed (${changes}×)`;
}

function reloadContextFrame() {
  const card = document.getElementById('contextCard');
  if (card.style.display === 'none') return;  // not a VICTR checkpoint -- skip the fetch entirely
  const img = new Image();
  img.onload = function() { document.getElementById('contextFrame').src = this.src; };
  img.onerror = function() { /* backend not up yet, or viewer port not reachable -- just retry next tick */ };
  img.src = `http://${location.hostname}:8082/context.jpg?t=` + Date.now();
}

loadCheckpoints();
loadExperiments();
refreshStatus();
setInterval(loadExperiments, 10000);
updatePublishToggleStyle();
setInterval(loadCheckpoints, 5000);
setInterval(refreshStatus, 1000);
setInterval(reloadFrame, 800);
setInterval(reloadContextFrame, 800);
setInterval(refreshRetrievalCheck, 1000);
applyGhostStyle();

// ?onion=1 opens straight into the alignment view -- worth bookmarking on the tablet you use
// at the robot, since that is the one thing you do before every rollout. It waits for the task
// seed so the overlay auto-selects the dataset task the backend is actually running.
initTaskFromBackend().then(() => {
  checkTaskMatch();
  if (new URLSearchParams(location.search).get('onion') === '1') {
    document.getElementById('onionOn').checked = true;
    setOnionEnabled(true);
  }
});
</script>
</body>
</html>
"""


def make_handler(backend: Backend, ghost: onion.Onion):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            # Match on the path alone: the page takes query parameters now (?onion=1), and an
            # exact self.path comparison 404s the moment one is present.
            route = urllib.parse.urlparse(self.path).path
            if route == "/":
                data = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif route == "/checkpoints":
                self._send_json(discover_checkpoints())
            elif route == "/experiments":
                self._send_json(experiments_view())
            elif route == "/status":
                self._send_json(backend.status())
            elif route == "/frame.jpg":
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                data = make_composite_jpeg(policy_view=q.get("policy", ["0"])[0] == "1")
                self._send_jpeg(data, cache=False)
            elif route.startswith("/onion/"):
                self._do_onion_get(urllib.parse.urlparse(self.path))
            else:
                self.send_response(404); self.end_headers()

        def _send_jpeg(self, data, cache: bool):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            # Ghost frames are immutable for a given episode+frame, so letting the browser keep
            # them is what makes scrubbing back and forth, and looping playback, instant. The
            # live frame must never be cached.
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _do_onion_get(self, url):
            q = urllib.parse.parse_qs(url.query)
            if url.path == "/onion/tasks":
                self._send_json(ghost.tasks())
            elif url.path == "/onion/episodes":
                self._send_json(ghost.episodes(int(q.get("task_index", ["-1"])[0])))
            elif url.path == "/onion/status":
                self._send_json(ghost.status())
            elif url.path == "/onion/frame.jpg":
                data = ghost.frame_jpeg(int(q.get("frame", ["0"])[0]))
                if data is None:
                    self.send_response(404); self.end_headers(); return
                self._send_jpeg(data, cache=True)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/start":
                log.info("POST /start from %s: %s", self.client_address[0], json.dumps(body))

                def _optional_int(key):
                    v = body.get(key)
                    if v is None or v == "":
                        return None
                    return int(v)

                def _optional_float(key):
                    v = body.get(key)
                    if v is None or v == "":
                        return None
                    return float(v)

                try:
                    backend.start(
                        checkpoint_path=body.get("path", ""),
                        label=body.get("label", body.get("path", "")),
                        config_name=body.get("config_name"),
                        task=body.get("task", DEFAULT_TASK),
                        family=body.get("family", "openpi"),
                        publish=bool(body.get("publish", False)),
                        replan_steps=_optional_int("replan_steps"),
                        action_horizon=_optional_int("action_horizon"),
                        retrieval_lambda=_optional_float("retrieval_lambda"),
                        retrieval_metric=body.get("retrieval_metric") or None,
                        retrieval_key_cameras=body.get("retrieval_key_cameras") or None,
                        value_estimator=body.get("value_estimator") or None,
                        interp_lamda=_optional_float("interp_lamda"),
                        experiment_arm=body.get("experiment_arm") or None,
                        rtc=bool(body.get("rtc", False)),
                        gripper_sigmoid_gain=_optional_float("gripper_sigmoid_gain"),
                        gripper_sigmoid_center=_optional_float("gripper_sigmoid_center"),
                    )
                    self._send_json(backend.status())
                except Exception as e:
                    log.exception("Start failed")
                    self._send_json({"error": str(e)}, status=400)
            elif self.path == "/stop":
                backend.stop()
                self._send_json(backend.status())
            elif self.path == "/task":
                backend.push_task(body.get("task", ""))
                self._send_json(backend.status())
            elif self.path == "/onion/select":
                try:
                    self._send_json(ghost.select(int(body["episode_index"])))
                except Exception as e:
                    log.exception("Onion select failed")
                    self._send_json({"error": str(e)}, status=400)
            elif self.path == "/annotate":
                # rollout_id accepted as a fallback for anything still posting the pre-attempts
                # payload shape.
                ok = backend.annotate(body.get("attempt_id") or body.get("rollout_id", ""),
                                      body.get("label", ""),
                                      body.get("partial_subtask_index"))
                self._send_json({"ok": ok})
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-port", type=int, default=8090)
    parser.add_argument("--cam-host", default="192.168.123.116")
    parser.add_argument("--cam-port", type=int, default=5550)
    parser.add_argument("--action-port", type=int, default=5572)
    parser.add_argument("--task-control-port", type=int, default=5573)
    parser.add_argument("--onion-dataset", type=pathlib.Path, default=onion.DEFAULT_DATASET_ROOT,
                        help="Dataset the onion overlay draws its reference episodes from. Needs "
                             "the v3.0 layout with videos; the index over it is built once into "
                             "assets/onion_index.json.")
    args = parser.parse_args()

    backend = Backend(task_control_port=args.task_control_port, action_port=args.action_port,
                      cam_host=args.cam_host, cam_port=args.cam_port)
    ghost = onion.Onion(args.onion_dataset.expanduser(), OPENPI_PYTHON)
    ghost.start()

    # Without this, killing this UI process (SIGTERM, e.g. `pkill`) leaves any backend
    # subprocess it had launched as an orphan -- reparented to init, invisible to /status
    # from that point on (a fresh Backend object after restart starts with self.proc=None,
    # with zero memory of it), yet still bound to its ports and, if --publish was set, still
    # potentially controlling the robot. Confirmed in practice 2026-09-06: a restart raced
    # against a run being started via the browser in the check-status-then-kill window left
    # exactly this kind of invisible orphan alive holding the action-publish port, which then
    # made the NEXT run fail immediately with "Address already in use" instead of ever
    # starting. Backend.stop() itself already does the right thing (terminate, wait, kill on
    # timeout) -- it just was never being called on shutdown.
    def _handle_shutdown_signal(signum, frame):
        log.info("Received signal %d, stopping backend before exit...", signum)
        backend.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    threading.Thread(target=_camera_reader_thread, args=(args.cam_host, args.cam_port), daemon=True).start()

    import http.server
    handler = make_handler(backend, ghost)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.ui_port), handler)
    log.info("YOR inference UI -> http://localhost:%d  (or http://%s:%d)",
              args.ui_port, _lan_ip(), args.ui_port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
