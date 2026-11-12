"""Deployment-time ablation experiments for VICTR retrieval context (EXP A, EXP B, EXP E).

Single source of truth for what each experiment arm loads and how it is deployed. Read by
inference.py (the launch UI, which runs in a venv without numpy/JAX -- keep this module
stdlib-only), and by retrieval_viewer_server.py. The inference server itself only receives the
resolved flags (--config/--checkpoint/--retrieval-metric/--context-mode) plus the
--experiment/--experiment-arm labels it records into every rollout's meta.json.

EXP A -- does the retrieval metric the policy was TRAINED with matter?
  Swap the metric at deployment only: a VICTR-value checkpoint retrieves by DINOv2 vision
  similarity, and a VICTR-vision checkpoint retrieves by Robometer progress (value) similarity.
  Each treatment is paired with the same checkpoint retrieving as it was trained (control).

EXP B -- does the context itself matter, in either direction?
  A context-trained VLA (VICTR-value) deployed with NO context, and a VLA trained without
  context (base pi0.5) deployed WITH a retrieved demonstration as context. Each treatment is
  paired with the same checkpoint in its trained condition (control). B3 keeps VICTR-value's context
  but replaces retrieval with a uniformly random chunk of the same task's pool, drawn afresh every
  inference (context_mode "random"): does it matter WHICH demonstration chunk the context holds?

2026-09-15: EXP A and EXP B1 arms were re-pointed from the deleted expanded checkpoints to the canonical
ctx3cam step-10000 checkpoints on disk (same metric/context design); A2 additionally fixes the IC-VFE
progress estimator. EXP B2/B2c were then moved to the canonical warmstart baseline pi0.5 (step 10000),
the plain control arm of the ctx3cam VICTR runs: B2 injects a vision-retrieved demonstration through
yor_icl_pi05_canonical_warmstart_baseline_with_context, B2c runs it as trained. Later the same day
B1/B1c moved from VICTR-vision to VICTR-value ctx3cam (step 10000), so B1 is the same deployment as
EXP E0 (B1c retrieves by value as trained, from the whole pool). No B1/B1c attempt had been annotated.

EXP E -- how many demonstrations does the retrieval buffer need?
  VICTR-value ctx3cam (step 10000), retrieving by value as trained, with only the N shortest
  keep=true demonstrations of the task pool as candidates (pool_demos, server --pool-demos):
  N = 10 (control), 5, 3, 1, and no context at all (0, the same deployment as EXP B1). A pool with fewer than N demonstrations keeps
  them all, so on the 5-demo in-domain pools E10 and E5 are the same buffer (the retrieval check says so).
  The IC-VFE progress estimator conditions on its own demo regardless of N; Robometer and the
  RECAP VFE use none.

Arms differ from their control in exactly one deployment setting; everything else (pool, query
preprocessing, key cameras, replan steps) is whatever the operator sets for both. Compare an arm
only with its own control, on the same tasks and start poses.
"""

EXPERIMENTS = {
    "EXP_A": {
        "name": "EXP A",
        "title": "Retrieval metric swap at deployment",
        "question": "Does the retrieval metric used during training matter at deployment?",
        "arms": {
            "A1": {
                "role": "treatment",
                "control": "A1c",
                "summary": "VICTR-value ctx3cam (step 10000), retrieving by vision (DINOv2) similarity",
                "trained": "value-retrieved context",
                "deployed": "vision-retrieved context",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "vision",
                "context_mode": "retrieved",
            },
            "A1c": {
                "role": "control",
                "control": None,
                "summary": "VICTR-value ctx3cam (step 10000), retrieving by value similarity (as trained)",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
            },
            "A2": {
                "role": "treatment",
                "control": "A2c",
                "summary": "VICTR-vision ctx3cam (step 10000), retrieving by value (IC-VFE progress) similarity",
                "trained": "vision-retrieved context",
                "deployed": "value-retrieved context",
                "config": "yor_icl_victr_vision_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_vision_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_vision_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                # IC-VFE only (2026-09-15): value retrieval scored by the lair IC-VFE, never Robometer.
                "value_estimator": "vfe",
                "context_mode": "retrieved",
            },
            "A2c": {
                "role": "control",
                "control": None,
                "summary": "VICTR-vision ctx3cam (step 10000), retrieving by vision similarity (as trained)",
                "trained": "vision-retrieved context",
                "deployed": "vision-retrieved context",
                "config": "yor_icl_victr_vision_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_vision_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_vision_canonical_ctx3cam-step10000",
                "retrieval_metric": "vision",
                "context_mode": "retrieved",
            },
        },
    },
    "EXP_B": {
        "name": "EXP B",
        "title": "Context ablation / injection at deployment",
        "question": "What happens when a context-trained VLA gets no context, "
                    "or a VLA trained without context gets a demonstration?",
        "arms": {
            "B1": {
                "role": "treatment",
                "control": "B1c",
                "summary": "VICTR-value ctx3cam (step 10000) with NO context",
                "trained": "value-retrieved context",
                "deployed": "no context (query only)",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": None,
                "context_mode": "none",
            },
            "B1c": {
                "role": "control",
                "control": None,
                "summary": "VICTR-value ctx3cam (step 10000) with value-retrieved context (as trained)",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
            },
            "B3": {
                "role": "treatment",
                "control": "B1c",
                "summary": "VICTR-value ctx3cam (step 10000) with a random chunk of the same task's pool as context",
                "trained": "value-retrieved context",
                "deployed": "random same-task context (new draw every inference)",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": None,
                "context_mode": "random",
            },
            "B2": {
                "role": "treatment",
                "control": "B2c",
                "summary": "Canonical warmstart baseline pi0.5 (step 10000, trained without context) given a vision-retrieved demonstration",
                "trained": "no context",
                "deployed": "vision-retrieved context (injected)",
                # Same base weights, loaded into the VICTR model class so it can take context
                # blocks; its neighbor-rank embedding (absent from the checkpoint) is zeros. See
                # openpi.models.pi0_victr.Pi0VictrConfig.missing_rank_embedding.
                "config": "yor_icl_pi05_canonical_warmstart_baseline_with_context",
                "checkpoint": "checkpoints/yor_icl_pi05_canonical_warmstart_baseline-step10000",
                "hf_repo": "lair-nyu/yor_icl_pi05_canonical_warmstart_baseline-step10000",
                "retrieval_metric": "vision",
                "context_mode": "retrieved",
            },
            "B2c": {
                "role": "control",
                "control": None,
                "summary": "Canonical warmstart baseline pi0.5 (step 10000) with no context (as trained)",
                "trained": "no context",
                "deployed": "no context",
                "config": "yor_icl_pi05_canonical_warmstart_baseline",
                "checkpoint": "checkpoints/yor_icl_pi05_canonical_warmstart_baseline-step10000",
                "hf_repo": "lair-nyu/yor_icl_pi05_canonical_warmstart_baseline-step10000",
                "retrieval_metric": None,
                "context_mode": None,
            },
        },
    },
    "EXP_E": {
        "name": "EXP E",
        "title": "Demonstration buffer size at deployment",
        "question": "How many demonstrations does VICTR-value need in its retrieval buffer "
                    "(the N shortest of the task pool: 0, 1, 3, 5, 10)?",
        "arms": {
            "E0": {
                "role": "treatment",
                "control": "E10",
                "summary": "VICTR-value ctx3cam (step 10000) with NO context (0 demos in the buffer; same deployment as B1)",
                "trained": "value-retrieved context",
                "deployed": "no context (query only)",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": None,
                "context_mode": "none",
                "pool_demos": None,
            },
            "E1": {
                "role": "treatment",
                "control": "E10",
                "summary": "VICTR-value ctx3cam (step 10000), value-retrieved context from the shortest demo only",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context, the shortest demo only",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
                "pool_demos": 1,
            },
            "E3": {
                "role": "treatment",
                "control": "E10",
                "summary": "VICTR-value ctx3cam (step 10000), value-retrieved context from the 3 shortest demos",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context, 3 shortest demos in the buffer",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
                "pool_demos": 3,
            },
            "E5": {
                "role": "treatment",
                "control": "E10",
                "summary": "VICTR-value ctx3cam (step 10000), value-retrieved context from the 5 shortest demos",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context, 5 shortest demos in the buffer",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
                "pool_demos": 5,
            },
            "E10": {
                "role": "control",
                "control": None,
                "summary": "VICTR-value ctx3cam (step 10000), value-retrieved context from the 10 shortest demos (the default buffer size)",
                "trained": "value-retrieved context",
                "deployed": "value-retrieved context, 10 shortest demos in the buffer",
                "config": "yor_icl_victr_value_canonical_ctx3cam_serve",
                "checkpoint": "checkpoints/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "hf_repo": "lair-nyu/yor_icl_victr_value_canonical_ctx3cam-step10000",
                "retrieval_metric": "value",
                "context_mode": "retrieved",
                "pool_demos": 10,
            },
        },
    },
}

# Server-side retrieval flags an arm fixes. The launch UI must not also send its own values
# for these, or the arm would silently stop being what its label says.
# value_estimator is optional per arm (arm.get): absent/None means the operator's choice stands.
# pool_demos (EXP E) is optional too: absent/None keeps every demonstration in the task pool.
ARM_FIXED_SETTINGS = ("retrieval_metric", "context_mode", "value_estimator", "pool_demos")


def find_arm(arm_id: str) -> tuple[str, dict, dict]:
    """(experiment_id, experiment, arm) for an arm id such as "A1" or "B2c"."""
    for exp_id, exp in EXPERIMENTS.items():
        if arm_id in exp["arms"]:
            return exp_id, exp, exp["arms"][arm_id]
    raise KeyError(f"Unknown experiment arm: {arm_id!r}")


def arm_label(arm_id: str) -> str:
    _, exp, arm = find_arm(arm_id)
    return f"{exp['name']} · {arm_id} ({arm['role']}): {arm['summary']}"


def arm_record(arm_id: str) -> dict:
    """The experiment block stored with every rollout and annotation for this arm."""
    exp_id, exp, arm = find_arm(arm_id)
    return {
        "experiment": exp_id,
        "experiment_name": exp["name"],
        "arm": arm_id,
        "role": arm["role"],
        "control_arm": arm["control"],
        "summary": arm["summary"],
        "trained": arm["trained"],
        "deployed": arm["deployed"],
        "config": arm["config"],
        "retrieval_metric": arm["retrieval_metric"],
        "context_mode": arm["context_mode"],
        "value_estimator": arm.get("value_estimator"),
        "pool_demos": arm.get("pool_demos"),
    }


def validate() -> None:
    """Structural checks: every treatment names an existing control on the same checkpoint that
    differs only in a fixed deployment setting."""
    for exp_id, exp in EXPERIMENTS.items():
        for arm_id, arm in exp["arms"].items():
            if arm["role"] not in ("treatment", "control"):
                raise ValueError(f"{exp_id}/{arm_id}: bad role {arm['role']!r}")
            if arm["context_mode"] not in (None, "retrieved", "random", "none"):
                raise ValueError(f"{exp_id}/{arm_id}: bad context_mode {arm['context_mode']!r}")
            if arm["retrieval_metric"] not in (None, "vision", "value", "vision_value"):
                raise ValueError(f"{exp_id}/{arm_id}: bad retrieval_metric {arm['retrieval_metric']!r}")
            if (arm["context_mode"] == "retrieved") != (arm["retrieval_metric"] is not None):
                raise ValueError(f"{exp_id}/{arm_id}: retrieved context needs a metric, and only it")
            estimator = arm.get("value_estimator")
            if estimator not in (None, "robometer", "robometer_ft", "vfe", "vfe_recap"):
                raise ValueError(f"{exp_id}/{arm_id}: bad value_estimator {estimator!r}")
            if estimator is not None and arm["retrieval_metric"] not in ("value", "vision_value"):
                raise ValueError(f"{exp_id}/{arm_id}: a progress estimator is only meaningful for value retrieval")
            demos = arm.get("pool_demos")
            if demos is not None and (isinstance(demos, bool) or not isinstance(demos, int) or demos < 1):
                raise ValueError(f"{exp_id}/{arm_id}: pool_demos must be a positive integer or None")
            if demos is not None and arm["context_mode"] not in ("retrieved", "random"):
                raise ValueError(f"{exp_id}/{arm_id}: pool_demos needs context from the pool")
            if arm["role"] == "treatment":
                control = exp["arms"].get(arm["control"])
                if control is None or control["role"] != "control":
                    raise ValueError(f"{exp_id}/{arm_id}: control {arm['control']!r} missing")
                if control["checkpoint"] != arm["checkpoint"]:
                    raise ValueError(f"{exp_id}/{arm_id}: control must use the same checkpoint")
                if all(control.get(k) == arm.get(k) for k in ARM_FIXED_SETTINGS) and control["config"] == arm["config"]:
                    raise ValueError(f"{exp_id}/{arm_id}: identical to its control")


validate()
