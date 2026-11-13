# Reproducing the released VFE models

## Data

Set paths in `configs/vfe/vfe_data_splits.json` for training and in the `evaluation`
section of `configs/vfe/paper_experiments.json` for evaluation.

| Resource | Required contents |
| --- | --- |
| Training dataset | LeRobot v2.1 `icl-dataset-keep-true`: metadata, parquet records, and three-camera videos |
| Training annotations | Dense RoboDopamine frame labels keyed by episode UID |
| Evaluation dataset | Original `icl-demo-dataset-keep-true` with 640×480 videos |
| Evaluation annotations | RoboDopamine frame references used during prediction generation |
| Scoring annotations | Dense human annotation directory or ZIP |

The checked-in training split contains 2,134 episodes over 36 tasks: 1,704 query
and 430 context episodes, disjoint within each task. The 80/20 division is a
query/context training partition, not a held-out test split.

The packaged evaluation configuration covers 240 successful trajectories over
27 tasks, with 120 seen and 120 unseen queries. Each query is evaluated at every
frame. Context models use every other same-task episode as a reference, excluding
the query itself. Context-free models evaluate each query once. Keep the original
video resolution and frame indexing.

Training percentages are clipped to [0, 100], mapped to [-1, 0], and discretized
into 201 bins. The linear pretraining preset instead maps episode position to
[-1, 0]. Generated labels are separated by semantics under `value_targets/`.
Progress labels are not provided as model context inputs.

Data and annotations are not included in checkpoint packages. Public download
URLs for these exact exports, and the teacher checkpoint/prompt/version used to
produce the labels, have not been established in this release.

## Training

The main model uses global batch 320 on eight GPUs. Its final training segment
resumes the step-6,800 checkpoint:

```bash
python scripts/vfe/paper.py ic_vfe train \
  --resume /path/to/checkpoint_step_6800_bs320_cursor26.pt --execute
```

The launcher stops at step 10,000. The preceding linear-label stage is available
through `preprocess-pretrain` and `pretrain` (30,000 steps, global batch 160).
The full annotation-finetuning and cursor-conversion lineage before step 6,800
has not been reconstructed as a complete from-scratch recipe. Use the released
final checkpoint for inference; exact continuation also needs the original
training artifacts and compatible worker count.

No-context and RECAP use global batch 160 on two GPUs. The pending no-block
launcher uses global batch 160 on four GPUs and a planned 20,000-step schedule.
For a controlled block-embedding comparison, match the initialization and
training schedule to the embedding-enabled control. `--gpus N` preserves the
global batch when N divides it.

To evaluate the pending ablation after training, supply its new checkpoint and
prediction directory explicitly:

```bash
python scripts/vfe/paper.py ic_vfe_no_block_embedding evaluate --checkpoint /path/to/checkpoint.pt --execute
python scripts/vfe/paper.py ic_vfe_no_block_embedding score --source /path/to/raw_predictions --execute
```

## Scoring

`score_paper.py` validates frame alignment, query/context coverage, model preset,
and checkpoint consistency. It scores raw saved predictions against human
annotations, averages over contexts within each query, and gives each query
equal weight. EMA, when configured, is applied once per trajectory.

The active conventions are recorded per experiment in `configs/vfe/paper_experiments.json`.
The archived configurations use causal EMA(0.5) for IC-VFE and the pending
no-block model; no-context and RECAP use raw predictions. IC-VFE and RECAP use
raw float64 human percentages and exclude undefined correlations (119 scored
queries per split); no-context uses clipped float32 values and assigns zero to
undefined correlations (120 per split).

These archived conventions differ from the raw-score rows in
[arXiv v1, Table I](https://arxiv.org/html/2611.08698v1#S4.T1).
The local dataset contract above also differs from the 16-task suite described
there. Exact correspondence to that evaluation suite requires its episode/task
manifest. The released pure no-block configuration has no measured result;
the paper's reported no-block row is not assigned to this untrained configuration.

## Release coverage

- Included: four model configurations; three checkpoint packages; data
  preprocessing; frozen feature caching; training/resume; dense evaluation;
  saved-prediction scoring; CPU regression tests.
- Not included: training/evaluation datasets and annotations, a trained pure
  no-block checkpoint, teacher-label generation, or VLA policy training/rollouts.
- Still needed for full reproduction: public data/annotation locations, the
  complete main-model training lineage, and the exact arXiv evaluation manifest.

Model implementations retain shared checkpoint-compatible code. The four
published presets define the supported experiments; internal alternatives are
not additional released experiments. Research-only runs and policy submodules
are kept outside this branch.
