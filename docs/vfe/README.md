# VICTR — In-Context Value Function Estimation

Code and checkpoints for the value-function estimation component of
**VICTR: Value-Informed In-Context Learning for Vision-Language-Action Models**.

[Paper](https://arxiv.org/abs/2611.08698) · [Project](https://ai4ce.github.io/VICTR/) · [Reproduction guide](reproduction.md)

IC-VFE estimates task progress from three camera views, proprioception, a task
instruction, and a reference demonstration. This repository provides data
preprocessing, context-feature caching, training, evaluation, and prediction
scoring. VLA policy training and robot rollout code are separate from this release.

## Models

| Experiment | Model | Checkpoint |
| --- | --- | --- |
| `ic_vfe` | Gemma native vision + Gemma 270M, 177 context slots | [10K](https://huggingface.co/Lewis31231/vfe-gemma270m-dualres-context177) |
| `ic_vfe_no_context` | Same architecture, current observation only | [20K](https://huggingface.co/Lewis31231/vfe-gemma270m-dualres-3cam-no-context) |
| `ic_vfe_no_block_embedding` | IC-VFE with block-index embeddings disabled | Pending training |
| `recap` | SigLIP So400M patch14-384 + Gemma 270M | [20K](https://huggingface.co/Lewis31231/vfe-gemma270m-recap-style-3cam-no-context) |

The no-block model changes only `context_slot_embedding_enabled`. It has no
released checkpoint or measured results. All models use `zed`, `fish0`, `fish1`
and `observation.state`; the two context models share the same feature cache.

## Installation

Use Python 3.10+ and install a PyTorch build compatible with your CUDA runtime.
Run the following commands from the repository root:

```bash
git clone --branch IC-VFE --single-branch https://github.com/ai4ce/VICTR.git
cd VICTR
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The recorded dependency versions are in `requirements.txt`. Access to the
Gemma pretrained backbones is required. GPU training and inference require
suitable GPU hardware; cluster users should use their scheduler. W&B is disabled
by the common launcher; set `WANDB_MODE=online` to enable it.

## Download and evaluate

```bash
python scripts/vfe/download_checkpoints.py all
python scripts/vfe/download_checkpoints.py all --verify-only
```

Downloads use immutable Hugging Face revisions and SHA256 verification. Each
package includes model parameters, optimizer/RNG state, model configuration,
and a portable `run_config.json`. Weights are stored outside Git.

Set the dataset and annotation paths as described in the [data guide](reproduction.md#data).
Then evaluate and score a released model:

```bash
python scripts/vfe/paper.py ic_vfe validate-data --execute
python scripts/vfe/paper.py ic_vfe evaluate --execute
python scripts/vfe/paper.py ic_vfe score --output outputs/ic_vfe_scores.json --execute
```

Replace `ic_vfe` with `ic_vfe_no_context` or `recap` to use a baseline. Defaults
read `checkpoints/<experiment>/` and write predictions to
`outputs/paper_<experiment>/eval`. Use `--checkpoint`, `--source`, or `--output`
to override these locations. Without `--execute`, the launcher only prints its
command. `validate-data` performs CPU checks without loading model weights.

## Training

```bash
python scripts/vfe/paper.py ic_vfe preprocess --execute
python scripts/vfe/paper.py ic_vfe features --execute
python scripts/vfe/paper.py ic_vfe_no_context train --execute
python scripts/vfe/paper.py recap train --execute
python scripts/vfe/paper.py ic_vfe_no_block_embedding train --execute
```

Feature caching is needed only for context models. The main IC-VFE training
recipe requires a resume checkpoint. See the [training guide](reproduction.md#training)
for that recipe, pretraining, batch sizes, and the pending ablation.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/vfe/` | Models, data, training, inference, and scoring helpers |
| `scripts/vfe/paper.py` | Common preprocessing, training, and evaluation launcher |
| `scripts/vfe/download_checkpoints.py` | Pinned, verified checkpoint downloads |
| `scripts/vfe/score_paper.py` | Saved-prediction scoring |
| `configs/vfe/vfe_presets.json` | Four model presets with shared defaults |
| `configs/vfe/vfe_data_splits.json` | Annotation training and linear pretraining presets |
| `configs/vfe/icl_dataset_keep_true_all36_query80_ctx20.json` | Fixed training episode split |
| `configs/vfe/paper_experiments.json` | Experiments, paths, training settings, scoring, and provenance |
| `configs/vfe/checkpoints.json` | Hugging Face revisions and weight checksums |
| `tests/` | CPU regression tests |

Run tests with `python -m pip install pytest` followed by `python -m pytest -q`.
Tests do not download backbones or run GPU training. Data/annotation download
links and the missing no-block checkpoint are listed in the
[release coverage](reproduction.md#release-coverage).

## Citation

```bibtex
@article{macesanu2026victr,
  title={VICTR: Value-Informed In-Context Learning for Vision-Language-Action Models},
  author={Macesanu, Luca and Zhang, Tianluo and Patil, Aditya and Sharma, Archit and Angers, Kevin and Clark, Christopher M. and Feng, Chen and Zhang, Juexiao},
  journal={arXiv preprint arXiv:2611.08698},
  year={2026},
  url={https://arxiv.org/abs/2611.08698}
}
```
