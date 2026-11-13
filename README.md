# VICTR: Value-Informed In-Context Learning for Vision-Language-Action Models

[Paper](https://arxiv.org/abs/2611.08698) ·
[Project](https://ai4ce.github.io/VICTR/) ·
[Hugging Face collection](https://huggingface.co/collections/ai4ce/victr-value-informed-in-context-learning-for-vlas-6af362deff10640a6be06b31) ·
[Dataset](https://huggingface.co/datasets/lair-nyu/victr-icl-dataset) ·
[Training assets](https://huggingface.co/lair-nyu/victr-assets)

VICTR (Value-Informed Context Retrieval) is an in-context-learning policy for
vision-language-action models. For every observation it retrieves a demonstration
chunk from a pool, using a score that fuses DINOv2 visual similarity with a progress
value estimate from IC-VFE, and conditions a π0.5 policy on that chunk.

This repository has three components:

| Component | Code | Guide |
|---|---|---|
| **VICTR policy training** (retrieval pools, openpi training configs) | `src/victr/`, `scripts/victr/`, `shells/`, `third_party/openpi` | [docs/victr/training.md](docs/victr/training.md) |
| **IC-VFE** (in-context value-function estimation) | `src/vfe/`, `scripts/vfe/`, `configs/vfe/`, `tests/vfe/` | [docs/vfe/README.md](docs/vfe/README.md) |
| **Inference** (robot deployment, inference server, operator UI) | `src/inference/` | [src/inference/README.md](src/inference/README.md) |

The coding-agent baseline experiments are in
[`experiments/agent_baseline/`](experiments/agent_baseline/).

## Setup

```bash
git clone --recurse-submodules https://github.com/ai4ce/VICTR.git
cd VICTR
uv sync                 # VICTR data-preparation utilities
uv sync --extra vfe     # + IC-VFE training / evaluation
```

VICTR policy training runs in the openpi environment (`third_party/openpi`), and the
inference stack spans robot, GPU-server and UI environments. See each component's
guide for details.

## Citation

See [CITATION.cff](CITATION.cff).

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0) and
[LeRobot](https://github.com/huggingface/lerobot). Action interpolation follows
[RICL](https://arxiv.org/abs/2508.02062).
