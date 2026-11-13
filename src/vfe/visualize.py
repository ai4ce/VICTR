from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_value_curve(timesteps: np.ndarray, target: np.ndarray, prediction: np.ndarray, path: str | Path, title: str) -> None:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timesteps, target, label="ground truth", linewidth=2.0)
    ax.plot(timesteps, prediction, label="prediction", linewidth=2.0)
    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    ax.set_ylim(-1.05, 0.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_curve(timesteps: np.ndarray, error: np.ndarray, path: str | Path, title: str) -> None:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timesteps, error, linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("prediction - ground truth")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bin_distribution(bin_values: np.ndarray, probabilities: np.ndarray, path: str | Path, title: str) -> None:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(bin_values, probabilities, width=(bin_values[-1] - bin_values[0]) / len(bin_values))
    ax.set_xlabel("value bin")
    ax.set_ylabel("probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
