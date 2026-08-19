#!/usr/bin/env python3
"""D1 -- rendering fidelity vs. training-set density.

Scores the ten self-contained density-sweep checkpoints written by
``train_density.py`` / ``train_density_MLP.py``::

    outputs/density/mimogs/model_{6,12,25,50,100}.pth
    outputs/density/MLP/model_{6,12,25,50,100}.pth

on the ORIGINAL full test set, adds a learning-free nearest-neighbour
baseline per fraction, and writes the density curve to
``analysis/eval_density/``.

Zero-argument runnable::

    python eval_density.py

Nothing in the repository is modified.  Every metric is imported from
``evaluation/eval_render.py`` (directly, or through
``evaluation/eval_baseline_rt.score_prediction``, which is itself built on
``eval_render``'s ``EPS`` / ``topk_metrics`` / ``normalize_mag_map``), so the
numbers here are produced by exactly the same arithmetic as E1 and E2.

This module doubles as the shared plumbing for ``eval_distance.py``,
``eval_spots.py`` and ``eval_complexity.py``; everything below the metric
helpers is import-safe (no work happens at import time).

Normalization convention
------------------------
Headline metric is the SHAPE NMSE: max-normalized prediction vs.
max-normalized target, averaged per location in dB.  It is the only one of
``eval_render``'s two conventions that is comparable across methods -- the raw
convention penalises any predictor that does not happen to carry the
target's normalization, which the nearest-neighbour baseline and Sionna RT do
not.  ``NMSE_raw_dB`` stays in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from argparse import Namespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# ``evaluation/*.py`` import repo-root packages (``scene``, ``arguments``,
# ``utils``) as top-level modules AND import each other as top-level modules
# (``eval_baseline_rt`` does ``from eval_render import ...``).  Both directories
# therefore have to be importable, exactly the way ``train_density.py`` arranges
# it for its subprocesses.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import eval_render as ER  # noqa: E402  (path set up above)
from eval_baseline_rt import (  # noqa: E402
    load_raw_mat,
    match_positions,
    score_prediction,
)
from evaluation.train_MLP import CONFIGS as MLP_CONFIGS  # noqa: E402
from evaluation.train_MLP import PositionMLP  # noqa: E402
from utils.loss import normalize_mag_map  # noqa: E402


# ---------------------------------------------------------------------------
# Subsampling rule, shared with the trainers
# ---------------------------------------------------------------------------
try:
    from train_density import (  # noqa: E402
        DATASET_EPS,
        FRACTION_TAG,
        FRACTIONS,
        SEED,
        build_keep_index_map,
        extreme_sample_index,
        load_train_mat,
    )

    SUBSET_RULE_SOURCE = "imported from train_density.py"
except Exception as _import_error:  # noqa: BLE001 - fall back to a replica
    import scipy.io as _sio

    SUBSET_RULE_SOURCE = f"replicated locally ({_import_error})"

    DATASET_EPS = 1e-6
    SEED = 0
    FRACTIONS = (0.0625, 0.125, 0.25, 0.5, 1.0)
    FRACTION_TAG = {0.0625: "6", 0.125: "12", 0.25: "25", 0.5: "50", 1.0: "100"}

    def load_train_mat(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Replica of ``train_density.load_train_mat``."""
        contents = _sio.loadmat(os.path.join(dataset_dir, "train.mat"))
        return np.asarray(contents["positions"]), np.asarray(contents["magnitude"])

    def extreme_sample_index(positions: np.ndarray) -> int:
        """Replica of ``train_density.extreme_sample_index``."""
        return int(np.argmax(np.abs(positions).max(axis=1)))

    def build_keep_index_map(
        num_samples: int, extreme_index: int, fractions: Sequence[float] = FRACTIONS
    ) -> Dict[float, np.ndarray]:
        """Replica of ``train_density.build_keep_index_map``."""
        permutation = np.random.RandomState(SEED).permutation(num_samples)
        index_map: Dict[float, np.ndarray] = {}
        for fraction in fractions:
            kept_count = max(1, min(num_samples, int(round(float(fraction) * num_samples))))
            kept = permutation[:kept_count].copy()
            if extreme_index not in set(int(index) for index in kept):
                kept[-1] = extreme_index
            index_map[float(fraction)] = np.sort(kept)
        return index_map


FRACTION_PERCENT: Dict[float, float] = {
    0.0625: 6.25,
    0.125: 12.5,
    0.25: 25.0,
    0.5: 50.0,
    1.0: 100.0,
}

DEFAULT_MIMOGS_DIR = os.path.join(REPO_ROOT, "outputs", "density", "mimogs")
DEFAULT_MLP_DIR = os.path.join(REPO_ROOT, "outputs", "density", "MLP")
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")
DEFAULT_ANALYSIS_ROOT = os.path.join(REPO_ROOT, "analysis")

# The run directory the 100% MIMO-GS checkpoint was repacked from.  Used only
# by the self-consistency sanity block, and skipped when it no longer exists.
REFERENCE_RUN_DIR = os.path.join(REPO_ROOT, "outputs", "20260811_062015")
REPACK_TOLERANCE_DB = 0.05

METHOD_MIMOGS = "MIMO-GS"
METHOD_MLP = "MLP"
METHOD_NN = "Nearest neighbor"
METHOD_RT = "Sionna RT"
LEGEND_ORDER: Tuple[str, ...] = (METHOD_MIMOGS, METHOD_MLP, METHOD_NN, METHOD_RT)

METHOD_STYLE: Dict[str, Dict[str, object]] = {
    METHOD_MIMOGS: {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    METHOD_MLP: {"color": "tab:green", "marker": "s", "linestyle": "-"},
    METHOD_NN: {"color": "tab:gray", "marker": "^", "linestyle": "-"},
    METHOD_RT: {"color": "tab:orange", "marker": None, "linestyle": "--"},
}

TOPK_REPORTED: Tuple[int, ...] = (1, 4, 8)
CAPTURE_REPORTED: Tuple[int, ...] = (1, 4)


# ---------------------------------------------------------------------------
# Figure conventions (shared by all four eval_* scripts)
# ---------------------------------------------------------------------------
AXIS_LABEL_FONTSIZE = 14
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 10
FIGURE_DPI = 300


def style_axis(axis, xlabel: str = "", ylabel: str = "") -> None:
    """Apply the shared axis conventions.  Titles are never set anywhere."""
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(labelsize=TICK_LABELSIZE)


def save_figure(figure, output_dir: str, stem: str, tight: bool = True) -> None:
    """Write ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=FIGURE_DPI)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_readme(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("\n".join(str(line) for line in lines).rstrip() + "\n")


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint loading -- from the repacked .pth dicts alone, no run dirs
# ---------------------------------------------------------------------------
class LoadedMIMOGS:
    """A MIMO-GS density checkpoint rebuilt from its repacked payload."""

    def __init__(
        self,
        path: str,
        payload: dict,
        scene,
        gaussians,
        model_params: Namespace,
        device: torch.device,
        use_cuda_rasterizer: bool,
    ) -> None:
        self.path = path
        self.payload = payload
        self.config = payload["config"]
        self.scene = scene
        self.gaussians = gaussians
        self.model_params = model_params
        self.device = device
        self.use_cuda_rasterizer = use_cuda_rasterizer
        self.fraction = float(payload["fraction"])
        self.seed = int(payload["seed"])
        self.n_train = int(self.config["n_train"])
        self.beam_rows = int(self.config["beam_rows"])
        self.beam_cols = int(self.config["beam_cols"])
        self.dataset_dir = str(self.config["dataset_path"])

    @property
    def num_gaussians(self) -> int:
        return int(self.gaussians.get_xyz.shape[0])

    def primitive_parameter_count(self) -> int:
        """Learnable primitive tensors + gain MLP, counted from the capture.

        Independent of the beam grid: the primitives carry no per-beam state.
        """
        capture = self.payload["capture"]
        total = 0
        # Positional slots holding learnable primitive tensors; the four
        # gradient/importance accumulators (7..10) are statistics, not weights.
        for index in (3, 4, 5, 6, 14, 15, 16):
            if index < len(capture) and torch.is_tensor(capture[index]):
                total += int(capture[index].numel())
        gain_state = capture[12] if len(capture) > 12 else None
        if isinstance(gain_state, dict):
            total += int(sum(int(v.numel()) for v in gain_state.values()))
        return total


def load_mimogs(
    path: str, device: torch.device, dataset_override: str = ""
) -> LoadedMIMOGS:
    """Rebuild a MIMO-GS density checkpoint from ``{capture, config, ...}``."""
    if not os.path.isfile(path):
        raise SystemExit(f"[eval] MIMO-GS checkpoint is missing: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("capture", "config", "fraction", "seed"):
        if key not in payload:
            raise SystemExit(f"[eval] '{path}' has no '{key}' entry; not a density repack.")

    config = payload["config"]
    model_params = Namespace(**dict(config["model_params"]))
    opt_params = Namespace(**dict(config["opt_params"]))

    dataset_dir = os.path.abspath(dataset_override or config["dataset_path"])
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[eval] Dataset directory is missing: {dataset_dir}")

    # The repack records the throwaway subset directory in ``model_params``;
    # evaluation always runs against the pristine dataset's test.mat.
    model_params.source_path = dataset_dir
    model_params.model_path = ""

    checkpoint_shim = {"gaussians": payload["capture"]}
    hidden_dim = ER.gain_net_hidden_dim(checkpoint_shim)
    with ER.gain_net_width(hidden_dim):
        scene, gaussians = ER.build_scene_and_model(
            model_params, opt_params, checkpoint_shim, device
        )

    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1))) and device.type == "cuda"
    )

    return LoadedMIMOGS(
        path, payload, scene, gaussians, model_params, device, use_cuda_rasterizer
    )


class LoadedMLP:
    """A Position-MLP density checkpoint rebuilt from ``{state_dict, arch, ...}``."""

    def __init__(self, path: str, payload: dict, model: PositionMLP, device: torch.device) -> None:
        self.path = path
        self.payload = payload
        self.model = model
        self.device = device
        self.arch = dict(payload["arch"])
        self.fraction = float(payload["fraction"])
        self.seed = int(payload["seed"])
        self.n_train = int(payload["n_train"])
        self.beam_rows = int(payload["beam_rows"])
        self.beam_cols = int(payload["beam_cols"])
        self.dataset_dir = str(payload["dataset_path"])

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))


def load_mlp(path: str, device: torch.device) -> LoadedMLP:
    if not os.path.isfile(path):
        raise SystemExit(f"[eval] MLP checkpoint is missing: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "arch", "fraction", "seed"):
        if key not in payload:
            raise SystemExit(f"[eval] '{path}' has no '{key}' entry; not a density repack.")

    arch = payload["arch"]
    model = PositionMLP(
        num_outputs=int(arch["num_outputs"]),
        hidden=int(arch["hidden"]),
        depth=int(arch["depth"]),
        num_frequencies=int(arch["num_frequencies"]),
        include_input=bool(arch["include_input"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return LoadedMLP(path, payload, model, device)


def checkpoint_paths(directory: str) -> Dict[float, str]:
    """``{fraction: path}`` for the five ``model_<tag>.pth`` files."""
    return {
        float(fraction): os.path.join(directory, f"model_{FRACTION_TAG[float(fraction)]}.pth")
        for fraction in FRACTIONS
    }


# ---------------------------------------------------------------------------
# Prediction paths
# ---------------------------------------------------------------------------
def render_mimogs_maps(
    loaded: LoadedMIMOGS, normalized_positions: torch.Tensor, batch_size: int = 256
) -> torch.Tensor:
    """``(B,3)`` normalized UE positions -> ``(B, Nr, Nt)`` rendered maps."""
    tx_pos = torch.as_tensor(
        loaded.scene.bs_position, dtype=torch.float32, device=loaded.device
    )
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(normalized_positions.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(normalized_positions.shape[0]))
            chunks.append(
                ER.render_batch(
                    normalized_positions[start:stop].to(loaded.device),
                    tx_pos,
                    loaded.gaussians,
                    loaded.scene,
                    loaded.model_params,
                    loaded.use_cuda_rasterizer,
                ).float()
            )
    return torch.cat(chunks, dim=0)


def predict_mlp_maps(
    loaded: LoadedMLP, normalized_positions: torch.Tensor, batch_size: int = 512
) -> torch.Tensor:
    """``(B,3)`` normalized UE positions -> ``(B, Nr, Nt)`` predicted maps."""
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(normalized_positions.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(normalized_positions.shape[0]))
            chunks.append(loaded.model(normalized_positions[start:stop].to(loaded.device)).float())
    return torch.cat(chunks, dim=0).reshape(-1, loaded.beam_rows, loaded.beam_cols)


def nearest_neighbour_indices(
    query_positions: np.ndarray, reference_positions: np.ndarray, k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """``(distances, indices)`` of the k nearest reference points, in meters.

    Matching happens in ORIGINAL coordinates, never the per-file normalized
    ones.  A KD-tree is used when SciPy provides one and a chunked brute-force
    pass otherwise, mirroring ``eval_baseline_rt._neighbour_candidates``.
    """
    k = max(1, min(int(k), int(reference_positions.shape[0])))
    query = np.asarray(query_positions, dtype=np.float64)
    reference = np.asarray(reference_positions, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 - optional fast path

        distances, indices = cKDTree(reference).query(query, k=k)
        return (
            np.asarray(distances, dtype=np.float64).reshape(query.shape[0], k),
            np.asarray(indices, dtype=np.int64).reshape(query.shape[0], k),
        )
    except ImportError:
        pass

    distances = np.empty((query.shape[0], k), dtype=np.float64)
    indices = np.empty((query.shape[0], k), dtype=np.int64)
    chunk = 512
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        deltas = query[start:stop, None, :] - reference[None, :, :]
        block = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        order = np.argsort(block, axis=1)[:, :k]
        indices[start:stop] = order
        distances[start:stop] = np.take_along_axis(block, order, axis=1)
    return distances, indices


def nearest_neighbour_maps(
    train_positions: np.ndarray,
    train_magnitude: np.ndarray,
    test_positions: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Predict every test map as the train map of the nearest train position."""
    distances, indices = nearest_neighbour_indices(test_positions, train_positions, k=1)
    picked = np.ascontiguousarray(train_magnitude[indices[:, 0]])
    return (
        torch.as_tensor(picked, dtype=torch.float32, device=device),
        distances[:, 0].astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Ground truth handling
# ---------------------------------------------------------------------------
class TestGroundTruth:
    """The original full test set, loaded once and shared by every method."""

    def __init__(self, dataset_dir: str, device: torch.device) -> None:
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.device = device

        positions, magnitude = load_raw_mat(os.path.join(self.dataset_dir, "test.mat"))
        self.positions_m = positions.astype(np.float64)
        self.magnitude = torch.as_tensor(magnitude, dtype=torch.float32, device=device)
        self.beam_rows = int(self.magnitude.shape[1])
        self.beam_cols = int(self.magnitude.shape[2])

        # ``DeepMIMODataset`` normalizes by max|coordinate| of the file it loads.
        self.scale_factor = (
            float(torch.as_tensor(positions, dtype=torch.float32).abs().max()) + DATASET_EPS
        )
        self.positions_normalized = torch.as_tensor(
            self.positions_m, dtype=torch.float32, device=device
        ) / self.scale_factor

        # Zero-power maps make the NMSE denominator degenerate; every method is
        # scored on the same surviving subset.
        peak = self.magnitude.reshape(self.magnitude.shape[0], -1).amax(dim=1)
        self.valid_mask = (peak > ER.EPS).cpu().numpy()
        self.valid_indices = np.nonzero(self.valid_mask)[0].astype(np.int64)
        self.num_skipped_zero_power = int(self.magnitude.shape[0] - self.valid_indices.size)

        self.target_normalized = normalize_mag_map(
            self.magnitude[torch.as_tensor(self.valid_indices, device=device)]
        )

    def __len__(self) -> int:
        return int(self.magnitude.shape[0])

    @property
    def num_scored(self) -> int:
        return int(self.valid_indices.size)

    @property
    def valid_positions_m(self) -> np.ndarray:
        return self.positions_m[self.valid_indices]

    def score(self, predicted_full: torch.Tensor) -> Dict[str, np.ndarray]:
        """Score a full-length ``(N, Nr, Nt)`` prediction stack."""
        selected = predicted_full[torch.as_tensor(self.valid_indices, device=predicted_full.device)]
        return score_prediction(selected.float(), self.target_normalized)


def assert_finite_nonnegative(maps: torch.Tensor, label: str) -> None:
    """Guard every predictor's output before it reaches a metric."""
    if not bool(torch.isfinite(maps).all()):
        raise AssertionError(f"[eval] {label}: rendered output contains non-finite values.")
    minimum = float(maps.min())
    if minimum < 0.0:
        raise AssertionError(
            f"[eval] {label}: rendered output has negative entries (min = {minimum:.6g})."
        )


# ---------------------------------------------------------------------------
# Subset reconstruction
# ---------------------------------------------------------------------------
def build_subset_index_map(train_positions: np.ndarray) -> Dict[float, np.ndarray]:
    """The exact per-fraction train subsets ``train_density.py`` carved out."""
    return build_keep_index_map(
        int(train_positions.shape[0]), extreme_sample_index(train_positions), FRACTIONS
    )


def verify_subset_against_checkpoint(
    keep_indices: np.ndarray, expected_n_train: int, label: str, warnings: List[str]
) -> bool:
    """Assert the reconstructed subset size matches the checkpoint's ``n_train``."""
    actual = int(keep_indices.shape[0])
    if actual == int(expected_n_train):
        return True
    warnings.append(
        f"WARN {label}: reconstructed subset holds {actual} samples but the "
        f"checkpoint records n_train={int(expected_n_train)}."
    )
    return False


# ---------------------------------------------------------------------------
# Sionna RT flat reference (read from existing eval_baseline_rt outputs)
# ---------------------------------------------------------------------------
def read_sionna_rt_reference(
    analysis_root: str, dataset_dir: str
) -> Tuple[Optional[float], List[str]]:
    """Mean shape NMSE for Sionna RT, if an E2 run for this dataset exists.

    Returns ``(value_dB, source_paths)``; ``(None, [])`` when no matching
    ``comparison_rt/metrics_summary.csv`` is on disk.  Silently skipped by the
    caller in that case, as specified.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    values: List[float] = []
    sources: List[str] = []

    if not os.path.isdir(analysis_root):
        return None, []

    for name in sorted(os.listdir(analysis_root)):
        path = os.path.join(analysis_root, name, "comparison_rt", "metrics_summary.csv")
        for row in ER.read_csv_rows(path):
            if not row.get("predictor", "").strip().lower().startswith("sionna"):
                continue
            if os.path.abspath(str(row.get("gt_source_path", ""))) != dataset_dir:
                continue
            value = ER._as_float(row.get("NMSE_shape_mean_dB"))
            if value is None:
                continue
            values.append(float(value))
            sources.append(os.path.relpath(path, REPO_ROOT))

    if not values:
        return None, []
    return float(np.mean(values)), sources


# ---------------------------------------------------------------------------
# eval_density -- evaluation
# ---------------------------------------------------------------------------
def summarize_scores(scored: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Flatten one ``score_prediction`` result into the reported scalars."""
    summary: Dict[str, float] = {
        "nmse_shape_mean_dB": float(np.mean(scored["nmse_shape_db"])),
        "nmse_shape_median_dB": float(np.median(scored["nmse_shape_db"])),
        "nmse_shape_meanlinear_dB": ER.mean_linear_db(scored["nmse_shape_db"]),
        "nmse_raw_mean_dB": float(np.mean(scored["nmse_raw_db"])),
        "nmse_raw_median_dB": float(np.median(scored["nmse_raw_db"])),
    }
    for k in TOPK_REPORTED:
        summary[f"topk_acc_K{k}"] = float(np.mean(scored[f"topk_acc_K{k}"]))
    for k in CAPTURE_REPORTED:
        summary[f"power_capture_K{k}"] = float(np.mean(scored[f"power_capture_K{k}"]))
    return summary


def evaluate_density_sweep(
    arguments: argparse.Namespace, device: torch.device
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[str]]:
    """Score all ten checkpoints plus the nearest-neighbour baseline."""
    warnings: List[str] = []

    mimogs_paths = checkpoint_paths(arguments.mimogs_dir)
    mlp_paths = checkpoint_paths(arguments.mlp_dir)

    # The dataset recorded in the checkpoints wins unless overridden.
    probe = torch.load(mimogs_paths[1.0], map_location="cpu", weights_only=False)
    dataset_dir = os.path.abspath(arguments.dataset or probe["config"]["dataset_path"])
    del probe

    ground_truth = TestGroundTruth(dataset_dir, device)
    train_positions, train_magnitude = load_train_mat(dataset_dir)
    index_map = build_subset_index_map(train_positions)

    print(f"[eval_density] dataset            : {dataset_dir}")
    print(f"[eval_density] test locations     : {len(ground_truth)} "
          f"(scored {ground_truth.num_scored}, "
          f"skipped zero-power {ground_truth.num_skipped_zero_power})")
    print(f"[eval_density] beam grid          : "
          f"{ground_truth.beam_rows} x {ground_truth.beam_cols}")
    print(f"[eval_density] train locations    : {train_positions.shape[0]}")
    print(f"[eval_density] subsampling rule   : {SUBSET_RULE_SOURCE} (seed {SEED})")
    print("")

    rows: List[Dict[str, object]] = []
    context: Dict[str, object] = {
        "dataset_dir": dataset_dir,
        "ground_truth": ground_truth,
        "train_positions": train_positions,
        "train_magnitude": train_magnitude,
        "index_map": index_map,
        "mimogs_paths": mimogs_paths,
        "mlp_paths": mlp_paths,
    }

    for fraction in sorted(FRACTIONS):
        percent = FRACTION_PERCENT[fraction]
        keep_indices = index_map[fraction]

        # -- MIMO-GS ----------------------------------------------------
        loaded_gs = load_mimogs(mimogs_paths[fraction], device, dataset_dir)
        verify_subset_against_checkpoint(
            keep_indices, loaded_gs.n_train, f"MIMO-GS {percent:g}%", warnings
        )
        gs_maps = render_mimogs_maps(loaded_gs, ground_truth.positions_normalized)
        assert_finite_nonnegative(gs_maps, f"MIMO-GS {percent:g}%")
        gs_scored = ground_truth.score(gs_maps)
        gs_summary = summarize_scores(gs_scored)
        gs_summary.update(
            {
                "method": METHOD_MIMOGS,
                "fraction": fraction,
                "percent": percent,
                "n_train": loaded_gs.n_train,
                "seed": loaded_gs.seed,
                "num_scored": ground_truth.num_scored,
                "checkpoint": os.path.relpath(loaded_gs.path, REPO_ROOT),
                "num_parameters": loaded_gs.primitive_parameter_count(),
            }
        )
        rows.append(gs_summary)
        if fraction == 1.0:
            context["mimogs_100"] = {
                "loaded": loaded_gs,
                "maps": gs_maps,
                "scored": gs_scored,
                "summary": gs_summary,
            }
        else:
            del loaded_gs, gs_maps

        # -- MLP --------------------------------------------------------
        loaded_mlp = load_mlp(mlp_paths[fraction], device)
        verify_subset_against_checkpoint(
            keep_indices, loaded_mlp.n_train, f"MLP {percent:g}%", warnings
        )
        mlp_maps = predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
        assert_finite_nonnegative(mlp_maps, f"MLP {percent:g}%")
        mlp_summary = summarize_scores(ground_truth.score(mlp_maps))
        mlp_summary.update(
            {
                "method": METHOD_MLP,
                "fraction": fraction,
                "percent": percent,
                "n_train": loaded_mlp.n_train,
                "seed": loaded_mlp.seed,
                "num_scored": ground_truth.num_scored,
                "checkpoint": os.path.relpath(loaded_mlp.path, REPO_ROOT),
                "num_parameters": loaded_mlp.parameter_count,
            }
        )
        rows.append(mlp_summary)
        del loaded_mlp, mlp_maps

        # -- Nearest neighbour on the same subset -----------------------
        nn_maps, nn_distance = nearest_neighbour_maps(
            train_positions[keep_indices],
            train_magnitude[keep_indices],
            ground_truth.positions_m,
            device,
        )
        assert_finite_nonnegative(nn_maps, f"Nearest neighbor {percent:g}%")
        nn_summary = summarize_scores(ground_truth.score(nn_maps))
        nn_summary.update(
            {
                "method": METHOD_NN,
                "fraction": fraction,
                "percent": percent,
                "n_train": int(keep_indices.shape[0]),
                "seed": SEED,
                "num_scored": ground_truth.num_scored,
                "checkpoint": "(no learning)",
                "num_parameters": 0,
                "mean_nn_distance_m": float(np.mean(nn_distance)),
            }
        )
        rows.append(nn_summary)
        del nn_maps

        print(
            f"[eval_density] {percent:>6.2f}%  n_train={int(keep_indices.shape[0]):>6}  "
            f"MIMO-GS {gs_summary['nmse_shape_mean_dB']:>8.3f} dB   "
            f"MLP {mlp_summary['nmse_shape_mean_dB']:>8.3f} dB   "
            f"NN {nn_summary['nmse_shape_mean_dB']:>8.3f} dB"
        )

    return rows, context, warnings


# ---------------------------------------------------------------------------
# eval_density -- sanity block
# ---------------------------------------------------------------------------
def sanity_repack_consistency(
    context: Dict[str, object], device: torch.device, batch_size: int
) -> Tuple[Dict[str, object], List[str]]:
    """model_100 scored twice: this script's loading path vs. eval_render's.

    The comparison validates the repack + reload path only.  It deliberately
    does NOT quote any externally remembered number, and it never touches
    ``eval_net_rate``'s pipeline, whose g_med rescale and private NMSE helper
    are a different convention.
    """
    warnings: List[str] = []
    entry = context["mimogs_100"]
    repack_db = float(entry["summary"]["nmse_shape_mean_dB"])

    result: Dict[str, object] = {
        "repack_shape_mean_dB": repack_db,
        "reference_run_dir": REFERENCE_RUN_DIR,
        "eval_render_shape_mean_dB": None,
        "delta_dB": None,
        "status": "skipped",
    }

    # Same metric function, this script's loading path, for a self-check that
    # the two scorers agree bit-for-bit on the identical render.
    loaded = entry["loaded"]
    via_evaluate_test_set = ER.evaluate_test_set(
        loaded.scene,
        loaded.gaussians,
        loaded.model_params,
        device,
        batch_size,
        loaded.use_cuda_rasterizer,
    )
    result["repack_evaluate_test_set_dB"] = float(
        np.mean(via_evaluate_test_set["nmse_shape_db"])
    )

    checkpoint_path = os.path.join(REFERENCE_RUN_DIR, ER.CHECKPOINT_NAME)
    if not os.path.isfile(checkpoint_path):
        result["status"] = "reference run dir absent"
        return result, warnings

    run_dir, checkpoint_path = ER.resolve_run_dir(
        REFERENCE_RUN_DIR, os.path.join(REPO_ROOT, "outputs")
    )
    reference_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = ER.restore_config(run_dir, reference_checkpoint)

    hidden_dim = ER.gain_net_hidden_dim(reference_checkpoint)
    with ER.gain_net_width(hidden_dim):
        scene, gaussians = ER.build_scene_and_model(
            model_params, opt_params, reference_checkpoint, device
        )

    reference_results = ER.evaluate_test_set(
        scene,
        gaussians,
        model_params,
        device,
        batch_size,
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1))) and device.type == "cuda",
    )
    reference_db = float(np.mean(reference_results["nmse_shape_db"]))
    delta = abs(repack_db - reference_db)

    result["eval_render_shape_mean_dB"] = reference_db
    result["delta_dB"] = delta
    result["status"] = "ok" if delta <= REPACK_TOLERANCE_DB else "MISMATCH"

    if delta > REPACK_TOLERANCE_DB:
        warnings.append(
            f"WARN model_100 repack consistency: |{repack_db:.4f} - {reference_db:.4f}| = "
            f"{delta:.4f} dB exceeds the {REPACK_TOLERANCE_DB:.2f} dB tolerance."
        )

    del scene, gaussians, reference_checkpoint
    return result, warnings


def sanity_monotonicity(rows: Sequence[Dict[str, object]]) -> List[str]:
    """WARN when a method's NMSE worsens as the training fraction grows."""
    warnings: List[str] = []
    for method in (METHOD_MIMOGS, METHOD_MLP, METHOD_NN):
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda r: r["fraction"]
        )
        for previous, current in zip(selected, selected[1:]):
            if float(current["nmse_shape_mean_dB"]) > float(previous["nmse_shape_mean_dB"]):
                warnings.append(
                    f"WARN monotonicity ({method}): NMSE worsens from "
                    f"{float(previous['percent']):g}% ({float(previous['nmse_shape_mean_dB']):.3f} dB) "
                    f"to {float(current['percent']):g}% "
                    f"({float(current['nmse_shape_mean_dB']):.3f} dB)."
                )
    return warnings


# ---------------------------------------------------------------------------
# eval_density -- figure
# ---------------------------------------------------------------------------
def plot_nmse_vs_density(
    output_dir: str,
    rows: Sequence[Dict[str, object]],
    rt_reference_db: Optional[float],
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6))

    for method in (METHOD_MIMOGS, METHOD_MLP, METHOD_NN):
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda r: r["fraction"]
        )
        if not selected:
            continue
        style = METHOD_STYLE[method]
        axis.plot(
            [float(row["percent"]) for row in selected],
            [float(row["nmse_shape_mean_dB"]) for row in selected],
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.5,
        )

    if rt_reference_db is not None:
        style = METHOD_STYLE[METHOD_RT]
        axis.axhline(
            float(rt_reference_db),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=METHOD_RT,
        )

    percents = [FRACTION_PERCENT[f] for f in sorted(FRACTIONS)]
    axis.set_xscale("log", base=2)
    axis.set_xticks(percents)
    axis.set_xticklabels([f"{value:g}" for value in percents])
    axis.minorticks_off()
    style_axis(axis, "Training-set fraction [%]", "Shape NMSE [dB]")
    axis.grid(alpha=0.3, linewidth=0.5)

    handles, labels = axis.get_legend_handles_labels()
    ordered = [
        (handles[labels.index(name)], name) for name in LEGEND_ORDER if name in labels
    ]
    axis.legend(
        [handle for handle, _ in ordered],
        [name for _, name in ordered],
        fontsize=LEGEND_FONTSIZE,
        loc="best",
    )

    save_figure(figure, output_dir, "fig_nmse_vs_density")


# ---------------------------------------------------------------------------
# eval_density -- outputs
# ---------------------------------------------------------------------------
DENSITY_CSV_COLUMNS: Tuple[str, ...] = (
    "method",
    "fraction",
    "percent",
    "n_train",
    "seed",
    "num_scored",
    "nmse_shape_mean_dB",
    "nmse_shape_median_dB",
    "nmse_shape_meanlinear_dB",
    "nmse_raw_mean_dB",
    "nmse_raw_median_dB",
    "topk_acc_K1",
    "topk_acc_K4",
    "topk_acc_K8",
    "power_capture_K1",
    "power_capture_K4",
    "mean_nn_distance_m",
    "num_parameters",
    "checkpoint",
)


def write_density_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    records = []
    for row in rows:
        record = []
        for column in DENSITY_CSV_COLUMNS:
            value = row.get(column, "")
            if isinstance(value, float):
                record.append(f"{value:.6f}")
            else:
                record.append(value)
        records.append(record)
    write_csv(path, DENSITY_CSV_COLUMNS, records)


def print_density_table(
    rows: Sequence[Dict[str, object]], rt_reference_db: Optional[float]
) -> None:
    print("")
    print("=" * 100)
    print("[eval_density] SUMMARY -- shape NMSE vs. training-set density (full test set)")
    print("=" * 100)
    header = (
        f"  {'fraction':>9}{'n_train':>9}"
        f"{'MIMO-GS':>11}{'MLP':>11}{'Nearest nb.':>13}"
        f"{'GS top-1':>10}{'MLP top-1':>11}{'NN top-1':>10}"
    )
    print(header)
    print(f"  {'':>9}{'':>9}{'[dB]':>11}{'[dB]':>11}{'[dB]':>13}{'':>10}{'':>11}{'':>10}")
    print("  " + "-" * (len(header) - 2))

    by_fraction: Dict[float, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_fraction.setdefault(float(row["fraction"]), {})[str(row["method"])] = row

    for fraction in sorted(by_fraction):
        entry = by_fraction[fraction]
        gs = entry.get(METHOD_MIMOGS, {})
        mlp = entry.get(METHOD_MLP, {})
        nn = entry.get(METHOD_NN, {})
        print(
            f"  {FRACTION_PERCENT[fraction]:>8.2f}%{int(gs.get('n_train', 0)):>9}"
            f"{float(gs.get('nmse_shape_mean_dB', float('nan'))):>11.3f}"
            f"{float(mlp.get('nmse_shape_mean_dB', float('nan'))):>11.3f}"
            f"{float(nn.get('nmse_shape_mean_dB', float('nan'))):>13.3f}"
            f"{float(gs.get('topk_acc_K1', float('nan'))):>10.4f}"
            f"{float(mlp.get('topk_acc_K1', float('nan'))):>11.4f}"
            f"{float(nn.get('topk_acc_K1', float('nan'))):>10.4f}"
        )

    print("  " + "-" * (len(header) - 2))
    if rt_reference_db is None:
        print("  Sionna RT reference : not available (no matching comparison_rt output)")
    else:
        print(f"  Sionna RT reference : {rt_reference_db:.3f} dB (flat, from eval_baseline_rt)")
    print("  NMSE convention     : max-normalized prediction vs. max-normalized target,")
    print("                        averaged per location in dB (eval_render 'shape').")
    print("=" * 100)


def build_density_readme(
    rows: Sequence[Dict[str, object]],
    context: Dict[str, object],
    sanity: Dict[str, object],
    rt_reference_db: Optional[float],
    rt_sources: Sequence[str],
    warnings: Sequence[str],
    device: torch.device,
) -> List[str]:
    ground_truth: TestGroundTruth = context["ground_truth"]
    lines = [
        "eval_density -- rendering fidelity vs. training-set density",
        "=" * 70,
        "",
        "CONVENTIONS",
        "  Metric      : shape NMSE, i.e. max-normalized prediction vs. max-normalized",
        "                target, averaged per location in dB.  Imported from",
        "                evaluation/eval_render.py (via eval_baseline_rt.score_prediction),",
        "                never reimplemented.  Raw NMSE and top-K numbers are in the CSV.",
        "  Test set    : the ORIGINAL full test.mat of the dataset recorded in each",
        f"                checkpoint ({context['dataset_dir']}).",
        f"                {len(ground_truth)} locations, "
        f"{ground_truth.num_scored} scored, "
        f"{ground_truth.num_skipped_zero_power} skipped for zero power.",
        f"  Beam grid   : {ground_truth.beam_rows} x {ground_truth.beam_cols}.",
        f"  Subsets     : reproduced with the train_density.py rule "
        f"({SUBSET_RULE_SOURCE}, seed {SEED}):",
        "                RandomState(seed).permutation prefix, with the extreme-|coord|",
        "                sample force-included so every fraction shares one",
        "                normalization scale.  Sizes are asserted against each",
        "                checkpoint's recorded n_train.",
        "  Baseline    : Nearest neighbor predicts each test map as the train map of the",
        "                nearest subsampled train position (3D Euclidean, original",
        "                meters).  No learning of any kind.",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "  Figures     : no titles; axis labels 14 pt, ticks 12 pt, legend 10 pt;",
        "                PNG at 300 dpi plus PDF.",
        "",
        "FILES",
        "  fig_nmse_vs_density.{png,pdf}  shape NMSE vs. training fraction (log2 x axis)",
        "  density_metrics.csv            every number plotted, plus raw NMSE / top-K",
        "  README.txt                     this file",
        "",
        "HEADLINE NUMBERS (shape NMSE [dB], mean over the scored test locations)",
        f"  {'fraction':>9}{'n_train':>9}{'MIMO-GS':>11}{'MLP':>11}{'Nearest nb.':>13}",
    ]

    by_fraction: Dict[float, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_fraction.setdefault(float(row["fraction"]), {})[str(row["method"])] = row
    for fraction in sorted(by_fraction):
        entry = by_fraction[fraction]
        lines.append(
            f"  {FRACTION_PERCENT[fraction]:>8.2f}%"
            f"{int(entry[METHOD_MIMOGS]['n_train']):>9}"
            f"{float(entry[METHOD_MIMOGS]['nmse_shape_mean_dB']):>11.3f}"
            f"{float(entry[METHOD_MLP]['nmse_shape_mean_dB']):>11.3f}"
            f"{float(entry[METHOD_NN]['nmse_shape_mean_dB']):>13.3f}"
        )

    lines.append("")
    if rt_reference_db is None:
        lines.append("  Sionna RT flat reference : not available; the dashed line is omitted.")
    else:
        lines.append(f"  Sionna RT flat reference : {rt_reference_db:.3f} dB")
        for source in rt_sources:
            lines.append(f"    source: {source}")

    lines += [
        "",
        "SANITY BLOCK",
        f"  model_100 via this script's repack loading path : "
        f"{float(sanity['repack_shape_mean_dB']):.4f} dB",
        f"  model_100 via eval_render.evaluate_test_set     : "
        f"{float(sanity['repack_evaluate_test_set_dB']):.4f} dB (same loading path)",
    ]
    if sanity["eval_render_shape_mean_dB"] is None:
        lines.append(
            f"  model_100 via eval_render's own loading path    : skipped "
            f"({sanity['status']}: {sanity['reference_run_dir']})"
        )
    else:
        lines += [
            f"  model_100 via eval_render's own loading path    : "
            f"{float(sanity['eval_render_shape_mean_dB']):.4f} dB "
            f"(run dir {os.path.relpath(str(sanity['reference_run_dir']), REPO_ROOT)})",
            f"  delta                                          : "
            f"{float(sanity['delta_dB']):.4f} dB "
            f"(tolerance {REPACK_TOLERANCE_DB:.2f} dB -> {sanity['status']})",
        ]
    lines += [
        "  The comparison validates the repack path itself.  It is deliberately not",
        "  compared against eval_net_rate logs: that pipeline applies a g_med rescale",
        "  and uses its own internal NMSE helper, so its numbers are not commensurate.",
        "  Monotonicity and finite/non-negative output checks also run; see WARNINGS.",
        "",
        "WARNINGS",
    ]
    lines += [f"  {warning}" for warning in warnings] or ["  none"]
    lines += [
        "",
        "RERUN",
        "  python eval_density.py",
    ]
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D1 -- MIMO-GS / MLP / nearest-neighbour fidelity vs. training density"
    )
    parser.add_argument("--mimogs_dir", type=str, default=DEFAULT_MIMOGS_DIR)
    parser.add_argument("--mlp_dir", type=str, default=DEFAULT_MLP_DIR)
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Override the dataset directory recorded in the checkpoints.",
    )
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the sanity block's eval_render.evaluate_test_set passes.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)

    print("=" * 100)
    print("[eval_density] Rendering fidelity vs. training-set density")
    print("=" * 100)
    print(f"[eval_density] device             : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[eval_density] MIMO-GS checkpoints: {arguments.mimogs_dir}")
    print(f"[eval_density] MLP checkpoints    : {arguments.mlp_dir}")
    print("")

    rows, context, warnings = evaluate_density_sweep(arguments, device)

    print("")
    print("-" * 100)
    print("[eval_density] SANITY BLOCK")
    print("-" * 100)

    sanity, sanity_warnings = sanity_repack_consistency(context, device, arguments.batch_size)
    warnings.extend(sanity_warnings)

    print(f"  (a) model_100 self-consistency")
    print(f"      this script's repack loading path      : "
          f"{float(sanity['repack_shape_mean_dB']):.4f} dB")
    print(f"      same path via evaluate_test_set        : "
          f"{float(sanity['repack_evaluate_test_set_dB']):.4f} dB")
    if sanity["eval_render_shape_mean_dB"] is None:
        print(f"      eval_render's own loading path         : SKIPPED "
              f"({sanity['status']}: "
              f"{os.path.relpath(str(sanity['reference_run_dir']), REPO_ROOT)})")
    else:
        print(f"      eval_render's own loading path         : "
              f"{float(sanity['eval_render_shape_mean_dB']):.4f} dB")
        print(f"      delta                                  : "
              f"{float(sanity['delta_dB']):.4f} dB "
              f"(tolerance {REPACK_TOLERANCE_DB:.2f} dB) -> {sanity['status']}")

    monotonicity_warnings = sanity_monotonicity(rows)
    warnings.extend(monotonicity_warnings)
    print("  (b) monotonicity (NMSE must not worsen as the fraction grows)")
    if monotonicity_warnings:
        for warning in monotonicity_warnings:
            print(f"      {warning}")
    else:
        print("      OK -- every method improves monotonically with more training data.")

    print("  (c) finite / non-negative rendered outputs")
    print("      OK -- asserted for every model at every fraction while scoring.")

    rt_reference_db, rt_sources = read_sionna_rt_reference(
        arguments.analysis_root, str(context["dataset_dir"])
    )
    if rt_reference_db is None:
        print("  Sionna RT flat reference: absent, dashed line omitted.")
    else:
        print(f"  Sionna RT flat reference: {rt_reference_db:.3f} dB "
              f"from {', '.join(rt_sources)}")

    output_dir = os.path.join(arguments.analysis_root, "eval_density")
    os.makedirs(output_dir, exist_ok=True)

    write_density_csv(os.path.join(output_dir, "density_metrics.csv"), rows)
    plot_nmse_vs_density(output_dir, rows, rt_reference_db)
    write_readme(
        os.path.join(output_dir, "README.txt"),
        build_density_readme(
            rows, context, sanity, rt_reference_db, rt_sources, warnings, device
        ),
    )

    print_density_table(rows, rt_reference_db)

    print("")
    if warnings:
        print(f"[eval_density] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_density] No warnings.")
    print(f"[eval_density] Outputs written to {output_dir}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
