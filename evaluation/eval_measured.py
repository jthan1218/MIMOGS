#!/usr/bin/env python3
"""D4 -- measured 60 GHz indoor dataset: quantitative table + rendering samples.

Scores the two 1000-epoch checkpoints trained on ``dataset/indoor_63by63``
(``DEFAULT_MIMOGS_CKPT`` / ``DEFAULT_MLP_CKPT`` below; the MIMO-GS run
directory's ``run_args.txt`` supplies its configuration) on the TEST split
only, adds a learning-free nearest-neighbour baseline, and writes the
measured-dataset table plus the qualitative rendering samples to
``analysis/eval_measured/``.

Zero-argument runnable::

    python eval_measured.py                      # table + candidate gallery
    python eval_measured.py --spots 12,34,56     # final publication figure

Nothing in the repository is modified.  Every metric comes from
``evaluation/eval_render.py`` -- directly, or through
``evaluation/eval_baseline_rt.score_prediction`` and the shared plumbing in
``eval_density.py`` (``TestGroundTruth``, the nearest-neighbour baseline, the
figure conventions), which are themselves built on ``eval_render``'s ``EPS`` /
``topk_metrics`` / ``normalize_mag_map``.  The measured table therefore aligns
column-for-column with the DeepMIMO table ``eval_density.py`` writes.

Normalization convention
------------------------
Headline metric is the SHAPE NMSE: max-normalized prediction vs. max-normalized
target, averaged per location in dB.  It is the only one of ``eval_render``'s
two conventions that is comparable across methods -- the raw convention
penalises any predictor that does not happen to carry the target's
normalization, which the nearest-neighbour baseline does not.  ``NMSE_raw_dB``
stays in the CSV.

Beam grid
---------
The dataset's 63 beams per side are a MEASURED analog steering codebook: 21
azimuth x 3 elevation directions, azimuth-fastest, which is not a DFT grid.
The renderer's beam centers are whatever the evaluated checkpoint was trained
with, and ``beam_grid_forwarding`` below makes ``eval_render``'s render path
honour a ``custom_angles`` checkpoint (that path hard-codes the DFT bins).
The restored mode is asserted against the dataset's beam count and reported in
the README, so a checkpoint trained on DFT bins is never quietly presented as
a measured-codebook one.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Dict, Iterator, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# ``evaluation/*.py`` import repo-root packages (``scene``, ``arguments``,
# ``utils``) as top-level modules AND import each other as top-level modules,
# so both directories have to be importable.  ``eval_density.py`` arranges the
# same thing at import time; this block only makes the order independent of it.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

try:
    from eval_density import (  # noqa: E402  (path set up above)
        AXIS_LABEL_FONTSIZE,
        CAPTURE_REPORTED,
        DEFAULT_ANALYSIS_ROOT,
        FIGURE_DPI,
        METHOD_MIMOGS,
        METHOD_MLP,
        METHOD_NN,
        TICK_LABELSIZE,
        TOPK_REPORTED,
        TestGroundTruth,
        assert_finite_nonnegative,
        load_raw_mat,
        nearest_neighbour_maps,
        resolve_device,
        summarize_scores,
        write_csv,
        write_readme,
    )
except ImportError as _error:  # pragma: no cover - a missing sibling script
    raise SystemExit(
        f"[eval_measured] Cannot import the shared evaluation plumbing from "
        f"'{os.path.join(REPO_ROOT, 'eval_density.py')}': {_error}"
    ) from _error

import eval_render as ER  # noqa: E402  (path set up above)
from evaluation.train_MLP import PositionMLP  # noqa: E402
from gaussian_renderer import (  # noqa: E402
    MEASURED_BEAM_AZ_DEG,
    MEASURED_BEAM_EL_DEG,
    _build_beam_uv_grid,
    _build_custom_uv_grid,
    parse_angle_list,
)
from gaussian_renderer.fast_renderer import render_fast  # noqa: E402
from utils.loss import normalize_mag_map  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "indoor_63by63")
DEFAULT_MIMOGS_CKPT = os.path.join(REPO_ROOT, "outputs", "20260811_034547", "model.pth")
DEFAULT_MLP_CKPT = os.path.join(REPO_ROOT, "outputs", "mlp_indoor_medium", "model.pth")

# Figure rows (the MLP is scored but never drawn) and table rows.
METHOD_GT = "Ground truth"
METHOD_ERROR = "Absolute error"
TABLE_ORDER: Tuple[str, ...] = (METHOD_MIMOGS, METHOD_MLP, METHOD_NN)

# The measured codebook, for the beam-grid assertion and the README text.
MEASURED_NUM_AZ = len(MEASURED_BEAM_AZ_DEG)
MEASURED_NUM_EL = len(MEASURED_BEAM_EL_DEG)

DB_FLOOR = -30.0
SELF_CHECK_TOLERANCE_DB = 0.05

# Error-panel color limit.  A fixed 0..1 scale would be honest but unreadable:
# the measured maps are so peaky that the median per-location peak error is
# around a quarter of the GT peak, so the whole row renders black.  The upper
# limit is therefore derived from the data once, shared by every panel of every
# figure so error panels stay comparable across locations, and reported.
ERROR_VMAX_PERCENTILE = 99.9
ERROR_VMAX_STEP = 0.05
ERROR_VMAX_MIN = 0.10
ERROR_VMAX_MAX = 1.00

# Panel geometry.  The maps are square (63 x 63) and drawn with an equal
# aspect, so the figure size is derived from one panel edge instead of being
# guessed: a mismatched figsize leaves a wide white margin beside the panels.
PANEL_INCH = 2.6
PANEL_MARGIN_W = 1.7
# Vertical slack for the x-axis label and ticks, plus the gallery's caption.
# The final figure has no caption, so it needs less or the rows drift apart.
PANEL_MARGIN_H = 1.5
PANEL_MARGIN_H_FINAL = 1.0
# The final figure names its columns "Spot k" so the text can refer to them;
# the DeepMIMO counterpart figure titles its columns the same way.
COLUMN_TITLE_FONTSIZE = 12

# Gallery size: 0 draws every scored test location, which is what picking the
# final figure's spots actually needs.  A positive --gallery_top keeps only that
# many, ranked by rendering quality, for a quick look at a large test set.
GALLERY_TOP_DEFAULT = 0

DISTANCE_PERCENTILE = 90.0

TABLE_COLUMNS: Tuple[str, ...] = (
    ("method",)
    + (
        "nmse_shape_mean_dB",
        "nmse_shape_median_dB",
        "nmse_shape_meanlinear_dB",
        "nmse_raw_mean_dB",
        "nmse_raw_median_dB",
    )
    + tuple(f"topk_acc_K{k}" for k in TOPK_REPORTED)
    + tuple(f"power_capture_K{k}" for k in CAPTURE_REPORTED)
    + ("checkpoint", "n_train", "n_test")
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
class LoadedMIMOGS:
    """A MIMO-GS run-directory checkpoint, restored the way eval_render does."""

    def __init__(
        self,
        path: str,
        run_dir: str,
        scene,
        gaussians,
        model_params,
        opt_params,
        device: torch.device,
        use_cuda_rasterizer: bool,
        iteration: int,
    ) -> None:
        self.path = path
        self.run_dir = run_dir
        self.scene = scene
        self.gaussians = gaussians
        self.model_params = model_params
        self.opt_params = opt_params
        self.device = device
        self.use_cuda_rasterizer = use_cuda_rasterizer
        self.iteration = iteration

    @property
    def num_gaussians(self) -> int:
        return int(self.gaussians.get_xyz.shape[0])

    @property
    def beam_grid_mode(self) -> str:
        return str(getattr(self.model_params, "beam_grid_mode", "dft") or "dft").lower()


def load_mimogs(
    checkpoint_path: str, device: torch.device, dataset_dir: str
) -> LoadedMIMOGS:
    """Restore ``model.pth`` + its ``run_args.txt`` through ``eval_render``."""
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"[eval_measured] MIMO-GS checkpoint is missing: {checkpoint_path}\n"
            f"                Train it first, or pass --mimogs_ckpt <run dir or "
            f"model.pth>."
        )

    run_dir, checkpoint_path = ER.resolve_run_dir(
        checkpoint_path, os.path.join(REPO_ROOT, "outputs")
    )
    if not os.path.isfile(os.path.join(run_dir, "run_args.txt")):
        print(
            f"[eval_measured] NOTE: '{run_dir}' has no run_args.txt; the "
            f"configuration is restored from the checkpoint payload alone."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = ER.restore_config(run_dir, checkpoint)

    model_params.source_path = os.path.abspath(dataset_dir)
    if not os.path.isdir(model_params.source_path):
        raise SystemExit(
            f"[eval_measured] Dataset directory is missing: {model_params.source_path}"
        )

    hidden_dim = ER.gain_net_hidden_dim(checkpoint)
    if hidden_dim is not None:
        print(f"[eval_measured] checkpoint DynamicGainNet hidden width: {hidden_dim}")

    with ER.gain_net_width(hidden_dim):
        scene, gaussians = ER.build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
        and device.type == "cuda"
    )

    return LoadedMIMOGS(
        checkpoint_path,
        run_dir,
        scene,
        gaussians,
        model_params,
        opt_params,
        device,
        use_cuda_rasterizer,
        int(checkpoint.get("iteration", -1)),
    )


def infer_mlp_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Recover ``PositionMLP``'s shape from its weights alone.

    ``PositionMLP`` is ``[Linear, ReLU] * depth`` followed by one output
    ``Linear``, all inside a single ``nn.Sequential`` called ``net``, so the
    ordered list of ``net.<i>.weight`` tensors determines every constructor
    argument.  The Fourier-feature input width is
    ``3 * ((1 if include_input else 0) + 2 * num_frequencies)``, which has a
    unique solution: an odd ``in_dim / 3`` means the raw input is concatenated.
    """
    weights = [
        (int(key.split(".")[1]), tensor)
        for key, tensor in state_dict.items()
        if key.startswith("net.") and key.endswith(".weight")
    ]
    if not weights:
        raise SystemExit(
            "[eval_measured] The MLP state_dict holds no 'net.<i>.weight' tensors; "
            "it was not written by evaluation/train_MLP.py's PositionMLP."
        )
    weights.sort(key=lambda item: item[0])

    hidden = int(weights[0][1].shape[0])
    input_dim = int(weights[0][1].shape[1])
    num_outputs = int(weights[-1][1].shape[0])
    depth = len(weights) - 1

    if input_dim % 3:
        raise SystemExit(
            f"[eval_measured] The MLP's input width {input_dim} is not a multiple "
            f"of 3, so it cannot come from FourierFeatures(in_dim=3)."
        )
    blocks = input_dim // 3
    include_input = bool(blocks % 2)
    num_frequencies = (blocks - 1) // 2 if include_input else blocks // 2

    return {
        "hidden": hidden,
        "depth": depth,
        "num_outputs": num_outputs,
        "num_frequencies": int(num_frequencies),
        "include_input": include_input,
        "input_dim": input_dim,
    }


class LoadedMLP:
    """A ``PositionMLP`` checkpoint written by ``train_MLP_indoor.py``."""

    def __init__(
        self,
        path: str,
        model: PositionMLP,
        architecture: Dict[str, object],
        beam_rows: int,
        beam_cols: int,
        weights_used: str,
        payload: dict,
    ) -> None:
        self.path = path
        self.model = model
        self.architecture = architecture
        self.beam_rows = int(beam_rows)
        self.beam_cols = int(beam_cols)
        self.weights_used = weights_used
        self.payload = payload

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))


def load_mlp(
    checkpoint_path: str,
    device: torch.device,
    beam_rows: int,
    beam_cols: int,
    weights: str,
    warnings: List[str],
) -> LoadedMLP:
    """Rebuild the MLP from whatever ``train_MLP.py`` stored in the file."""
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"[eval_measured] MLP checkpoint is missing: {checkpoint_path}\n"
            f"                Train it first (python train_MLP_indoor.py), or pass "
            f"--mlp_ckpt <model.pth>."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # A bare state_dict is a legal payload; so is the full dict train_MLP.py
    # writes.  Both end up as {tensor name: tensor} plus an optional config.
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored_config = payload.get("config") or {}
    else:
        state_dict = payload
        stored_config = {}
        payload = {"state_dict": state_dict}

    if weights == "best":
        best_state = payload.get("best_state_dict")
        if best_state is None:
            warnings.append(
                "WARN --mlp_weights best requested but the checkpoint has no "
                "'best_state_dict'; the final weights were used instead."
            )
            weights = "final"
        else:
            state_dict = best_state

    inferred = infer_mlp_architecture(state_dict)
    print(
        f"[eval_measured] MLP architecture inferred from the tensor shapes: "
        f"hidden={inferred['hidden']} depth={inferred['depth']} "
        f"num_outputs={inferred['num_outputs']} "
        f"(PE input {inferred['input_dim']} -> num_frequencies="
        f"{inferred['num_frequencies']}, include_input={inferred['include_input']})"
    )
    if stored_config:
        print(f"[eval_measured] MLP architecture stored in the checkpoint : {dict(stored_config)}")
        for key in ("hidden", "depth", "num_outputs", "num_frequencies", "include_input"):
            if key in stored_config and stored_config[key] != inferred[key]:
                warnings.append(
                    f"WARN MLP config mismatch on '{key}': the file says "
                    f"{stored_config[key]!r} but the weights say {inferred[key]!r}; "
                    f"the weights win."
                )
    else:
        print(
            "[eval_measured] The MLP checkpoint stores no config block; every "
            "constructor argument above was inferred from the tensor shapes."
        )

    model = PositionMLP(
        num_outputs=int(inferred["num_outputs"]),
        hidden=int(inferred["hidden"]),
        depth=int(inferred["depth"]),
        num_frequencies=int(inferred["num_frequencies"]),
        include_input=bool(inferred["include_input"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = int(stored_config.get("beam_rows", beam_rows))
    cols = int(stored_config.get("beam_cols", beam_cols))
    if rows * cols != int(inferred["num_outputs"]):
        raise SystemExit(
            f"[eval_measured] The MLP emits {inferred['num_outputs']} values but the "
            f"beam grid is {rows} x {cols} = {rows * cols}."
        )

    return LoadedMLP(
        checkpoint_path, model, inferred, rows, cols, weights, payload
    )


# ---------------------------------------------------------------------------
# Prediction paths
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def beam_grid_forwarding(model_params) -> Iterator[str]:
    """Make ``eval_render``'s render path honour a non-DFT beam grid.

    ``eval_render.render_batch`` calls ``render_fast`` without a
    ``beam_grid_mode``, so it always builds the DFT bins.  For a checkpoint
    trained on the measured steering codebook that is the wrong dictionary
    entirely, so the module-level function is swapped for one that forwards the
    restored beam grid -- the same trick ``eval_render.gain_net_width`` uses to
    load a checkpoint whose gain MLP no longer matches the current default.
    Everything inside the ``with`` block, including ``eval_render``'s own
    ``evaluate_test_set``, then renders on the checkpoint's real beam centers.

    A DFT checkpoint yields unchanged: the stock code path is already correct.

    Note that ``beam_splat`` forces its PyTorch reference implementation
    whenever ``periodic`` is false, so a custom-angle run silently drops the
    fused CUDA kernel and is slower.  That is a correctness requirement -- the
    kernel wraps beam deltas modulo 2, which only holds for a DFT grid.
    """
    mode = str(getattr(model_params, "beam_grid_mode", "dft") or "dft").lower()
    if mode not in ("dft", "custom_angles"):
        raise SystemExit(f"[eval_measured] Unknown beam_grid_mode {mode!r}.")

    if mode == "dft":
        yield mode
        return

    az = parse_angle_list(getattr(model_params, "beam_az_deg", ""), MEASURED_BEAM_AZ_DEG)
    el = parse_angle_list(getattr(model_params, "beam_el_deg", ""), MEASURED_BEAM_EL_DEG)
    original = ER.render_batch

    def render_batch_custom_angles(
        rx_pos: torch.Tensor,
        tx_pos: torch.Tensor,
        gaussians,
        scene,
        params,
        use_cuda_rasterizer: bool,
    ) -> torch.Tensor:
        """``eval_render.render_batch``, plus the three beam-grid arguments."""
        rendered = render_fast(
            rx_pos=rx_pos.reshape(-1, 3),
            tx_pos=tx_pos,
            pc=gaussians,
            rx_shape=scene.rx_shape,
            tx_shape=scene.tx_shape,
            covariance_floor=1e-4,
            weight_floor=1e-4,
            max_active_rx_beams=int(getattr(params, "max_active_rx_beams", 2)),
            max_active_tx_beams=int(getattr(params, "max_active_tx_beams", 2)),
            use_cuda_rasterizer=use_cuda_rasterizer,
            beam_grid_mode=mode,
            beam_az_deg=az,
            beam_el_deg=el,
        )
        predicted = rendered["render"]
        if predicted.ndim == 2:
            predicted = predicted.unsqueeze(0)
        return predicted

    ER.render_batch = render_batch_custom_angles
    try:
        yield mode
    finally:
        ER.render_batch = original


def render_mimogs_maps(
    loaded: LoadedMIMOGS, normalized_positions: torch.Tensor, batch_size: int
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
    device = next(loaded.model.parameters()).device
    with torch.no_grad():
        for start in range(0, int(normalized_positions.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(normalized_positions.shape[0]))
            chunks.append(
                loaded.model(normalized_positions[start:stop].to(device)).float()
            )
    return torch.cat(chunks, dim=0).reshape(-1, loaded.beam_rows, loaded.beam_cols)


# ---------------------------------------------------------------------------
# Beam-grid assertion
# ---------------------------------------------------------------------------
def assert_beam_grid(loaded: LoadedMIMOGS, ground_truth: TestGroundTruth) -> Dict[str, object]:
    """Assert the centers the renderer builds match the dataset's beam count.

    The centers are rebuilt here exactly the way ``render_fast`` builds them
    for the restored ``beam_grid_mode`` -- DFT bins from the array shape, or
    the measured steering codebook with its per-side azimuth sign -- so the
    assertion covers the grid actually used for scoring, not a stand-in.
    """
    scene = loaded.scene
    mode = loaded.beam_grid_mode
    device = loaded.gaussians.get_xyz.device
    dtype = loaded.gaussians.get_xyz.dtype

    if mode not in ("dft", "custom_angles"):
        raise SystemExit(f"[eval_measured] Unknown beam_grid_mode {mode!r}.")

    # The measured codebook the dataset was recorded with.  Built even on the
    # DFT path, so its beam count is checked against the data either way.
    az = parse_angle_list(getattr(loaded.model_params, "beam_az_deg", ""), MEASURED_BEAM_AZ_DEG)
    el = parse_angle_list(getattr(loaded.model_params, "beam_el_deg", ""), MEASURED_BEAM_EL_DEG)
    measured_centers = _build_custom_uv_grid(az, el, device=device, dtype=dtype, side="rx")

    if mode == "custom_angles":
        rx_centers = measured_centers
        tx_centers = _build_custom_uv_grid(az, el, device=device, dtype=dtype, side="tx")
    else:
        rx_centers = _build_beam_uv_grid(
            scene.rx_shape[0], scene.rx_shape[1], device=device, dtype=dtype
        )
        tx_centers = _build_beam_uv_grid(
            scene.tx_shape[0], scene.tx_shape[1], device=device, dtype=dtype
        )

    expected_rows = int(ground_truth.beam_rows)
    expected_cols = int(ground_truth.beam_cols)

    for label, centers, expected in (
        ("Rx", rx_centers, expected_rows),
        ("Tx", tx_centers, expected_cols),
    ):
        if int(centers.shape[0]) != expected:
            raise AssertionError(
                f"[eval_measured] The renderer builds {int(centers.shape[0])} {label} "
                f"beam centers but the dataset has {expected}."
            )
    if int(scene.beam_rows) != expected_rows or int(scene.beam_cols) != expected_cols:
        raise AssertionError(
            f"[eval_measured] Scene reports a {scene.beam_rows} x {scene.beam_cols} "
            f"beam grid but test.mat holds {expected_rows} x {expected_cols}."
        )
    if int(measured_centers.shape[0]) != expected_rows:
        raise AssertionError(
            f"[eval_measured] The measured codebook gives {len(az)} x {len(el)} = "
            f"{int(measured_centers.shape[0])} beams, but the dataset has "
            f"{expected_rows}."
        )

    # beam_splat forces its PyTorch reference path for non-periodic centers,
    # so a custom-angle checkpoint never reaches the fused CUDA kernel.
    renderer_path = (
        "CUDA kernel"
        if mode == "dft" and loaded.use_cuda_rasterizer
        else "PyTorch reference"
    )

    return {
        "mode": mode,
        "rx_shape": tuple(int(v) for v in scene.rx_shape),
        "tx_shape": tuple(int(v) for v in scene.tx_shape),
        "num_rx_centers": int(rx_centers.shape[0]),
        "num_tx_centers": int(tx_centers.shape[0]),
        "num_az": len(az),
        "num_el": len(el),
        "az_deg": tuple(float(v) for v in az),
        "el_deg": tuple(float(v) for v in el),
        "renderer_path": renderer_path,
    }


# ---------------------------------------------------------------------------
# Nearest-train distance statistics
# ---------------------------------------------------------------------------
def distance_statistics(distances: np.ndarray) -> Dict[str, float]:
    """min / median / p90 (plus mean and max) of the nearest-train distances."""
    values = np.asarray(distances, dtype=np.float64)
    return {
        "min_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, DISTANCE_PERCENTILE)),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_all(
    arguments: argparse.Namespace, device: torch.device, warnings: List[str]
) -> Dict[str, object]:
    """Score MIMO-GS, the MLP and the nearest-neighbour baseline on the test split."""
    dataset_dir = os.path.abspath(arguments.dataset or DEFAULT_DATASET_DIR)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[eval_measured] Dataset directory is missing: {dataset_dir}")

    ground_truth = TestGroundTruth(dataset_dir, device)
    train_positions, train_magnitude = load_raw_mat(os.path.join(dataset_dir, "train.mat"))

    print(f"[eval_measured] dataset         : {dataset_dir}")
    print(
        f"[eval_measured] train / test    : {int(train_positions.shape[0])} / "
        f"{len(ground_truth)} locations "
        f"(scored {ground_truth.num_scored}, skipped zero-power "
        f"{ground_truth.num_skipped_zero_power})"
    )
    print(
        f"[eval_measured] beam grid       : "
        f"{ground_truth.beam_rows} x {ground_truth.beam_cols}"
    )

    # -- MIMO-GS ----------------------------------------------------------
    loaded_gs = load_mimogs(arguments.mimogs_ckpt, device, dataset_dir)
    beam_grid = assert_beam_grid(loaded_gs, ground_truth)
    print(
        f"[eval_measured] MIMO-GS         : {os.path.relpath(loaded_gs.path, REPO_ROOT)} "
        f"| iteration {loaded_gs.iteration} | {loaded_gs.num_gaussians} gaussians "
        f"| beam_grid_mode={beam_grid['mode']} "
        f"rx{beam_grid['rx_shape']} tx{beam_grid['tx_shape']} "
        f"| cuda_rasterizer={int(loaded_gs.use_cuda_rasterizer)} "
        f"| render path: {beam_grid['renderer_path']}"
    )
    if beam_grid["mode"] == "custom_angles":
        print(
            f"[eval_measured] beam centers    : measured steering codebook, "
            f"{beam_grid['num_az']} az x {beam_grid['num_el']} el = "
            f"{beam_grid['num_rx_centers']} beams per side (non-DFT); "
            f"eval_render's render path is wrapped to forward it."
        )

    # The Scene the checkpoint was restored with must see the same test split
    # the shared TestGroundTruth loads, or the two halves of this script would
    # be scoring different locations.
    scene_positions = (
        loaded_gs.scene.test_set.positions.cpu().numpy().astype(np.float64)
        * float(getattr(loaded_gs.scene.test_set, "scale_factor", 1.0))
    )
    if scene_positions.shape != ground_truth.positions_m.shape or not np.allclose(
        scene_positions, ground_truth.positions_m, atol=1e-4
    ):
        raise AssertionError(
            "[eval_measured] Scene's test split does not match the test.mat this "
            "script scores; the two loading paths disagree."
        )

    with beam_grid_forwarding(loaded_gs.model_params):
        gs_maps = render_mimogs_maps(
            loaded_gs, ground_truth.positions_normalized, arguments.batch_size
        )
    assert_finite_nonnegative(gs_maps, METHOD_MIMOGS)
    gs_scored = ground_truth.score(gs_maps)

    # -- MLP --------------------------------------------------------------
    loaded_mlp = load_mlp(
        arguments.mlp_ckpt,
        device,
        ground_truth.beam_rows,
        ground_truth.beam_cols,
        arguments.mlp_weights,
        warnings,
    )
    print(
        f"[eval_measured] MLP             : "
        f"{os.path.relpath(loaded_mlp.path, REPO_ROOT)} | "
        f"{loaded_mlp.parameter_count:,} parameters | "
        f"weights={loaded_mlp.weights_used}"
    )
    mlp_maps = predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
    assert_finite_nonnegative(mlp_maps, METHOD_MLP)
    mlp_scored = ground_truth.score(mlp_maps)

    # -- Nearest neighbour -------------------------------------------------
    nn_maps, nn_distance = nearest_neighbour_maps(
        train_positions, train_magnitude, ground_truth.positions_m, device
    )
    assert_finite_nonnegative(nn_maps, METHOD_NN)
    nn_scored = ground_truth.score(nn_maps)

    valid = torch.as_tensor(ground_truth.valid_indices, device=device)
    maps: Dict[str, torch.Tensor] = {
        METHOD_GT: ground_truth.magnitude[valid],
        METHOD_MIMOGS: gs_maps[valid],
        METHOD_MLP: mlp_maps[valid],
        METHOD_NN: nn_maps[valid],
    }
    scored: Dict[str, Dict[str, np.ndarray]] = {
        METHOD_MIMOGS: gs_scored,
        METHOD_MLP: mlp_scored,
        METHOD_NN: nn_scored,
    }

    return {
        "dataset_dir": dataset_dir,
        "ground_truth": ground_truth,
        "loaded_gs": loaded_gs,
        "loaded_mlp": loaded_mlp,
        "beam_grid": beam_grid,
        "maps": maps,
        "scored": scored,
        "n_train": int(train_positions.shape[0]),
        "nn_distance_all": nn_distance.astype(np.float64),
        "nn_distance_scored": nn_distance.astype(np.float64)[ground_truth.valid_indices],
        "checkpoints": {
            METHOD_MIMOGS: os.path.relpath(loaded_gs.path, REPO_ROOT),
            METHOD_MLP: os.path.relpath(loaded_mlp.path, REPO_ROOT),
            METHOD_NN: "(no learning)",
        },
    }


def build_table_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    """One row per method, in the order the paper table lists them."""
    ground_truth: TestGroundTruth = results["ground_truth"]
    rows: List[Dict[str, object]] = []
    for method in TABLE_ORDER:
        row = summarize_scores(results["scored"][method])
        row.update(
            {
                "method": method,
                "checkpoint": results["checkpoints"][method],
                "n_train": int(results["n_train"]),
                "n_test": int(ground_truth.num_scored),
            }
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sanity block -- the same checkpoint through eval_render's own scorer
# ---------------------------------------------------------------------------
def sanity_cross_check(
    results: Dict[str, object], device: torch.device, batch_size: int, warnings: List[str]
) -> Dict[str, object]:
    """Score MIMO-GS again with ``eval_render.evaluate_test_set`` and compare.

    ``evaluate_test_set`` is eval_render's own end-to-end path: its own loader,
    its own batching, its own metric block.  It renders through the same
    forwarded beam grid, so the comparison isolates this script's plumbing
    rather than re-testing the beam-grid question.
    """
    loaded: LoadedMIMOGS = results["loaded_gs"]
    here = float(np.mean(results["scored"][METHOD_MIMOGS]["nmse_shape_db"]))

    with beam_grid_forwarding(loaded.model_params):
        reference_results = ER.evaluate_test_set(
            loaded.scene,
            loaded.gaussians,
            loaded.model_params,
            device,
            batch_size,
            loaded.use_cuda_rasterizer,
        )
    reference = float(np.mean(reference_results["nmse_shape_db"]))
    delta = abs(here - reference)
    status = "ok" if delta <= SELF_CHECK_TOLERANCE_DB else "MISMATCH"

    if status != "ok":
        warnings.append(
            f"WARN cross-check MISMATCH: {here:.4f} dB here vs. {reference:.4f} dB "
            f"through eval_render.evaluate_test_set (delta {delta:.4f} dB, tolerance "
            f"{SELF_CHECK_TOLERANCE_DB:.2f} dB)."
        )

    return {
        "here_dB": here,
        "eval_render_dB": reference,
        "delta_dB": delta,
        "status": status,
        "num_scored_reference": int(reference_results["index"].shape[0]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def to_linear_panel(single_map: torch.Tensor) -> np.ndarray:
    """Max-normalized map in [0, 1]."""
    return normalize_mag_map(single_map.unsqueeze(0))[0].detach().cpu().numpy()


def to_db_panel(single_map: torch.Tensor) -> np.ndarray:
    """``10*log10`` of the max-normalized map, floored at ``DB_FLOOR``."""
    linear = to_linear_panel(single_map)
    with np.errstate(divide="ignore"):
        decibel = 10.0 * np.log10(np.maximum(linear, 1e-12))
    return np.maximum(decibel, DB_FLOOR)


def to_error_panel(gt_map: torch.Tensor, predicted_map: torch.Tensor) -> np.ndarray:
    """Absolute error, normalized by the ground-truth maximum of the location.

    Both maps are divided by their own maximum first, which is exactly the pair
    of panels drawn above the error row and the convention the headline shape
    NMSE is computed in.  The ground-truth panel then has unit maximum, so this
    is ``|GT - rendered|`` expressed in units of that location's GT peak.  The
    error row is always linear, in both the linear and the dB figure.
    """
    return np.abs(to_linear_panel(gt_map) - to_linear_panel(predicted_map))


def panel_limits(scale: str) -> Tuple[float, float, str]:
    if scale == "linear":
        return 0.0, 1.0, "Normalized power"
    return DB_FLOOR, 0.0, "Normalized power [dB]"


def compute_error_vmax(results: Dict[str, object]) -> float:
    """One error-panel upper limit for the whole run, derived from the data.

    Taken over EVERY scored test location, not only the drawn ones, so the
    scale does not move when the gallery selection changes.
    """
    errors = (
        normalize_mag_map(results["maps"][METHOD_GT])
        - normalize_mag_map(results["maps"][METHOD_MIMOGS])
    ).abs()
    percentile = float(
        np.percentile(errors.detach().cpu().numpy(), ERROR_VMAX_PERCENTILE)
    )
    stepped = np.ceil(percentile / ERROR_VMAX_STEP) * ERROR_VMAX_STEP
    return float(min(max(stepped, ERROR_VMAX_MIN), ERROR_VMAX_MAX))


def render_gallery_figure(
    output_dir: str, row: int, results: Dict[str, object], scale: str
) -> str:
    """One candidate figure: GT / MIMO-GS / absolute error at one location."""
    ground_truth: TestGroundTruth = results["ground_truth"]
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = to_linear_panel if scale == "linear" else to_db_panel
    error_vmax = float(results["error_vmax"])

    test_index = int(ground_truth.valid_indices[row])
    position = ground_truth.valid_positions_m[row]
    gt_map = results["maps"][METHOD_GT][row]
    gs_map = results["maps"][METHOD_MIMOGS][row]

    panels = [
        (METHOD_GT, convert(gt_map), vmin, vmax),
        (METHOD_MIMOGS, convert(gs_map), vmin, vmax),
        (METHOD_ERROR, to_error_panel(gt_map, gs_map), 0.0, error_vmax),
    ]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(PANEL_INCH + PANEL_MARGIN_W, 3.0 * PANEL_INCH + PANEL_MARGIN_H),
        squeeze=False,
        layout="constrained",
    )

    map_image = None
    error_image = None
    for panel, (label, data, panel_vmin, panel_vmax) in enumerate(panels):
        axis = axes[panel][0]
        image = axis.imshow(
            data,
            aspect="equal",
            interpolation="nearest",
            vmin=panel_vmin,
            vmax=panel_vmax,
            cmap="viridis" if label != METHOD_ERROR else "magma",
        )
        if label == METHOD_ERROR:
            error_image = image
        else:
            map_image = image

        axis.set_ylabel(label, fontsize=AXIS_LABEL_FONTSIZE)
        axis.tick_params(labelsize=TICK_LABELSIZE)
        if panel == len(panels) - 1:
            axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            axis.set_xticklabels([])

    # Above the panels rather than inside them: an overlay box would cover the
    # first beam rows of the ground-truth map, which are real data.  A figure
    # suptitle spans the full width, so the caption is never clipped by the
    # narrow axes column.
    figure.suptitle(
        f"test index {test_index}   "
        f"(x, y, z) = ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m\n"
        f"shape NMSE: MIMO-GS "
        f"{float(results['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):.2f} dB   "
        f"MLP {float(results['scored'][METHOD_MLP]['nmse_shape_db'][row]):.2f} dB",
        fontsize=9,
    )

    colorbar = figure.colorbar(
        map_image, ax=[axes[0][0], axes[1][0]], fraction=0.045, pad=0.015
    )
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    error_bar = figure.colorbar(error_image, ax=axes[2][0], fraction=0.045, pad=0.015)
    error_bar.set_label("|error| / GT max", fontsize=10)
    error_bar.ax.tick_params(labelsize=9)

    target_dir = os.path.join(output_dir, "gallery", scale)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"loc_{test_index}.png")
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def render_sample_grid(
    output_dir: str, rows: Sequence[int], results: Dict[str, object], scale: str
) -> None:
    """The final publication figure: chosen locations as columns.

    Two rows only -- ground truth and MIMO-GS.  The error row belongs to the
    gallery, where it helps rank locations; in the paper figure the two map
    rows carry the comparison and one shared colorbar, and the per-location
    error is already quantified by the NMSE column of the table.
    """
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = to_linear_panel if scale == "linear" else to_db_panel
    row_labels = (METHOD_GT, METHOD_MIMOGS)

    figure, axes = plt.subplots(
        len(row_labels),
        len(rows),
        figsize=(
            PANEL_INCH * len(rows) + PANEL_MARGIN_W,
            PANEL_INCH * len(row_labels) + PANEL_MARGIN_H_FINAL,
        ),
        squeeze=False,
        layout="constrained",
    )

    map_image = None
    for column, row in enumerate(rows):
        panels = [
            (METHOD_GT, convert(results["maps"][METHOD_GT][row])),
            (METHOD_MIMOGS, convert(results["maps"][METHOD_MIMOGS][row])),
        ]

        for panel, (label, data) in enumerate(panels):
            axis = axes[panel][column]
            map_image = axis.imshow(
                data,
                aspect="equal",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )

            axis.tick_params(labelsize=TICK_LABELSIZE)
            if panel == 0:
                axis.set_title(f"Spot {column + 1}", fontsize=COLUMN_TITLE_FONTSIZE)
            if column == 0:
                axis.set_ylabel(label, fontsize=AXIS_LABEL_FONTSIZE)
            else:
                axis.set_yticklabels([])
            if panel == len(panels) - 1:
                axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                axis.set_xticklabels([])

    colorbar = figure.colorbar(
        map_image,
        ax=[axis for row_axes in axes for axis in row_axes],
        fraction=0.030,
        pad=0.012,
    )
    # The colorbar is a labelled axis like any other here, so its label and
    # ticks follow the axis convention rather than the gallery's smaller sizes.
    colorbar.set_label(colorbar_label, fontsize=AXIS_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig_measured_samples_{scale}"
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=FIGURE_DPI)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# Gallery selection
# ---------------------------------------------------------------------------
def select_gallery_rows(
    results: Dict[str, object], top_n: int
) -> Tuple[List[int], str]:
    """Which locations to draw, best rendering quality first.

    ``top_n <= 0`` draws every scored location, which is the default: choosing
    the final figure's spots means looking at all of them.  The ordering is by
    MIMO-GS shape NMSE either way, so the printed candidate table doubles as a
    ranking whether or not it was truncated.
    """
    shape_db = np.asarray(
        results["scored"][METHOD_MIMOGS]["nmse_shape_db"], dtype=np.float64
    )
    count = int(shape_db.shape[0])
    order = np.argsort(shape_db, kind="stable")

    if int(top_n) <= 0 or int(top_n) >= count:
        return [int(row) for row in order], (
            f"every one of the {count} scored test locations, listed by rendering "
            f"quality (lowest MIMO-GS shape NMSE first)"
        )

    return [int(row) for row in order[: int(top_n)]], (
        f"the top {int(top_n)} of {count} scored test locations by rendering "
        f"quality (lowest MIMO-GS shape NMSE first), as requested by --gallery_top"
    )


def print_gallery_candidates(
    rows: Sequence[int], results: Dict[str, object], reason: str
) -> None:
    """List the drawn locations so ``--spots`` can be chosen from the terminal."""
    ground_truth: TestGroundTruth = results["ground_truth"]

    print("")
    print("-" * 118)
    print(f"[eval_measured] GALLERY CANDIDATES -- {reason}")
    print("-" * 118)
    header = (
        f"  {'test idx':>9}{'x [m]':>9}{'y [m]':>9}{'z [m]':>9}"
        f"{'GS [dB]':>11}{'MLP [dB]':>11}{'NN [dB]':>11}"
        f"{'GS top-1':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        position = ground_truth.valid_positions_m[row]
        print(
            f"  {int(ground_truth.valid_indices[row]):>9}"
            f"{position[0]:>9.2f}{position[1]:>9.2f}{position[2]:>9.2f}"
            f"{float(results['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):>11.2f}"
            f"{float(results['scored'][METHOD_MLP]['nmse_shape_db'][row]):>11.2f}"
            f"{float(results['scored'][METHOD_NN]['nmse_shape_db'][row]):>11.2f}"
            f"{float(results['scored'][METHOD_MIMOGS]['topk_acc_K1'][row]):>10.2f}"
        )
    print("  " + "-" * (len(header) - 2))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_table_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    records = []
    for row in rows:
        record = []
        for column in TABLE_COLUMNS:
            value = row.get(column, "")
            record.append(f"{value:.6f}" if isinstance(value, float) else value)
        records.append(record)
    write_csv(path, TABLE_COLUMNS, records)


def write_per_location_csv(path: str, results: Dict[str, object]) -> None:
    """Per-location numbers, so the ``--spots`` choice can be made from a file."""
    ground_truth: TestGroundTruth = results["ground_truth"]
    header = ["test_index", "x_m", "y_m", "z_m", "nn_distance_m"]
    for method in TABLE_ORDER:
        header += [
            f"nmse_shape_dB_{method}",
            f"nmse_raw_dB_{method}",
            f"topk_acc_K1_{method}",
        ]

    records = []
    for row in range(ground_truth.num_scored):
        position = ground_truth.valid_positions_m[row]
        record = [
            int(ground_truth.valid_indices[row]),
            f"{position[0]:.6f}",
            f"{position[1]:.6f}",
            f"{position[2]:.6f}",
            f"{float(results['nn_distance_scored'][row]):.6f}",
        ]
        for method in TABLE_ORDER:
            scored = results["scored"][method]
            record += [
                f"{float(scored['nmse_shape_db'][row]):.6f}",
                f"{float(scored['nmse_raw_db'][row]):.6f}",
                f"{float(scored['topk_acc_K1'][row]):.6f}",
            ]
        records.append(record)

    write_csv(path, header, records)


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    print("")
    print("=" * 118)
    print("[eval_measured] MEASURED-DATASET TABLE (test split, 63 x 63 beam-pair maps)")
    print("=" * 118)
    header = (
        f"  {'method':<18}{'shape mean':>12}{'shape med.':>12}{'shape m-lin':>13}"
        f"{'raw mean':>11}{'raw med.':>11}"
        + "".join(f"{'top-' + str(k):>9}" for k in TOPK_REPORTED)
        + "".join(f"{'cap@' + str(k):>9}" for k in CAPTURE_REPORTED)
        + f"{'n_train':>9}{'n_test':>8}"
    )
    print(header)
    print(
        f"  {'':<18}{'[dB]':>12}{'[dB]':>12}{'[dB]':>13}{'[dB]':>11}{'[dB]':>11}"
        + "".join(f"{'':>9}" for _ in TOPK_REPORTED)
        + "".join(f"{'':>9}" for _ in CAPTURE_REPORTED)
        + f"{'':>9}{'':>8}"
    )
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        line = (
            f"  {str(row['method']):<18}"
            f"{float(row['nmse_shape_mean_dB']):>12.3f}"
            f"{float(row['nmse_shape_median_dB']):>12.3f}"
            f"{float(row['nmse_shape_meanlinear_dB']):>13.3f}"
            f"{float(row['nmse_raw_mean_dB']):>11.3f}"
            f"{float(row['nmse_raw_median_dB']):>11.3f}"
        )
        line += "".join(f"{float(row[f'topk_acc_K{k}']):>9.4f}" for k in TOPK_REPORTED)
        line += "".join(
            f"{float(row[f'power_capture_K{k}']):>9.4f}" for k in CAPTURE_REPORTED
        )
        line += f"{int(row['n_train']):>9}{int(row['n_test']):>8}"
        print(line)
    print("  " + "-" * (len(header) - 2))
    print("  NMSE convention : max-normalized prediction vs. max-normalized target,")
    print("                    averaged per location in dB (eval_render 'shape').")
    print("                    'm-lin' averages the linear NMSE and converts once.")
    print("=" * 118)


def build_readme(
    results: Dict[str, object],
    table_rows: Sequence[Dict[str, object]],
    distance_stats: Dict[str, float],
    sanity: Dict[str, object],
    gallery_paths: Sequence[str],
    gallery_locations: int,
    gallery_rendered: bool,
    gallery_reason: str,
    spot_indices: Sequence[int],
    device: torch.device,
    warnings: Sequence[str],
) -> List[str]:
    ground_truth: TestGroundTruth = results["ground_truth"]
    beam_grid = results["beam_grid"]
    loaded_gs: LoadedMIMOGS = results["loaded_gs"]
    loaded_mlp: LoadedMLP = results["loaded_mlp"]
    by_method = {str(row["method"]): row for row in table_rows}

    lines = [
        "eval_measured -- measured 60 GHz indoor dataset: table + rendering samples",
        "=" * 78,
        "",
        "DATASET",
        f"  Directory        : {results['dataset_dir']}",
        f"  Train locations  : {int(results['n_train'])}",
        f"  Test locations   : {len(ground_truth)} "
        f"({ground_truth.num_scored} scored, "
        f"{ground_truth.num_skipped_zero_power} skipped for zero power)",
        f"  Map size         : {ground_truth.beam_rows} x {ground_truth.beam_cols} "
        f"beam pairs (Rx x Tx)",
        "  Split            : the prebaked train.mat / test.mat pair; no random",
        "                     splitting happens anywhere in this script.",
        "",
        "  Nearest-train distance of the test set (3D Euclidean, original meters):",
        f"    min    {distance_stats['min_m']:.4f} m",
        f"    median {distance_stats['median_m']:.4f} m",
        f"    p90    {distance_stats['p90_m']:.4f} m",
        f"    (mean  {distance_stats['mean_m']:.4f} m, max {distance_stats['max_m']:.4f} m)",
        "  These document the split style: the smaller they are, the more the test",
        "  set is interleaved with the training grid rather than held out as a",
        "  separate region.",
        "",
        "BEAM GRID",
        f"  The measured codebook is {MEASURED_NUM_AZ} azimuth x {MEASURED_NUM_EL} "
        f"elevation = {MEASURED_NUM_AZ * MEASURED_NUM_EL} analog steering",
        "  directions per side, emitted azimuth-fastest, i.e. NOT a DFT beam grid:",
        f"    azimuth   : {beam_grid['az_deg'][0]:g} .. {beam_grid['az_deg'][-1]:g} deg "
        f"in {len(beam_grid['az_deg'])} steps",
        f"    elevation : "
        f"{', '.join(f'{value:g}' for value in beam_grid['el_deg'])} deg",
        f"  Renderer beam centers actually used for the MIMO-GS row: "
        f"beam_grid_mode='{beam_grid['mode']}'",
        f"    {beam_grid['num_rx_centers']} Rx centers, "
        f"{beam_grid['num_tx_centers']} Tx centers; render path: "
        f"{beam_grid['renderer_path']}.",
        "  The count is asserted against the dataset's beam count before scoring,",
        "  and the centers are rebuilt exactly the way render_fast builds them.",
    ]

    if beam_grid["mode"] == "custom_angles":
        lines += [
            "  The checkpoint was trained on the measured codebook, so the renderer",
            "  uses the same non-DFT steering directions the data was recorded with.",
            "  eval_render.render_batch does not forward beam_grid_mode to render_fast,",
            "  so this script wraps that function for the duration of the evaluation",
            "  (the way eval_render.gain_net_width swaps DynamicGainNet); both the",
            "  scored render and the eval_render cross-check below go through it.",
            "  beam_splat forces its PyTorch reference path for non-periodic centers --",
            "  the fused CUDA kernel wraps beam deltas modulo 2, which only holds for a",
            "  DFT grid -- so this run is correct but slower than a DFT one.",
        ]
    else:
        lines += [
            "  NOTE: this checkpoint was trained with the DFT beam centers of a",
            f"  {beam_grid['rx_shape'][0]}x{beam_grid['rx_shape'][1]} UPA, not with the "
            "measured steering codebook.  The beam",
            f"  COUNT matches the data ({beam_grid['num_rx_centers']} per side) but the "
            "beam DIRECTIONS do not, so",
            "  the MIMO-GS row is a DFT-grid model scored on measured-codebook data.",
            "  The MLP never looks at beam geometry -- it regresses the flattened map --",
            "  so its row is unaffected by the mode.",
        ]

    lines += [
        "",
        "CHECKPOINTS",
        f"  MIMO-GS : {results['checkpoints'][METHOD_MIMOGS]}",
        f"            run dir {os.path.relpath(loaded_gs.run_dir, REPO_ROOT)}, "
        f"iteration {loaded_gs.iteration}, {loaded_gs.num_gaussians} gaussians,",
        f"            config restored from run_args.txt + the checkpoint payload "
        f"(eval_render.restore_config).",
        f"  MLP     : {results['checkpoints'][METHOD_MLP]}",
        f"            PositionMLP hidden={loaded_mlp.architecture['hidden']} "
        f"depth={loaded_mlp.architecture['depth']} "
        f"outputs={loaded_mlp.architecture['num_outputs']} "
        f"(PE {loaded_mlp.architecture['input_dim']} dims: "
        f"num_frequencies={loaded_mlp.architecture['num_frequencies']}, "
        f"include_input={loaded_mlp.architecture['include_input']}),",
        f"            {loaded_mlp.parameter_count:,} parameters, "
        f"'{loaded_mlp.weights_used}' weights.  The architecture is inferred from the",
        "            tensor shapes and cross-checked against the stored config block.",
        "  Nearest neighbor : no learning.  Each test map is predicted as the train map",
        "                     at the nearest train position (3D Euclidean, original",
        "                     meters, full training set).",
        "",
        "CONVENTIONS",
        "  Metric      : shape NMSE, i.e. max-normalized prediction vs. max-normalized",
        "                target, averaged per location in dB.  Imported from",
        "                evaluation/eval_render.py (via eval_baseline_rt.score_prediction),",
        "                never reimplemented, so this table aligns column-for-column",
        "                with the DeepMIMO table eval_density.py writes.",
        "  Also given  : raw/scale NMSE (raw prediction vs. normalized target), the",
        "                mean-linear shape NMSE, top-K overlap accuracy for K = "
        f"{', '.join(str(k) for k in TOPK_REPORTED)},",
        f"                and power capture for K = "
        f"{', '.join(str(k) for k in CAPTURE_REPORTED)}.",
        "  Guards      : every predictor's output is asserted finite and non-negative",
        "                before it reaches a metric; the renderer's beam count is",
        "                asserted against the dataset's.",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "  Figures     : gallery panels are ground truth / MIMO-GS / absolute error and",
        "                carry a small caption.  The final figure drops the error row --",
        "                two rows, ground truth and MIMO-GS, one column per chosen",
        "                location -- and names its columns 'Spot k' at "
        f"{COLUMN_TITLE_FONTSIZE} pt.  Axis",
        f"                labels {AXIS_LABEL_FONTSIZE} pt, ticks {TICK_LABELSIZE} pt, "
        "colorbar label and ticks the",
        "                same.  Gallery PNG at 300 dpi, final figure PNG at",
        "                300 dpi plus PDF.",
        "  Scales      : 'linear' = each map divided by its own max, shared 0..1",
        "                colorbar for ground truth and rendered.  'db' = 10*log10 of",
        f"                that, floored at {DB_FLOOR:.0f} dB.  The gallery's error row is",
        "                the absolute difference of the two max-normalized maps, i.e.",
        "                |GT - rendered| in units of that location's GT peak.  It stays",
        "                LINEAR on both gallery scales, on one shared upper limit of",
        f"                {float(results['error_vmax']):.2f}, so error panels are "
        f"comparable across locations.  That",
        f"                limit is the p{ERROR_VMAX_PERCENTILE:g} of the absolute error "
        f"over all {ground_truth.num_scored} scored",
        f"                locations, rounded up to {ERROR_VMAX_STEP:g}; a fixed 0..1 "
        f"scale renders the row",
        "                black, because these maps are extremely peaky.",
        "",
        "HEADLINE NUMBERS (test split, mean shape NMSE [dB] / top-1 / power capture @K=4)",
    ]
    for method in TABLE_ORDER:
        row = by_method[method]
        lines.append(
            f"  {method:<18}{float(row['nmse_shape_mean_dB']):>10.3f} dB"
            f"{float(row['topk_acc_K1']):>10.4f}"
            f"{float(row['power_capture_K4']):>10.4f}"
        )

    lines += [
        "",
        "SANITY",
        f"  MIMO-GS shape NMSE, this script                  : "
        f"{float(sanity['here_dB']):.4f} dB",
        f"  MIMO-GS shape NMSE, eval_render.evaluate_test_set: "
        f"{float(sanity['eval_render_dB']):.4f} dB "
        f"({int(sanity['num_scored_reference'])} locations)",
        f"  delta                                            : "
        f"{float(sanity['delta_dB']):.4f} dB "
        f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB) -> {sanity['status']}",
        "  Both paths render the same checkpoint; the check validates this script's",
        "  batching and scoring against eval_render's own end-to-end path.",
        "",
        "FILES",
        "  measured_table.csv                     the table above, one row per method",
        "  metrics_per_location.csv               per-location numbers for every method",
        f"  gallery/linear/loc_<index>.png         {gallery_locations} candidate "
        f"figures, linear scale",
        f"  gallery/db/loc_<index>.png             the same locations, dB scale",
        "  fig_measured_samples_{linear,db}.*     final figure (--spots mode only)",
        "  README.txt                             this file",
        "",
        "GALLERY",
    ]
    if gallery_rendered:
        lines.append(
            f"  {gallery_locations} locations x 2 scales = "
            f"{len(gallery_paths)} files written in this run."
        )
    else:
        lines.append(
            "  Not re-rendered in this run (--spots was given without --gallery);"
        )
        lines.append("  anything under gallery/ is left over from an earlier run.")
    lines += [
        f"  Selection rule: {gallery_reason}.",
        "  Rows are ground truth / MIMO-GS rendered / absolute error.  The MLP maps are",
        "  never drawn; only its per-location shape NMSE is printed in the caption.",
        "",
        "FINAL FIGURE",
    ]
    lines.append(
        "  Two rows (ground truth, MIMO-GS) x one column per chosen location; no error"
    )
    lines.append("  row, column titles 'Spot k', one shared colorbar.")
    if spot_indices:
        lines.append(
            f"  Rendered for test indices "
            f"{', '.join(str(int(v)) for v in spot_indices)}."
        )
    else:
        lines.append("  Not requested in this run.")
    lines += [
        "",
        "WARNINGS",
    ]
    lines += [f"  {warning}" for warning in warnings] or ["  none"]
    lines += [
        "",
        "RERUN",
        "  python eval_measured.py",
        "  python eval_measured.py --spots <i>,<j>[,<k>]",
    ]
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_spot_argument(text: str) -> List[int]:
    tokens = [token for token in str(text).replace(",", " ").split() if token]
    values: List[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as error:
            raise SystemExit(
                f"[eval_measured] --spots value is not an integer: {token!r}"
            ) from error
    if len(values) not in (2, 3):
        raise SystemExit(
            f"[eval_measured] --spots takes 2 or 3 test indices, got {len(values)}."
        )
    if len(set(values)) != len(values):
        raise SystemExit("[eval_measured] --spots holds a duplicated test index.")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D4 -- measured 60 GHz indoor dataset: table + rendering samples"
    )
    parser.add_argument(
        "--mimogs_ckpt",
        type=str,
        default=DEFAULT_MIMOGS_CKPT,
        help=f"MIMO-GS run directory or model.pth (default: "
        f"{os.path.relpath(DEFAULT_MIMOGS_CKPT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--mlp_ckpt",
        type=str,
        default=DEFAULT_MLP_CKPT,
        help=f"MLP model.pth (default: {os.path.relpath(DEFAULT_MLP_CKPT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--mlp_weights",
        type=str,
        default="final",
        choices=("final", "best"),
        help="Which MLP weights to score: the final epoch, or the checkpoint's "
        "recorded best epoch when it stores one.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=f"Override the dataset directory (default: "
        f"{os.path.relpath(DEFAULT_DATASET_DIR, REPO_ROOT)})",
    )
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Rendering batch size for the MIMO-GS forward passes.",
    )
    parser.add_argument(
        "--spots",
        type=str,
        default="",
        help="Final-figure mode: 2 or 3 comma-separated test indices, "
        "e.g. --spots 12,34,56",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Also render the candidate gallery when --spots is given.",
    )
    parser.add_argument(
        "--gallery_top",
        type=int,
        default=GALLERY_TOP_DEFAULT,
        help="Draw only the N best-rendered test locations instead of all of "
        "them (0, the default, draws every scored location).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)
    warnings: List[str] = []

    print("=" * 118)
    print("[eval_measured] Measured 60 GHz indoor dataset -- table + rendering samples")
    print("=" * 118)
    print(
        f"[eval_measured] device          : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    # --spots is validated before any model is loaded, so a typo fails fast.
    requested_spots = parse_spot_argument(arguments.spots) if arguments.spots else []

    results = evaluate_all(arguments, device, warnings)
    ground_truth: TestGroundTruth = results["ground_truth"]
    results["error_vmax"] = compute_error_vmax(results)

    # -- nearest-train distance stats -------------------------------------
    distance_stats = distance_statistics(results["nn_distance_scored"])
    print("")
    print("-" * 118)
    print("[eval_measured] NEAREST-TRAIN DISTANCE OF THE TEST SET (3D Euclidean, meters)")
    print("-" * 118)
    print(
        f"  min {distance_stats['min_m']:.4f}   "
        f"median {distance_stats['median_m']:.4f}   "
        f"p90 {distance_stats['p90_m']:.4f}   "
        f"(mean {distance_stats['mean_m']:.4f}, max {distance_stats['max_m']:.4f})"
    )

    # -- table -------------------------------------------------------------
    table_rows = build_table_rows(results)

    # -- sanity ------------------------------------------------------------
    print("")
    print("-" * 118)
    print("[eval_measured] SANITY -- the same MIMO-GS checkpoint through eval_render")
    print("-" * 118)
    sanity = sanity_cross_check(results, device, arguments.batch_size, warnings)
    print(f"  this script                       : {float(sanity['here_dB']):.4f} dB")
    print(
        f"  eval_render.evaluate_test_set     : "
        f"{float(sanity['eval_render_dB']):.4f} dB "
        f"({int(sanity['num_scored_reference'])} locations)"
    )
    print(
        f"  delta                             : {float(sanity['delta_dB']):.4f} dB "
        f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB) -> {sanity['status']}"
    )

    output_dir = arguments.analysis_root
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    output_dir = os.path.join(output_dir, "eval_measured")
    os.makedirs(output_dir, exist_ok=True)

    write_table_csv(os.path.join(output_dir, "measured_table.csv"), table_rows)
    write_per_location_csv(
        os.path.join(output_dir, "metrics_per_location.csv"), results
    )
    print_table(table_rows)

    # -- figures -----------------------------------------------------------
    gallery_rows, gallery_reason = select_gallery_rows(results, arguments.gallery_top)
    gallery_paths: List[str] = []
    gallery_rendered = not requested_spots or arguments.gallery

    print("")
    print(
        f"[eval_measured] error-panel scale: 0..{float(results['error_vmax']):.2f} "
        f"(p{ERROR_VMAX_PERCENTILE:g} of |GT - rendered| over all "
        f"{ground_truth.num_scored} scored locations, rounded up to "
        f"{ERROR_VMAX_STEP:g}); shared by every panel."
    )

    if gallery_rendered:
        print(
            f"[eval_measured] rendering the candidate gallery: {gallery_reason} "
            f"x 2 scales..."
        )
        for row in gallery_rows:
            for scale in ("linear", "db"):
                path = render_gallery_figure(output_dir, row, results, scale)
                gallery_paths.append(path)
        print(
            f"[eval_measured] {len(gallery_paths)} gallery figures "
            f"({len(gallery_rows)} locations x 2 scales) written to "
            f"{os.path.join(output_dir, 'gallery', '{linear,db}')}"
        )
        print_gallery_candidates(gallery_rows, results, gallery_reason)
    else:
        print(
            "[eval_measured] candidate gallery not re-rendered (--spots without "
            "--gallery); anything under gallery/ is from an earlier run."
        )

    spot_rows: List[int] = []
    if requested_spots:
        lookup = {
            int(value): position
            for position, value in enumerate(ground_truth.valid_indices.tolist())
        }
        missing = [value for value in requested_spots if value not in lookup]
        if missing:
            raise SystemExit(
                f"[eval_measured] --spots refers to test indices that are not scored: "
                f"{missing}.  Scored indices run 0..{len(ground_truth) - 1}."
            )
        spot_rows = [lookup[value] for value in requested_spots]

        print("")
        print(
            f"[eval_measured] final-figure mode: test indices "
            f"{[int(ground_truth.valid_indices[row]) for row in spot_rows]}"
        )
        for scale in ("linear", "db"):
            render_sample_grid(output_dir, spot_rows, results, scale)
        print(
            f"[eval_measured] fig_measured_samples_{{linear,db}}.{{png,pdf}} written "
            f"to {output_dir}"
        )

    # -- README ------------------------------------------------------------
    write_readme(
        os.path.join(output_dir, "README.txt"),
        build_readme(
            results,
            table_rows,
            distance_stats,
            sanity,
            gallery_paths,
            len(gallery_rows) if gallery_rendered else 0,
            gallery_rendered,
            gallery_reason,
            [int(ground_truth.valid_indices[row]) for row in spot_rows],
            device,
            warnings,
        ),
    )

    print("")
    print("=" * 118)
    print("[eval_measured] SUMMARY")
    print("=" * 118)
    print(f"  {'method':<20}{'shape NMSE [dB]':>18}{'top-1':>10}{'capture@4':>12}")
    print("  " + "-" * 58)
    for row in table_rows:
        print(
            f"  {str(row['method']):<20}"
            f"{float(row['nmse_shape_mean_dB']):>18.3f}"
            f"{float(row['topk_acc_K1']):>10.4f}"
            f"{float(row['power_capture_K4']):>12.4f}"
        )
    print("  " + "-" * 58)
    print(
        f"  train / test locations : {int(results['n_train'])} / "
        f"{ground_truth.num_scored}"
    )
    print(
        f"  gallery figures written: {len(gallery_paths)}"
        + (
            f" ({len(gallery_rows)} locations x 2 scales)"
            if gallery_rendered
            else " (not re-rendered this run)"
        )
    )
    print(
        f"  final-figure spots     : "
        f"{[int(ground_truth.valid_indices[row]) for row in spot_rows] or 'not requested'}"
    )
    print("")
    if warnings:
        print(f"[eval_measured] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_measured] No warnings.")
    print(f"[eval_measured] Outputs written to {output_dir}")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    sys.exit(main())
