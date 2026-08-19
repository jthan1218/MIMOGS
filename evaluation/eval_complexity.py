#!/usr/bin/env python3
"""D4 -- rendering cost vs. beam-grid size.

Times one map at a time (batch 1) for MIMO-GS and for the ``mlp_medium``
coordinate MLP across five beam grids::

    (Nr, Nt) in {(4,16), (8,32), (16,64), (32,128), (64,256)}

and writes the scaling curve to ``analysis/eval_complexity/``.

Zero-argument runnable::

    python eval_complexity.py

Nothing in the repository is modified.  The MIMO-GS model is
``outputs/density/mimogs/model_100.pth``, loaded once and reused for every
grid.

How the non-native grids are built
----------------------------------
``gaussian_renderer/fast_renderer.render_fast`` builds its beam centers from
the Rx/Tx UPA shapes alone (``_build_beam_uv_grid`` takes the DFT u/v bins of a
``horizontal x vertical`` array), and ``Scene`` derives those shapes from the
beam counts with ``scene.square_array_shape``.  A grid of any size is therefore
constructible synthetically -- no retraining, no dataset -- by handing the
renderer the ``square_array_shape(Nr)`` / ``square_array_shape(Nt)`` lattice for
that resolution.  That is exactly what this script does; the primitives are
untouched, so the (16, 64) row reproduces the trained configuration.

Accuracy at non-native sizes is UNDEFINED for both methods: the MIMO-GS
primitives were fitted against a 16x64 codebook, and the MLP shells are
freshly initialized.  Only cost is measured here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_density import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_MIMOGS_DIR,
    LEGEND_FONTSIZE,
    METHOD_MIMOGS,
    METHOD_MLP,
    METHOD_STYLE,
    MLP_CONFIGS,
    PositionMLP,
    REPO_ROOT,
    load_mimogs,
    resolve_device,
    save_figure,
    style_axis,
    write_csv,
    write_readme,
)
import eval_render as ER
from scene import square_array_shape


BEAM_GRIDS: Tuple[Tuple[int, int], ...] = (
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
)

MLP_CONFIG_NAME = "mlp_medium"
MLP_NUM_FREQUENCIES = 6
MLP_INCLUDE_INPUT = True

WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 100
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(
    call, device: torch.device, warmup: int = WARMUP_ITERATIONS, timed: int = TIMED_ITERATIONS
) -> Dict[str, float]:
    """Median and IQR of the per-call wall time, in milliseconds."""
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            call()
        synchronize(device)

        samples: List[float] = []
        for _ in range(max(1, int(timed))):
            started = time.perf_counter()
            call()
            synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(samples, dtype=np.float64)
    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "median_ms": float(q50),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "num_samples": int(values.size),
    }


# ---------------------------------------------------------------------------
# Model-side helpers
# ---------------------------------------------------------------------------
def scene_shim(rx_shape: Tuple[int, int], tx_shape: Tuple[int, int], bs_position) -> SimpleNamespace:
    """The three attributes ``eval_render.render_batch`` reads off a Scene."""
    return SimpleNamespace(rx_shape=tuple(rx_shape), tx_shape=tuple(tx_shape),
                           bs_position=bs_position)


def build_mlp_shell(num_outputs: int, device: torch.device) -> PositionMLP:
    """A fresh, UNTRAINED ``mlp_medium`` with ``num_outputs`` outputs."""
    config = MLP_CONFIGS[MLP_CONFIG_NAME]
    model = PositionMLP(
        num_outputs=int(num_outputs),
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        num_frequencies=MLP_NUM_FREQUENCIES,
        include_input=MLP_INCLUDE_INPUT,
    ).to(device)
    model.eval()
    return model


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_time_vs_beams(output_dir: str, rows: Sequence[Dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6))

    for method in (METHOD_MIMOGS, METHOD_MLP):
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda r: r["num_beams"]
        )
        if not selected:
            continue
        beams = np.asarray([int(row["num_beams"]) for row in selected], dtype=np.float64)
        median = np.asarray([float(row["median_ms"]) for row in selected])
        low = np.asarray([float(row["q25_ms"]) for row in selected])
        high = np.asarray([float(row["q75_ms"]) for row in selected])

        style = METHOD_STYLE[method]
        axis.plot(
            beams, median, label=method, color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], linewidth=1.8, markersize=5.5,
        )
        axis.fill_between(beams, low, high, color=style["color"], alpha=0.22, linewidth=0)

    ticks = [int(nr * nt) for nr, nt in BEAM_GRIDS]
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])
    # The two methods sit an order of magnitude apart, so a linear y axis would
    # flatten the MLP curve into the frame edge.
    axis.set_yscale("log")
    axis.tick_params(axis="x", which="minor", bottom=False)
    style_axis(axis, "Beam pairs $N_r \\times N_t$", "Time per map [ms]")
    axis.grid(alpha=0.3, linewidth=0.5, which="both")
    axis.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    save_figure(figure, output_dir, "fig_time_vs_beams")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D4 -- MIMO-GS and MLP rendering cost vs. beam-grid size"
    )
    parser.add_argument("--mimogs_dir", type=str, default=DEFAULT_MIMOGS_DIR)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=TIMED_ITERATIONS)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    warnings: List[str] = []

    print("=" * 100)
    print("[eval_complexity] Rendering cost vs. beam-grid size")
    print("=" * 100)
    print(f"[eval_complexity] device : {device} ({device_name})")

    mimogs_path = os.path.join(arguments.mimogs_dir, "model_100.pth")
    loaded = load_mimogs(mimogs_path, device, arguments.dataset)
    gs_parameters = loaded.primitive_parameter_count()
    gain_net_parameters = int(
        sum(int(v.numel()) for v in loaded.payload["capture"][12].values())
    )

    native_rx = tuple(int(v) for v in loaded.config["rx_shape"])
    native_tx = tuple(int(v) for v in loaded.config["tx_shape"])
    derived_rx = tuple(int(v) for v in square_array_shape(loaded.beam_rows))
    derived_tx = tuple(int(v) for v in square_array_shape(loaded.beam_cols))

    print(f"[eval_complexity] checkpoint : {os.path.relpath(mimogs_path, REPO_ROOT)}")
    print(f"[eval_complexity] gaussians  : {loaded.num_gaussians:,} "
          f"| primitive+gain parameters {gs_parameters:,} (grid independent)")
    print(f"[eval_complexity] native grid: {loaded.beam_rows} x {loaded.beam_cols} "
          f"-> rx_shape {native_rx}, tx_shape {native_tx}")
    print(f"[eval_complexity] square_array_shape reproduces it: "
          f"rx {derived_rx}, tx {derived_tx} -> "
          f"{'OK' if (derived_rx == native_rx and derived_tx == native_tx) else 'MISMATCH'}")
    if derived_rx != native_rx or derived_tx != native_tx:
        warnings.append(
            f"WARN square_array_shape gives rx {derived_rx} / tx {derived_tx} but the "
            f"checkpoint recorded rx {native_rx} / tx {native_tx}; the synthetic "
            f"lattices are not built the way this run's Scene built them."
        )
    print(f"[eval_complexity] timing     : batch {BATCH_SIZE}, warmup {arguments.warmup}, "
          f"{arguments.iterations} timed renders, CUDA-synchronised per call")
    print("")

    max_rx = int(getattr(loaded.model_params, "max_active_rx_beams", 8))
    max_tx = int(getattr(loaded.model_params, "max_active_tx_beams", 8))

    rx_pos = loaded.scene.test_set.positions[:BATCH_SIZE].reshape(BATCH_SIZE, 3).to(device)
    tx_pos = torch.as_tensor(loaded.scene.bs_position, dtype=torch.float32, device=device)
    mlp_input = rx_pos.clone()

    rows: List[Dict[str, object]] = []

    header = (
        f"  {'Nr x Nt':>10}{'beams':>8}{'rx_shape':>11}{'tx_shape':>11}"
        f"{'k_rx':>6}{'k_tx':>6}"
        f"{'GS med [ms]':>13}{'GS IQR':>10}{'MLP med [ms]':>14}{'MLP IQR':>10}"
        f"{'MLP params':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for beam_rows, beam_cols in BEAM_GRIDS:
        rx_shape = tuple(int(v) for v in square_array_shape(beam_rows))
        tx_shape = tuple(int(v) for v in square_array_shape(beam_cols))
        shim = scene_shim(rx_shape, tx_shape, loaded.scene.bs_position)

        k_rx = min(max_rx, beam_rows)
        k_tx = min(max_tx, beam_cols)

        def render_once(shim=shim):
            return ER.render_batch(
                rx_pos, tx_pos, loaded.gaussians, shim, loaded.model_params,
                loaded.use_cuda_rasterizer,
            )

        # Shape check before timing: a silently wrong grid would make the
        # timings meaningless.
        with torch.no_grad():
            probe = render_once()
        expected = (BATCH_SIZE, beam_rows, beam_cols)
        if tuple(probe.shape) != expected:
            raise AssertionError(
                f"[eval_complexity] synthetic grid {beam_rows}x{beam_cols} rendered "
                f"{tuple(probe.shape)}, expected {expected}."
            )
        if not bool(torch.isfinite(probe).all()) or float(probe.min()) < 0.0:
            raise AssertionError(
                f"[eval_complexity] synthetic grid {beam_rows}x{beam_cols} produced "
                f"non-finite or negative output."
            )
        del probe

        gs_timing = time_callable(render_once, device, arguments.warmup, arguments.iterations)

        mlp_model = build_mlp_shell(beam_rows * beam_cols, device)
        mlp_parameters = count_parameters(mlp_model)

        def forward_once(model=mlp_model):
            return model(mlp_input)

        mlp_timing = time_callable(forward_once, device, arguments.warmup, arguments.iterations)
        del mlp_model

        for method, timing, parameters in (
            (METHOD_MIMOGS, gs_timing, gs_parameters),
            (METHOD_MLP, mlp_timing, mlp_parameters),
        ):
            row: Dict[str, object] = {
                "method": method,
                "Nr": int(beam_rows),
                "Nt": int(beam_cols),
                "num_beams": int(beam_rows * beam_cols),
                "rx_shape": f"{rx_shape[0]}x{rx_shape[1]}",
                "tx_shape": f"{tx_shape[0]}x{tx_shape[1]}",
                "k_rx": int(k_rx) if method == METHOD_MIMOGS else "",
                "k_tx": int(k_tx) if method == METHOD_MIMOGS else "",
                "num_parameters": int(parameters),
                "parameters_grid_dependent": int(method == METHOD_MLP),
                "native_grid": int(beam_rows == loaded.beam_rows and beam_cols == loaded.beam_cols),
                "batch_size": BATCH_SIZE,
                "warmup": int(arguments.warmup),
                "device": str(device),
                "device_name": device_name,
            }
            row.update(timing)
            rows.append(row)

        grid_label = "{0}x{1}".format(beam_rows, beam_cols)
        rx_label = "{0}x{1}".format(rx_shape[0], rx_shape[1])
        tx_label = "{0}x{1}".format(tx_shape[0], tx_shape[1])
        print(
            f"  {grid_label:>10}{beam_rows * beam_cols:>8}"
            f"{rx_label:>11}{tx_label:>11}"
            f"{k_rx:>6}{k_tx:>6}"
            f"{gs_timing['median_ms']:>13.3f}{gs_timing['iqr_ms']:>10.3f}"
            f"{mlp_timing['median_ms']:>14.3f}{mlp_timing['iqr_ms']:>10.3f}"
            f"{mlp_parameters:>13,}"
        )

    print("  " + "-" * (len(header) - 2))
    primitive_scalars = (gs_parameters - gain_net_parameters) / max(loaded.num_gaussians, 1)
    print(f"  MIMO-GS parameters (grid independent): {gs_parameters:,} "
          f"= {loaded.num_gaussians:,} primitives x {primitive_scalars:.0f} scalars "
          f"+ {gain_net_parameters:,} gain-MLP weights")

    # -- outputs ----------------------------------------------------------
    output_dir = os.path.join(arguments.analysis_root, "eval_complexity")
    os.makedirs(output_dir, exist_ok=True)

    plot_time_vs_beams(output_dir, rows)

    csv_columns = (
        "method", "Nr", "Nt", "num_beams", "rx_shape", "tx_shape", "k_rx", "k_tx",
        "median_ms", "q25_ms", "q75_ms", "iqr_ms", "mean_ms", "min_ms", "max_ms",
        "num_samples", "batch_size", "warmup", "num_parameters",
        "parameters_grid_dependent", "native_grid", "device", "device_name",
    )
    write_csv(
        os.path.join(output_dir, "complexity.csv"),
        csv_columns,
        [
            [
                f"{row[column]:.6f}" if isinstance(row.get(column), float) else row.get(column, "")
                for column in csv_columns
            ]
            for row in rows
        ],
    )

    native_gs = next(
        row for row in rows
        if row["method"] == METHOD_MIMOGS and int(row["native_grid"]) == 1
    )
    native_mlp = next(
        row for row in rows if row["method"] == METHOD_MLP and int(row["native_grid"]) == 1
    )

    readme = [
        "eval_complexity -- rendering cost vs. beam-grid size",
        "=" * 70,
        "",
        "CONVENTIONS",
        f"  Device      : {device} ({device_name})",
        f"  Timing      : batch {BATCH_SIZE} (one map per call), {arguments.warmup} warmup",
        f"                calls, then {arguments.iterations} timed calls with",
        "                torch.cuda.synchronize around every timer.  Reported as the",
        "                median with the inter-quartile range (25th-75th percentile).",
        f"  MIMO-GS     : {os.path.relpath(mimogs_path, REPO_ROOT)}, "
        f"{loaded.num_gaussians:,} primitives,",
        f"                loaded once and reused for every grid.  k_rx / k_tx are the",
        f"                retained beams per primitive, min(max_active, N) with",
        f"                max_active_rx_beams={max_rx}, max_active_tx_beams={max_tx}.",
        f"  MLP         : {MLP_CONFIG_NAME} (hidden {MLP_CONFIGS[MLP_CONFIG_NAME]['hidden']},",
        f"                depth {MLP_CONFIGS[MLP_CONFIG_NAME]['depth']}, "
        f"{MLP_NUM_FREQUENCIES} Fourier frequencies, include_input="
        f"{MLP_INCLUDE_INPUT}),",
        "                instantiated fresh at num_outputs = Nr*Nt for every grid.",
        "  Beam grids  : the renderer builds beam centers from the Rx/Tx UPA shapes",
        "                (fast_renderer._build_beam_uv_grid), and Scene derives those",
        "                from the beam counts with scene.square_array_shape.  The",
        "                non-native grids are therefore synthesized by handing the",
        "                renderer square_array_shape(Nr) / square_array_shape(Nt); no",
        "                retraining and no dataset are involved.  Output shapes are",
        "                asserted per grid before timing.",
        "  Figures     : no titles; axis labels 14 pt, ticks 12 pt, legend 10 pt;",
        "                PNG at 300 dpi plus PDF.  Shaded bands are the IQR.",
        "",
        "IMPORTANT -- WHAT IS AND IS NOT MEASURED",
        "  Only COST is measured.  Accuracy at any grid other than",
        f"  {loaded.beam_rows}x{loaded.beam_cols} is UNDEFINED for both methods:",
        "    * the MLP entries at non-native sizes are UNTRAINED SHELLS, timed purely",
        "      for their forward cost -- they predict nothing meaningful;",
        f"    * the MIMO-GS primitives were fitted against the {loaded.beam_rows}x"
        f"{loaded.beam_cols} codebook, so",
        "      rendering them onto a different lattice is a cost probe, not a",
        "      prediction.",
        "",
        "PARAMETER COUNTS",
        f"  MIMO-GS : {gs_parameters:,}  (primitive tensors + gain MLP, counted from the",
        "            checkpoint's capture; GRID INDEPENDENT -- the primitives carry no",
        "            per-beam state, so the same number applies to every row).",
        f"            = {loaded.num_gaussians:,} primitives x {primitive_scalars:.0f} scalars",
        f"              (xyz, scaling, rotation, opacity, and the Tx-side xyz / scaling /",
        f"              rotation) + {gain_net_parameters:,} gain-MLP weights.",
        "  MLP     : grid dependent, Nr*Nt output units:",
    ]
    for row in rows:
        if row["method"] != METHOD_MLP:
            continue
        readme.append(
            f"            {int(row['Nr']):>3} x {int(row['Nt']):>3} "
            f"({int(row['num_beams']):>5} beams) : {int(row['num_parameters']):>10,}"
        )

    readme += [
        "",
        "MEASURED TIME PER MAP [ms]",
        f"  {'Nr x Nt':>10}{'beams':>8}{'MIMO-GS median':>16}{'MIMO-GS IQR':>13}"
        f"{'MLP median':>13}{'MLP IQR':>10}",
    ]
    by_grid: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_grid.setdefault(int(row["num_beams"]), {})[str(row["method"])] = row
    for beams in sorted(by_grid):
        gs = by_grid[beams][METHOD_MIMOGS]
        mlp = by_grid[beams][METHOD_MLP]
        label = "{0}x{1}".format(int(gs["Nr"]), int(gs["Nt"]))
        readme.append(
            f"  {label:>10}{beams:>8}"
            f"{float(gs['median_ms']):>16.3f}{float(gs['iqr_ms']):>13.3f}"
            f"{float(mlp['median_ms']):>13.3f}{float(mlp['iqr_ms']):>10.3f}"
        )

    readme += [
        "",
        "HEADLINE NUMBERS (native grid "
        f"{loaded.beam_rows}x{loaded.beam_cols})",
        f"  MIMO-GS : {float(native_gs['median_ms']):.3f} ms/map "
        f"(IQR {float(native_gs['iqr_ms']):.3f} ms), {gs_parameters:,} parameters",
        f"  MLP     : {float(native_mlp['median_ms']):.3f} ms/map "
        f"(IQR {float(native_mlp['iqr_ms']):.3f} ms), "
        f"{int(native_mlp['num_parameters']):,} parameters",
        f"  ratio   : MIMO-GS is "
        f"{float(native_gs['median_ms']) / max(float(native_mlp['median_ms']), 1e-12):.1f}x "
        f"the MLP's per-map time at the native grid.",
        "",
        "FILES",
        "  fig_time_vs_beams.{png,pdf}  median time per map vs. Nr*Nt (log2), IQR bands",
        "  complexity.csv               every number plotted, plus parameter counts",
        "  README.txt                   this file",
        "",
        "WARNINGS",
    ]
    readme += [f"  {warning}" for warning in warnings] or ["  none"]
    readme += ["", "RERUN", "  python eval_complexity.py"]

    write_readme(os.path.join(output_dir, "README.txt"), readme)

    print("")
    print("=" * 100)
    print("[eval_complexity] SUMMARY")
    print("=" * 100)
    print(f"  {'Nr x Nt':>10}{'beams':>8}{'MIMO-GS [ms]':>15}{'MLP [ms]':>12}"
          f"{'GS/MLP':>9}{'MLP params':>13}{'GS params':>13}")
    print("  " + "-" * 80)
    for beams in sorted(by_grid):
        gs = by_grid[beams][METHOD_MIMOGS]
        mlp = by_grid[beams][METHOD_MLP]
        label = "{0}x{1}".format(int(gs["Nr"]), int(gs["Nt"]))
        print(
            f"  {label:>10}{beams:>8}"
            f"{float(gs['median_ms']):>15.3f}{float(mlp['median_ms']):>12.3f}"
            f"{float(gs['median_ms']) / max(float(mlp['median_ms']), 1e-12):>9.1f}"
            f"{int(mlp['num_parameters']):>13,}{int(gs['num_parameters']):>13,}"
        )
    print("  " + "-" * 80)
    print(f"  GPU: {device_name}   |   MIMO-GS parameter count is grid independent.")
    print("  MLP rows at non-native grids are untrained shells timed for cost only.")
    print("")
    if warnings:
        print(f"[eval_complexity] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_complexity] No warnings.")
    print(f"[eval_complexity] Outputs written to {output_dir}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
