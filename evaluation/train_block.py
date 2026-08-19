#!/usr/bin/env python3
"""Region (block) hold-out split: retrain MIMO-GS and the position-MLP once.

The shipped ``dataset/asu_campus_16by64_lt`` split interleaves train and test on
a 1 m lattice, so 99.4% of the test locations sit exactly 1.000 m from a
training location and every number measured on it is an *interpolation* number.
This script re-partitions the very same data into a REGION hold-out: the xy
plane is tiled into squares, whole tiles are held out, and a guard band around
the held-out tiles is deleted from the training set.  Both models are then
retrained once, with the UNMODIFIED trainers, on that split.

Outputs::

    outputs/block/mimogs/model_block.pth   repacked MIMO-GS checkpoint
    outputs/block/MLP/model_block.pth      repacked position-MLP checkpoint
    outputs/block/split.npz                the split itself (eval reads this)

Zero-argument runnable::

    python train_block.py

Nothing in the repository is modified.  The split is materialized into a
throwaway dataset directory under ``./.block_tmp/`` (a subsampled ``train.mat``,
a new ``test.mat`` and a byte-for-byte copy of ``bs_info.yml``), ``train.py`` and
``evaluation/train_MLP.py`` are launched as subprocesses against it, and the
results are repacked into self-contained checkpoints in exactly the dict format
``train_density.py`` / ``train_density_MLP.py`` produce.

Normalization safety
--------------------
``scene/dataloader.DeepMIMODataset`` normalizes positions by the per-file
``abs().max() + 1e-6``.  Train and test therefore each carry their own scale
factor, and the two only agree because the shipped split is interleaved
(both files max out at the same 184.449 m).  A region hold-out can easily break
that, which would silently feed the models mis-scaled coordinates at test time.
The split is therefore redrawn with the next seed until the block split's
train/test ``max|coord|`` gap is no larger than the ORIGINAL split's gap, which
is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch

from train_density import (
    DATASET_EPS,
    REPO_ROOT,
    convergence_check,
    dataset_scale_factor,
    default_dataset_dir,
    existing_repack_is_loadable,
    find_run_dir,
    load_train_mat,
    parse_loss_trajectory,
    print_failure_tail,
    repack_mimogs_checkpoint,
    run_and_tee,
    snapshot_outputs,
)
from train_density_MLP import (
    CONFIG_NAME,
    TRAIN_MLP_MODULE,
    repack_mlp_checkpoint,
    verify_train_mlp_cli,
)


SEED = 0
EPOCHS = 50

DEFAULT_TILE_SIZE = 20.0
DEFAULT_GUARD = 3.0
DEFAULT_SPLIT_SEED = 0

# A tile has to hold this many locations before it may be held out; anything
# smaller is a sliver at the scene boundary and stays in TRAIN.
MIN_TILE_LOCATIONS = 20

# Target share of the pool that ends up in the block TEST set.
TEST_FRACTION_LOW = 0.20
TEST_FRACTION_HIGH = 0.25

# How many consecutive seeds the normalization guard may burn through.
MAX_REDRAWS = 32

# Slack on the max|coord| comparison: positions are float32 in the .mat files,
# so an exact tie still has to survive a float64 round trip.
SCALE_TOLERANCE_M = 1e-6

TEMP_ROOT = os.path.join(REPO_ROOT, ".block_tmp")
BLOCK_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "block")
MIMOGS_OUTPUT_DIR = os.path.join(BLOCK_OUTPUT_ROOT, "mimogs")
MLP_OUTPUT_DIR = os.path.join(BLOCK_OUTPUT_ROOT, "MLP")
MLP_RUN_ROOT = os.path.join(MLP_OUTPUT_DIR, "_runs")
SPLIT_NPZ = os.path.join(BLOCK_OUTPUT_ROOT, "split.npz")

CHECKPOINT_NAME = "model_block.pth"


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------
def load_test_mat(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``test.mat`` and return ``(positions, magnitude)``.

    ``train_density.load_train_mat`` is the train-side twin; the validation is
    the same, so the messages are kept parallel.
    """
    test_mat_path = os.path.join(dataset_dir, "test.mat")

    if not os.path.isfile(test_mat_path):
        raise SystemExit(f"[block] test.mat is missing: {test_mat_path}")

    contents = sio.loadmat(test_mat_path)
    positions = np.asarray(contents["positions"])
    magnitude = np.asarray(contents["magnitude"])

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SystemExit(f"[block] positions must be (N,3); got {positions.shape}")

    if magnitude.ndim != 3 or magnitude.shape[0] != positions.shape[0]:
        raise SystemExit(
            f"[block] magnitude must be (N,Nr,Nt) matching positions; got "
            f"{magnitude.shape} vs {positions.shape}"
        )

    return positions, magnitude


def build_pool(dataset_dir: str) -> Dict[str, np.ndarray]:
    """Merge ``train.mat`` and ``test.mat`` into one deduplicated location pool.

    ``pool_source`` is 0 for rows that came from ``train.mat`` and 1 for rows
    that came from ``test.mat``; together with ``pool_source_row`` it lets
    ``eval_block.py`` rebuild the magnitudes from the pristine dataset without
    ever storing them in ``split.npz``.
    """
    train_positions, train_magnitude = load_train_mat(dataset_dir)
    test_positions, test_magnitude = load_test_mat(dataset_dir)

    if train_magnitude.shape[1:] != test_magnitude.shape[1:]:
        raise SystemExit(
            f"[block] beam grids disagree: train {train_magnitude.shape[1:]} vs "
            f"test {test_magnitude.shape[1:]}"
        )

    positions = np.concatenate(
        [np.asarray(train_positions), np.asarray(test_positions)], axis=0
    )
    magnitude = np.concatenate(
        [np.asarray(train_magnitude), np.asarray(test_magnitude)], axis=0
    )
    source = np.concatenate(
        [
            np.zeros(train_positions.shape[0], dtype=np.int64),
            np.ones(test_positions.shape[0], dtype=np.int64),
        ]
    )
    source_row = np.concatenate(
        [
            np.arange(train_positions.shape[0], dtype=np.int64),
            np.arange(test_positions.shape[0], dtype=np.int64),
        ]
    )

    # Exact duplicate positions would put the same location on both sides of
    # the hold-out.  The first occurrence (train.mat's, by construction) wins.
    _, first_occurrence = np.unique(
        np.round(positions.astype(np.float64), 6), axis=0, return_index=True
    )
    keep = np.sort(first_occurrence.astype(np.int64))
    num_duplicates = int(positions.shape[0] - keep.size)

    return {
        "positions": np.ascontiguousarray(positions[keep]),
        "magnitude": np.ascontiguousarray(magnitude[keep]),
        "source": np.ascontiguousarray(source[keep]),
        "source_row": np.ascontiguousarray(source_row[keep]),
        "num_duplicates_dropped": num_duplicates,
        "n_original_train": int(train_positions.shape[0]),
        "n_original_test": int(test_positions.shape[0]),
        "original_train_positions": np.asarray(train_positions),
        "original_test_positions": np.asarray(test_positions),
    }


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def assign_tiles(positions: np.ndarray, tile_size: float) -> Dict[str, np.ndarray]:
    """Tile the xy plane into ``tile_size`` squares anchored at the min corner."""
    if float(tile_size) <= 0.0:
        raise SystemExit(f"[block] --tile_size must be positive; got {tile_size}")

    coordinates = np.asarray(positions, dtype=np.float64)
    origin = coordinates[:, :2].min(axis=0)

    tile_x = np.floor((coordinates[:, 0] - origin[0]) / float(tile_size)).astype(np.int64)
    tile_y = np.floor((coordinates[:, 1] - origin[1]) / float(tile_size)).astype(np.int64)

    num_x = int(tile_x.max()) + 1
    tile_id = tile_y * num_x + tile_x

    return {
        "origin": origin,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_id": tile_id,
        "num_x": num_x,
        "num_y": int(tile_y.max()) + 1,
    }


def tile_counts(tile_id: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(tile_id, return_counts=True)
    return {int(key): int(value) for key, value in zip(unique, counts)}


def extreme_pool_index(positions: np.ndarray) -> int:
    """Index of the location carrying the pool's largest ``max(|x|,|y|,|z|)``.

    ``scene/dataloader.py`` normalizes by exactly this quantity, so the tile
    holding it is pinned to TRAIN and the trained model's coordinate scale
    equals the pool's.
    """
    return int(np.argmax(np.abs(np.asarray(positions, dtype=np.float64)).max(axis=1)))


def choose_test_tiles(
    tile_id: np.ndarray,
    counts: Dict[int, int],
    forced_train_tile: int,
    rng: np.random.RandomState,
    min_tile_locations: int = MIN_TILE_LOCATIONS,
    low: float = TEST_FRACTION_LOW,
    high: float = TEST_FRACTION_HIGH,
) -> Tuple[Optional[np.ndarray], int]:
    """Draw whole tiles until the test share lands inside ``[low, high]``."""
    total = int(tile_id.shape[0])
    low_count = int(np.ceil(float(low) * total))
    high_count = int(np.floor(float(high) * total))

    eligible = sorted(
        key
        for key, value in counts.items()
        if value >= int(min_tile_locations) and key != int(forced_train_tile)
    )

    chosen: List[int] = []
    running = 0

    for position in rng.permutation(len(eligible)):
        if running >= low_count:
            break
        candidate = eligible[int(position)]
        if running + counts[candidate] <= high_count:
            chosen.append(candidate)
            running += counts[candidate]

    if low_count <= running <= high_count:
        return np.asarray(sorted(chosen), dtype=np.int64), running

    return None, running


# ---------------------------------------------------------------------------
# Guard band and leakage diagnostics
# ---------------------------------------------------------------------------
def pairwise_nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from every ``query`` row to its nearest ``reference`` row [m].

    3D Euclidean in ORIGINAL meters, never the per-file normalized coordinates.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if reference.shape[0] == 0 or query.shape[0] == 0:
        return np.full(query.shape[0], np.inf, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 - optional fast path

        distances, _ = cKDTree(reference).query(query, k=1)
        return np.asarray(distances, dtype=np.float64).reshape(-1)
    except ImportError:
        pass

    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = 512
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        deltas = query[start:stop, None, :] - reference[None, :, :]
        out[start:stop] = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min(axis=1)
    return out


def apply_guard_band(
    positions: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, guard: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop every TRAIN location strictly closer than ``guard`` to any test one.

    Returns ``(kept_train_indices, dropped_indices)``.  Dropped locations leave
    the experiment entirely -- they are neither trained on nor tested on.
    """
    if float(guard) <= 0.0:
        return np.asarray(train_indices, dtype=np.int64), np.empty(0, dtype=np.int64)

    distances = pairwise_nearest(positions[train_indices], positions[test_indices])
    too_close = distances < float(guard)

    return (
        np.asarray(train_indices[~too_close], dtype=np.int64),
        np.asarray(train_indices[too_close], dtype=np.int64),
    )


def distance_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10.0)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Split construction (with the normalization redraw loop)
# ---------------------------------------------------------------------------
def build_block_split(
    pool: Dict[str, np.ndarray],
    tile_size: float,
    guard: float,
    split_seed: int,
    original_scale_gap: float,
    max_redraws: int = MAX_REDRAWS,
    min_tile_locations: int = MIN_TILE_LOCATIONS,
) -> Dict[str, object]:
    """Draw a region hold-out that keeps the two files' coordinate scales aligned."""
    positions = np.asarray(pool["positions"], dtype=np.float64)
    tiling = assign_tiles(positions, tile_size)
    tile_id = tiling["tile_id"]
    counts = tile_counts(tile_id)

    extreme_index = extreme_pool_index(positions)
    forced_train_tile = int(tile_id[extreme_index])

    attempts: List[Dict[str, object]] = []
    accepted: Optional[Dict[str, object]] = None

    for offset in range(int(max_redraws)):
        seed = int(split_seed) + offset
        rng = np.random.RandomState(seed)

        test_tiles, test_count = choose_test_tiles(
            tile_id, counts, forced_train_tile, rng, min_tile_locations
        )

        if test_tiles is None:
            attempts.append(
                {
                    "seed": seed,
                    "status": "test share outside [20%, 25%]",
                    "n_test": int(test_count),
                    "scale_gap_m": None,
                }
            )
            continue

        test_mask = np.isin(tile_id, test_tiles)
        test_indices = np.nonzero(test_mask)[0].astype(np.int64)
        train_candidates = np.nonzero(~test_mask)[0].astype(np.int64)

        train_indices, dropped_indices = apply_guard_band(
            positions, train_candidates, test_indices, guard
        )

        if train_indices.size == 0:
            attempts.append(
                {
                    "seed": seed,
                    "status": "guard band emptied the train set",
                    "n_test": int(test_indices.size),
                    "scale_gap_m": None,
                }
            )
            continue

        train_scale = dataset_scale_factor(positions[train_indices])
        test_scale = dataset_scale_factor(positions[test_indices])
        scale_gap = abs(train_scale - test_scale)

        candidate = {
            "seed": seed,
            "test_tiles": test_tiles,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "dropped_indices": dropped_indices,
            "train_scale": train_scale,
            "test_scale": test_scale,
            "scale_gap_m": scale_gap,
        }

        ok = scale_gap <= float(original_scale_gap) + SCALE_TOLERANCE_M
        attempts.append(
            {
                "seed": seed,
                "status": "accepted" if ok else "max|coord| gap exceeds the original split",
                "n_test": int(test_indices.size),
                "scale_gap_m": float(scale_gap),
            }
        )

        if ok:
            accepted = candidate
            break

        if accepted is None or scale_gap < float(accepted["scale_gap_m"]):
            accepted = candidate  # keep the best fallback seen so far

    if accepted is None:
        raise SystemExit(
            f"[block] {max_redraws}개의 시드를 모두 시도했지만 20-25% 크기의 "
            f"블록 테스트 집합을 만들지 못했습니다. --tile_size 를 줄여 보세요."
        )

    accepted_gap = float(accepted["scale_gap_m"])
    scale_ok = accepted_gap <= float(original_scale_gap) + SCALE_TOLERANCE_M

    train_indices = accepted["train_indices"]
    test_indices = accepted["test_indices"]

    test_to_train = pairwise_nearest(positions[test_indices], positions[train_indices])
    train_to_test = pairwise_nearest(positions[train_indices], positions[test_indices])

    return {
        "tiling": tiling,
        "tile_id": tile_id,
        "tile_counts": counts,
        "forced_train_tile": forced_train_tile,
        "extreme_index": extreme_index,
        "test_tiles": accepted["test_tiles"],
        "train_indices": train_indices,
        "test_indices": test_indices,
        "dropped_indices": accepted["dropped_indices"],
        "effective_seed": int(accepted["seed"]),
        "requested_seed": int(split_seed),
        "train_scale_factor": float(accepted["train_scale"]),
        "test_scale_factor": float(accepted["test_scale"]),
        "scale_gap_m": accepted_gap,
        "original_scale_gap_m": float(original_scale_gap),
        "scale_ok": bool(scale_ok),
        "attempts": attempts,
        "test_to_train": test_to_train,
        "train_to_test": train_to_test,
        "tile_size": float(tile_size),
        "guard": float(guard),
        "min_tile_locations": int(min_tile_locations),
    }


def assert_no_leakage(split: Dict[str, object]) -> None:
    """Hard-fail when the guard band did not actually separate the two sets."""
    guard = float(split["guard"])
    minimum = float(np.asarray(split["train_to_test"], dtype=np.float64).min())

    if minimum + 1e-9 < guard:
        raise AssertionError(
            f"[block] LEAKAGE: the closest train-to-test distance is {minimum:.6f} m, "
            f"below the {guard:.3f} m guard band."
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def ascii_overview(split: Dict[str, object]) -> List[str]:
    """A tiny north-up map: ``T`` test, ``#`` train, ``X`` the pinned tile."""
    tiling = split["tiling"]
    counts = split["tile_counts"]
    test_tiles = set(int(value) for value in np.asarray(split["test_tiles"]).ravel())
    forced = int(split["forced_train_tile"])
    num_x = int(tiling["num_x"])
    num_y = int(tiling["num_y"])

    lines = [
        f"  tile grid {num_x} x {num_y} of {split['tile_size']:.1f} m squares, "
        f"origin ({tiling['origin'][0]:.3f}, {tiling['origin'][1]:.3f}) m, y up",
        "  legend: T = held-out test tile, # = train tile, "
        "x = train tile below the size threshold, X = scale-pinned train tile, . = empty",
        "",
    ]

    for row in range(num_y - 1, -1, -1):
        cells = []
        for column in range(num_x):
            key = row * num_x + column
            count = counts.get(key, 0)
            if count == 0:
                cells.append(".")
            elif key == forced:
                cells.append("X")
            elif key in test_tiles:
                cells.append("T")
            elif count < int(split["min_tile_locations"]):
                cells.append("x")
            else:
                cells.append("#")
        lines.append(f"    y={row:<2d} | " + " ".join(cells))

    lines.append("           " + "  ".join(f"{column:<1d}" for column in range(num_x)))
    lines.append("            x tile index ->")
    return lines


def print_split_report(split: Dict[str, object], pool: Dict[str, np.ndarray]) -> None:
    positions = np.asarray(pool["positions"], dtype=np.float64)
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]
    dropped_indices = split["dropped_indices"]
    total = int(positions.shape[0])
    counts = split["tile_counts"]

    print("-" * 100)
    print("[block] SPLIT")
    print("-" * 100)
    print(f"  pool locations      : {total:,} "
          f"(train.mat {pool['n_original_train']:,} + test.mat {pool['n_original_test']:,}"
          f", {pool['num_duplicates_dropped']} exact duplicate position(s) dropped)")
    print(f"  tile size / guard   : {split['tile_size']:.1f} m / {split['guard']:.1f} m")
    print(f"  eligible tile size  : >= {split['min_tile_locations']} locations")
    print(f"  requested seed      : {split['requested_seed']}")
    print(f"  effective seed      : {split['effective_seed']}"
          + ("" if split["effective_seed"] == split["requested_seed"]
             else "  (redrawn by the normalization guard)"))
    print("")
    print(f"  held-out test tiles : {len(split['test_tiles'])} of "
          f"{len(counts)} occupied tiles")

    tiling = split["tiling"]
    num_x = int(tiling["num_x"])
    print(f"  {'tile id':>8}{'(tx,ty)':>12}{'locations':>12}")
    for key in np.asarray(split["test_tiles"]).ravel().tolist():
        key = int(key)
        print(f"  {key:>8}{f'({key % num_x},{key // num_x})':>12}{counts.get(key, 0):>12}")

    print("")
    for line in ascii_overview(split):
        print(line)

    print("")
    print(f"  n_train             : {train_indices.size:,} "
          f"({100.0 * train_indices.size / total:.2f}% of the pool)")
    print(f"  n_test              : {test_indices.size:,} "
          f"({100.0 * test_indices.size / total:.2f}% of the pool)")
    print(f"  n_dropped (guard)   : {dropped_indices.size:,} "
          f"({100.0 * dropped_indices.size / total:.2f}% of the pool, "
          f"removed from TRAIN entirely)")

    print("")
    print("  NORMALIZATION SAFETY (scene/dataloader.py divides by max|coord| + 1e-6)")
    print(f"    original split train / test max|coord| gap : "
          f"{split['original_scale_gap_m']:.9f} m")
    print(f"    block split    train scale factor         : "
          f"{split['train_scale_factor']:.9f}")
    print(f"    block split    test  scale factor         : "
          f"{split['test_scale_factor']:.9f}")
    print(f"    block split    gap                        : "
          f"{split['scale_gap_m']:.9f} m -> "
          f"{'OK' if split['scale_ok'] else 'EXCEEDS THE ORIGINAL GAP'}")
    print("    seed ladder:")
    for attempt in split["attempts"]:
        gap = attempt["scale_gap_m"]
        gap_text = "-" if gap is None else f"{float(gap):.9f} m"
        print(f"      seed {int(attempt['seed']):>4}  n_test={int(attempt['n_test']):>6}  "
              f"gap={gap_text:>16}  {attempt['status']}")

    print("")
    print("  EXTRAPOLATION DIAGNOSTIC (the headline of this experiment)")
    test_stats = distance_stats(split["test_to_train"])
    print("    distance from each block TEST location to its nearest block TRAIN location [m]")
    print(f"      min {test_stats['min']:.3f} | p10 {test_stats['p10']:.3f} | "
          f"median {test_stats['median']:.3f} | mean {test_stats['mean']:.3f} | "
          f"p90 {test_stats['p90']:.3f} | max {test_stats['max']:.3f}")
    train_stats = distance_stats(split["train_to_test"])
    print(f"    minimum train-to-test distance = {train_stats['min']:.6f} m "
          f"(guard {split['guard']:.3f} m) -> leakage assert PASSED")
    print("-" * 100)


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------
def materialize_block_dataset(
    dataset_dir: str,
    destination_dir: str,
    positions: np.ndarray,
    magnitude: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    expected_train_scale: float,
    expected_test_scale: float,
) -> Dict[str, object]:
    """Write a throwaway dataset dir holding the block train.mat and test.mat.

    ``bs_info.yml`` is copied byte-for-byte; the pristine dataset is never
    touched.  Both written files are read back so the assertions cover what the
    trainers will actually see.
    """
    if os.path.isdir(destination_dir):
        shutil.rmtree(destination_dir)

    os.makedirs(destination_dir, exist_ok=True)

    for name, indices in (("train.mat", train_indices), ("test.mat", test_indices)):
        sio.savemat(
            os.path.join(destination_dir, name),
            {
                "positions": np.ascontiguousarray(positions[indices]),
                "magnitude": np.ascontiguousarray(magnitude[indices]),
            },
            do_compression=False,
        )

    source = os.path.join(dataset_dir, "bs_info.yml")
    if not os.path.isfile(source):
        raise SystemExit(f"[block] required dataset file is missing: {source}")
    shutil.copy2(source, os.path.join(destination_dir, "bs_info.yml"))

    written: Dict[str, object] = {"path": destination_dir}

    for name, indices, expected in (
        ("train.mat", train_indices, expected_train_scale),
        ("test.mat", test_indices, expected_test_scale),
    ):
        contents = sio.loadmat(os.path.join(destination_dir, name))
        written_positions = np.asarray(contents["positions"])

        if written_positions.shape[0] != int(indices.shape[0]):
            raise AssertionError(f"[block] written {name} has the wrong sample count")

        actual = dataset_scale_factor(written_positions)
        if abs(actual - float(expected)) > SCALE_TOLERANCE_M:
            raise AssertionError(
                f"[block] normalization scale drifted in {name}: "
                f"{actual!r} != {float(expected)!r}"
            )

        written[name] = {
            "n": int(indices.shape[0]),
            "scale_factor": float(actual),
            "max_abs_position": float(np.abs(written_positions).max()),
        }

    return written


def save_split_npz(
    path: str, pool: Dict[str, np.ndarray], split: Dict[str, object], dataset_dir: str
) -> str:
    """Persist everything ``eval_block.py`` needs to rebuild the split."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tiling = split["tiling"]

    np.savez_compressed(
        path,
        pool_positions=np.asarray(pool["positions"], dtype=np.float64),
        pool_source=np.asarray(pool["source"], dtype=np.int64),
        pool_source_row=np.asarray(pool["source_row"], dtype=np.int64),
        tile_id=np.asarray(split["tile_id"], dtype=np.int64),
        tile_x=np.asarray(tiling["tile_x"], dtype=np.int64),
        tile_y=np.asarray(tiling["tile_y"], dtype=np.int64),
        tile_origin=np.asarray(tiling["origin"], dtype=np.float64),
        tile_grid=np.asarray([tiling["num_x"], tiling["num_y"]], dtype=np.int64),
        test_tiles=np.asarray(split["test_tiles"], dtype=np.int64),
        train_indices=np.asarray(split["train_indices"], dtype=np.int64),
        test_indices=np.asarray(split["test_indices"], dtype=np.int64),
        dropped_indices=np.asarray(split["dropped_indices"], dtype=np.int64),
        forced_train_tile=np.asarray(split["forced_train_tile"], dtype=np.int64),
        extreme_index=np.asarray(split["extreme_index"], dtype=np.int64),
        test_to_train_distance_m=np.asarray(split["test_to_train"], dtype=np.float64),
        train_to_test_distance_m=np.asarray(split["train_to_test"], dtype=np.float64),
        tile_size=np.asarray(split["tile_size"], dtype=np.float64),
        guard=np.asarray(split["guard"], dtype=np.float64),
        split_seed=np.asarray(split["requested_seed"], dtype=np.int64),
        effective_seed=np.asarray(split["effective_seed"], dtype=np.int64),
        min_tile_locations=np.asarray(split["min_tile_locations"], dtype=np.int64),
        train_scale_factor=np.asarray(split["train_scale_factor"], dtype=np.float64),
        test_scale_factor=np.asarray(split["test_scale_factor"], dtype=np.float64),
        scale_gap_m=np.asarray(split["scale_gap_m"], dtype=np.float64),
        original_scale_gap_m=np.asarray(split["original_scale_gap_m"], dtype=np.float64),
        n_original_train=np.asarray(pool["n_original_train"], dtype=np.int64),
        n_original_test=np.asarray(pool["n_original_test"], dtype=np.int64),
        dataset_dir=np.asarray(os.path.abspath(dataset_dir)),
    )

    print(f"[block] 스플릿 저장 완료 -> {path}")
    return path


# ---------------------------------------------------------------------------
# Repack augmentation
# ---------------------------------------------------------------------------
def block_metadata(
    split: Dict[str, object], pool_size: int, positions: np.ndarray
) -> Dict[str, object]:
    """The block-split provenance stamped into both checkpoints."""
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]

    return {
        "tile_size": float(split["tile_size"]),
        "guard": float(split["guard"]),
        "split_seed": int(split["requested_seed"]),
        "effective_seed": int(split["effective_seed"]),
        "min_tile_locations": int(split["min_tile_locations"]),
        "test_tiles": [int(value) for value in np.asarray(split["test_tiles"]).ravel()],
        "forced_train_tile": int(split["forced_train_tile"]),
        "n_pool": int(pool_size),
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
        "n_dropped": int(np.asarray(split["dropped_indices"]).size),
        "train_scale_factor": float(split["train_scale_factor"]),
        "test_scale_factor": float(split["test_scale_factor"]),
        "scale_gap_m": float(split["scale_gap_m"]),
        "original_scale_gap_m": float(split["original_scale_gap_m"]),
        "scale_ok": bool(split["scale_ok"]),
        "test_to_train_distance_m": distance_stats(split["test_to_train"]),
        "train_to_test_min_m": float(np.asarray(split["train_to_test"]).min()),
        "split_npz": os.path.relpath(SPLIT_NPZ, REPO_ROOT),
    }


def augment_checkpoint(
    path: str,
    split: Dict[str, object],
    metadata: Dict[str, object],
    positions: np.ndarray,
    smoke_rows: int = 3,
) -> None:
    """Add the block metadata and a self-contained smoke sample to a repack.

    ``smoke_positions_m`` plus ``test_scale_factor`` are what make the smoke
    test standalone: it never has to find a dataset directory on disk.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    sample = test_indices[: int(smoke_rows)]

    payload["split"] = "block"
    payload["block"] = dict(metadata)
    payload["smoke_positions_m"] = np.ascontiguousarray(
        np.asarray(positions, dtype=np.float64)[sample]
    )
    payload["test_scale_factor"] = float(split["test_scale_factor"])

    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Standalone smoke tests
# ---------------------------------------------------------------------------
def smoke_positions(payload: dict) -> Tuple[torch.Tensor, int]:
    """Normalized smoke-test positions, read from the checkpoint dict alone."""
    positions = torch.as_tensor(
        np.asarray(payload["smoke_positions_m"]), dtype=torch.float32
    )
    scale = float(payload.get("test_scale_factor", 0.0))

    if scale <= 0.0:
        scale = float(positions.abs().max()) + DATASET_EPS

    return positions / scale, int(positions.shape[0])


def smoke_test_block_mimogs(path: str) -> None:
    """Rebuild MIMO-GS from the dict alone and render the stored test locations."""
    from types import SimpleNamespace

    from gaussian_renderer.fast_renderer import render_fast
    from scene import GaussianModel

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]

    model_params = SimpleNamespace(**config["model_params"])
    opt_params = SimpleNamespace(**config["opt_params"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussians = GaussianModel(
        target_gaussians=int(getattr(model_params, "target_gaussians", 25_000)),
        optimizer_type=str(getattr(opt_params, "optimizer_type", "default")),
        device=str(device),
        init_range=1.0,
        tie_covariance=bool(int(getattr(model_params, "tie_covariance", 0))),
    )
    gaussians.restore(payload["capture"], opt_params)
    gaussians.dynamic_gain_net.eval()

    normalized, expected_rows = smoke_positions(payload)
    rx_pos = normalized.to(device)
    tx_pos = torch.as_tensor(config["bs_position"], dtype=torch.float32, device=device)

    with torch.inference_mode():
        rendered = render_fast(
            rx_pos=rx_pos,
            tx_pos=tx_pos,
            pc=gaussians,
            rx_shape=tuple(config["rx_shape"]),
            tx_shape=tuple(config["tx_shape"]),
            covariance_floor=1e-4,
            weight_floor=1e-4,
            max_active_rx_beams=int(getattr(model_params, "max_active_rx_beams", 8)),
            max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 8)),
            use_cuda_rasterizer=bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
            and torch.cuda.is_available(),
        )["render"]

    if rendered.ndim == 2:
        rendered = rendered.unsqueeze(0)

    check_smoke_output(
        rendered,
        expected_rows,
        (int(config["beam_rows"]), int(config["beam_cols"])),
        os.path.basename(path),
        payload,
    )


def smoke_test_block_mlp(path: str) -> None:
    """Rebuild PositionMLP from ``arch`` alone and forward the stored locations."""
    from evaluation.train_MLP import PositionMLP

    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload["arch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PositionMLP(
        num_outputs=int(arch["num_outputs"]),
        hidden=int(arch["hidden"]),
        depth=int(arch["depth"]),
        num_frequencies=int(arch["num_frequencies"]),
        include_input=bool(arch["include_input"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    beam_rows = int(payload["beam_rows"])
    beam_cols = int(payload["beam_cols"])

    if beam_rows * beam_cols != int(arch["num_outputs"]):
        raise AssertionError(
            f"[smoke] num_outputs {arch['num_outputs']} is not reshapeable to "
            f"({beam_rows}, {beam_cols})"
        )

    normalized, expected_rows = smoke_positions(payload)

    with torch.inference_mode():
        predicted = model(normalized.to(device))

    check_smoke_output(
        predicted.reshape(expected_rows, beam_rows, beam_cols),
        expected_rows,
        (beam_rows, beam_cols),
        os.path.basename(path),
        payload,
    )


def check_smoke_output(
    maps: torch.Tensor,
    expected_rows: int,
    expected_shape: Tuple[int, int],
    name: str,
    payload: dict,
) -> None:
    """Shared assertions: right count, right shape, finite and non-negative."""
    if int(maps.shape[0]) != int(expected_rows):
        raise AssertionError(
            f"[smoke] expected {expected_rows} rendered locations, got {int(maps.shape[0])}"
        )

    for index in range(int(maps.shape[0])):
        single = maps[index]

        if tuple(single.shape) != tuple(expected_shape):
            raise AssertionError(
                f"[smoke] location {index}: shape {tuple(single.shape)} != {tuple(expected_shape)}"
            )

        if not bool(torch.isfinite(single).all()):
            raise AssertionError(f"[smoke] location {index}: non-finite values in the output")

    minimum = float(maps.min())

    if minimum < 0.0:
        raise AssertionError(f"[smoke] negative entries in the output (min = {minimum:.6g})")

    block = payload.get("block", {})
    print(
        f"[smoke] OK {name} | split=block "
        f"(n_train={int(block.get('n_train', 0)):,}, n_test={int(block.get('n_test', 0)):,}) | "
        f"output {tuple(maps.shape)} | finite | non-negative | "
        f"range [{minimum:.4g}, {float(maps.max()):.4g}]"
    )


def dispatch_smoke_test(path: str) -> None:
    """Pick the right smoke test from the payload's own keys."""
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "capture" in payload:
        smoke_test_block_mimogs(path)
    elif "state_dict" in payload:
        smoke_test_block_mlp(path)
    else:
        raise SystemExit(f"[smoke] '{path}' is neither a MIMO-GS nor an MLP repack.")


def run_smoke_test_subprocess(checkpoint_path: str) -> bool:
    """Run the smoke test in a fresh interpreter so the reload is truly standalone."""
    command = [sys.executable, os.path.abspath(__file__), "--smoke_test", checkpoint_path]
    returncode, lines = run_and_tee(command, cwd=REPO_ROOT)

    if returncode != 0:
        print_failure_tail(lines)

    return returncode == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mimogs(
    temp_dir: str,
    dataset_dir: str,
    destination: str,
    split: Dict[str, object],
    metadata: Dict[str, object],
    positions: np.ndarray,
    epochs: int,
) -> Dict[str, object]:
    """Launch the UNMODIFIED train.py on the block split and repack the result."""
    row: Dict[str, object] = {
        "name": "MIMO-GS",
        "status": "실패",
        "path": None,
        "note": "",
        "final_loss": None,
        "train_seconds": None,
        "convergence_warning": False,
        "convergence_note": "",
    }

    before = snapshot_outputs()
    started = time.perf_counter()

    # ``num_epochs`` and ``seed`` are train.py's own CLI options, so the
    # protocol is expressed without touching the file.
    command = [
        sys.executable,
        os.path.join(REPO_ROOT, "train.py"),
        "--source_path",
        temp_dir,
        "--num_epochs",
        str(int(epochs)),
        "--seed",
        str(SEED),
    ]

    returncode, lines = run_and_tee(command, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started

    trajectory = parse_loss_trajectory(lines)
    row["train_seconds"] = elapsed
    row["final_loss"] = trajectory[-1] if trajectory else None

    if returncode != 0:
        print_failure_tail(lines)
        row["note"] = f"train.py 종료 코드 {returncode}"
        return row

    run_dir = find_run_dir(lines, before)

    if run_dir is None:
        row["note"] = "train.py 가 만든 실행 디렉터리를 찾지 못함"
        print(f"[block] {row['note']}")
        return row

    print(f"[block] 실행 디렉터리: {run_dir}")

    try:
        repack_mimogs_checkpoint(
            run_dir=run_dir,
            destination=destination,
            fraction=float(metadata["n_train"]) / float(metadata["n_pool"]),
            dataset_dir=dataset_dir,
            n_train=int(metadata["n_train"]),
            normalization_scale_factor=float(metadata["train_scale_factor"]),
            max_abs_position=float(np.abs(positions[split["train_indices"]]).max()),
            final_loss=row["final_loss"],
            train_seconds=elapsed,
            loss_trajectory=trajectory,
            epochs=int(epochs),
        )
        augment_checkpoint(destination, split, metadata, positions)
    except Exception as error:  # noqa: BLE001
        row["note"] = f"재패킹 실패: {error}"
        print(f"[block] {row['note']}")
        return row

    row["note"] = f"run dir: {os.path.relpath(run_dir, REPO_ROOT)}"
    warned, note = convergence_check(trajectory)
    row["convergence_warning"] = warned
    row["convergence_note"] = note
    row["status"] = "재패킹완료"
    row["path"] = destination
    return row


def train_mlp(
    temp_dir: str,
    dataset_dir: str,
    destination: str,
    split: Dict[str, object],
    metadata: Dict[str, object],
    positions: np.ndarray,
    epochs: int,
) -> Dict[str, object]:
    """Launch the UNMODIFIED evaluation/train_MLP.py (mlp_medium only) and repack."""
    row: Dict[str, object] = {
        "name": "MLP",
        "status": "실패",
        "path": None,
        "note": "",
        "final_loss": None,
        "train_seconds": None,
        "convergence_warning": False,
        "convergence_note": "",
    }

    run_root = os.path.join(MLP_RUN_ROOT, "block")
    os.makedirs(run_root, exist_ok=True)

    started = time.perf_counter()

    command = [
        sys.executable,
        "-m",
        TRAIN_MLP_MODULE,
        "--configs",
        CONFIG_NAME,
        "--epochs",
        str(int(epochs)),
        "--source_path",
        temp_dir,
        "--outputs_root",
        run_root,
    ]

    returncode, lines = run_and_tee(command, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started
    row["train_seconds"] = elapsed

    if returncode != 0:
        print_failure_tail(lines)
        row["note"] = f"train_MLP.py 종료 코드 {returncode}"
        return row

    run_dir = os.path.join(run_root, CONFIG_NAME)

    if not os.path.isfile(os.path.join(run_dir, "model.pth")):
        row["note"] = f"train_MLP.py 결과를 찾지 못함: {run_dir}/model.pth"
        print(f"[block] {row['note']}")
        return row

    print(f"[block] 실행 디렉터리: {run_dir}")

    try:
        _, trajectory = repack_mlp_checkpoint(
            run_dir=run_dir,
            destination=destination,
            fraction=float(metadata["n_train"]) / float(metadata["n_pool"]),
            dataset_dir=dataset_dir,
            n_train=int(metadata["n_train"]),
            normalization_scale_factor=float(metadata["train_scale_factor"]),
            train_seconds=elapsed,
            epochs=int(epochs),
        )
        augment_checkpoint(destination, split, metadata, positions)
    except Exception as error:  # noqa: BLE001
        row["note"] = f"재패킹 실패: {error}"
        print(f"[block] {row['note']}")
        return row

    row["final_loss"] = trajectory[-1] if trajectory else None
    row["note"] = f"run dir: {os.path.relpath(run_dir, REPO_ROOT)}"
    warned, note = convergence_check(trajectory)
    row["convergence_warning"] = warned
    row["convergence_note"] = note
    row["status"] = "재패킹완료"
    row["path"] = destination
    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_block_summary(
    rows: Sequence[Dict[str, object]], split: Dict[str, object], pool_size: int
) -> None:
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]
    dropped_indices = split["dropped_indices"]
    stats = distance_stats(split["test_to_train"])
    counts = split["tile_counts"]

    print("")
    print("=" * 100)
    print("[block] 지역(블록) 홀드아웃 재학습 요약")
    print("=" * 100)
    print(f"  타일 크기 / 가드밴드   : {split['tile_size']:.1f} m / {split['guard']:.1f} m")
    print(f"  분할 시드              : 요청 {split['requested_seed']} -> "
          f"실제 {split['effective_seed']}")
    print(f"  테스트 타일            : {len(split['test_tiles'])}개 / "
          f"전체 점유 타일 {len(counts)}개  {list(int(t) for t in np.asarray(split['test_tiles']).ravel())}")
    print(f"  n_train / n_test / n_dropped : {train_indices.size:,} / "
          f"{test_indices.size:,} / {dropped_indices.size:,} "
          f"(풀 {pool_size:,}개)")
    print(f"  테스트->학습 최근접 거리 [m] : min {stats['min']:.3f} / "
          f"median {stats['median']:.3f} / p90 {stats['p90']:.3f} / max {stats['max']:.3f}")
    print(f"  정규화 스케일 (train/test)   : {split['train_scale_factor']:.6f} / "
          f"{split['test_scale_factor']:.6f} "
          f"(gap {split['scale_gap_m']:.9f} m, 원본 gap {split['original_scale_gap_m']:.9f} m)")
    print("")

    header = f"  {'모델':<10}{'최종 loss':>16}{'소요 시간':>14}  {'상태':<12}  파일"
    print(header)
    print("  " + "-" * (len(header) + 20))

    for row in rows:
        final_loss = row.get("final_loss")
        seconds = row.get("train_seconds")
        loss_text = "-" if final_loss is None else f"{float(final_loss):.8f}"
        time_text = "-" if seconds is None else f"{float(seconds) / 60.0:.1f} 분"
        path = str(row.get("path") or "-")

        if path != "-" and os.path.isabs(path):
            if os.path.commonpath([os.path.abspath(path), REPO_ROOT]) == REPO_ROOT:
                path = os.path.relpath(path, REPO_ROOT)

        print(
            f"  {str(row['name']):<10}{loss_text:>16}{time_text:>14}  "
            f"{str(row.get('status', '')):<12}  {path}"
        )

    print("  " + "-" * (len(header) + 20))
    print(f"  스플릿 파일            : {os.path.relpath(SPLIT_NPZ, REPO_ROOT)}")

    warnings = [row for row in rows if row.get("convergence_warning")]

    if warnings:
        print("")
        print("  [수렴 경고] 아래 실행은 마지막 구간에서 아직 손실이 유의미하게 감소 중입니다.")
        for row in warnings:
            print(f"    - {row['name']} : {row.get('convergence_note', '')}")
        print("    (에폭 수는 자동으로 바꾸지 않았습니다.)")
    else:
        print("  [수렴 경고] 없음")

    if not bool(split["scale_ok"]):
        print("")
        print("  [경고] 블록 스플릿의 train/test max|coord| 차이가 원본 스플릿보다 큽니다. "
              "모든 시드를 소진하여 가장 작은 차이를 가진 시드를 사용했습니다.")

    failures = [row for row in rows if str(row.get("status", "")) != "성공"]

    if failures:
        print("")
        print("  [실패/건너뜀]")
        for row in failures:
            print(f"    - {row['name']} : {row.get('status')} / {row.get('note', '')}")

    print("=" * 100)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Region (block) hold-out split: retrain MIMO-GS and the position-MLP once",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset directory (default: the source_path default in arguments/__init__.py)",
    )
    parser.add_argument("--tile_size", type=float, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--guard", type=float, default=DEFAULT_GUARD)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max_redraws", type=int, default=MAX_REDRAWS)
    parser.add_argument(
        "--min_tile_locations",
        type=int,
        default=MIN_TILE_LOCATIONS,
        help="Tiles smaller than this stay in TRAIN and are never held out",
    )
    parser.add_argument("--output_root", type=str, default=BLOCK_OUTPUT_ROOT)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build, report and save the split, then stop before training",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep ./.block_tmp/split/ instead of deleting it after training",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even when a loadable repack already exists",
    )
    parser.add_argument(
        "--smoke_test",
        type=str,
        default="",
        help="Internal: reload one repacked checkpoint standalone and exit",
    )
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()

    if arguments.smoke_test:
        dispatch_smoke_test(arguments.smoke_test)
        return 0

    dataset_dir = os.path.abspath(arguments.dataset) if arguments.dataset else default_dataset_dir()

    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[block] 데이터셋 디렉터리가 없습니다: {dataset_dir}")

    output_root = os.path.abspath(arguments.output_root)
    mimogs_destination = os.path.join(output_root, "mimogs", CHECKPOINT_NAME)
    mlp_destination = os.path.join(output_root, "MLP", CHECKPOINT_NAME)
    split_npz_path = os.path.join(output_root, "split.npz")

    os.makedirs(os.path.dirname(mimogs_destination), exist_ok=True)
    os.makedirs(os.path.dirname(mlp_destination), exist_ok=True)

    print("=" * 100)
    print("[block] 지역(블록) 홀드아웃 스플릿 재학습")
    print("=" * 100)
    print(f"  dataset      : {dataset_dir}")
    print(f"  tile_size    : {arguments.tile_size} m")
    print(f"  guard        : {arguments.guard} m")
    print(f"  split_seed   : {arguments.split_seed}")
    print(f"  epochs       : {arguments.epochs}")
    print(f"  output_root  : {output_root}")
    print("")

    pool = build_pool(dataset_dir)
    positions = np.asarray(pool["positions"], dtype=np.float64)
    magnitude = pool["magnitude"]
    pool_size = int(positions.shape[0])

    # The original split's own train/test scale gap is the yardstick the block
    # split has to match; it is measured here, never assumed.
    original_gap = abs(
        dataset_scale_factor(pool["original_train_positions"])
        - dataset_scale_factor(pool["original_test_positions"])
    )

    split = build_block_split(
        pool=pool,
        tile_size=float(arguments.tile_size),
        guard=float(arguments.guard),
        split_seed=int(arguments.split_seed),
        original_scale_gap=float(original_gap),
        max_redraws=int(arguments.max_redraws),
        min_tile_locations=int(arguments.min_tile_locations),
    )

    assert_no_leakage(split)
    print_split_report(split, pool)

    metadata = block_metadata(split, pool_size, positions)
    save_split_npz(split_npz_path, pool, split, dataset_dir)

    if arguments.dry_run:
        print("[block] --dry_run 지정: 학습을 건너뛰고 종료합니다.")
        return 0

    cli_problem = verify_train_mlp_cli()

    if cli_problem is not None:
        print("")
        print("=" * 100)
        print("[block] 중단: evaluation/train_MLP.py 를 수정하지 않고는 요구 조건을 만족할 수 없습니다.")
        print(f"[block] 사유: {cli_problem}")
        print("=" * 100)
        return 1

    temp_dir = os.path.join(TEMP_ROOT, "split")

    plans = [
        ("MIMO-GS", mimogs_destination, ("capture", "config", "block"), train_mimogs),
        ("MLP", mlp_destination, ("state_dict", "arch", "block"), train_mlp),
    ]

    pending = [
        plan
        for plan in plans
        if arguments.overwrite or not existing_repack_is_loadable(plan[1], plan[2])
    ]

    rows: List[Dict[str, object]] = []
    had_failure = False

    if pending:
        print("")
        print(f"[block] 임시 데이터셋 생성: {temp_dir}")
        materialized = materialize_block_dataset(
            dataset_dir=dataset_dir,
            destination_dir=temp_dir,
            positions=positions,
            magnitude=magnitude,
            train_indices=split["train_indices"],
            test_indices=split["test_indices"],
            expected_train_scale=float(split["train_scale_factor"]),
            expected_test_scale=float(split["test_scale_factor"]),
        )
        print(f"[block]   train.mat n={materialized['train.mat']['n']:,} "
              f"scale={materialized['train.mat']['scale_factor']:.6f}")
        print(f"[block]   test.mat  n={materialized['test.mat']['n']:,} "
              f"scale={materialized['test.mat']['scale_factor']:.6f}")
    else:
        print("")
        print("[block] 두 체크포인트가 모두 존재하여 학습을 건너뜁니다 (resume).")

    for name, destination, required_keys, trainer in plans:
        print("-" * 100)
        print(f"[block] {name} 시작")

        if not arguments.overwrite and existing_repack_is_loadable(destination, required_keys):
            print(f"[block] 이미 존재하여 건너뜁니다: {destination}")
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "name": name,
                    "status": "성공",
                    "path": destination,
                    "note": "기존 결과 재사용 (resume)",
                    "final_loss": existing.get("final_loss"),
                    "train_seconds": existing.get("train_seconds"),
                    "convergence_warning": False,
                    "convergence_note": "기존 결과 재사용",
                }
            )
            continue

        row = trainer(
            temp_dir=temp_dir,
            dataset_dir=dataset_dir,
            destination=destination,
            split=split,
            metadata=metadata,
            positions=positions,
            epochs=int(arguments.epochs),
        )

        if row["status"] != "재패킹완료":
            had_failure = True
            rows.append(row)
            continue

        if run_smoke_test_subprocess(destination):
            row["status"] = "성공"
        else:
            row["status"] = "스모크실패"
            row["note"] = (str(row["note"]) + " / " if row["note"] else "") + \
                "독립 재로드 스모크 테스트 실패"
            had_failure = True

        rows.append(row)

    if pending and not arguments.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[block] 임시 디렉터리 삭제: {temp_dir}")
    elif pending:
        print(f"[block] --keep_temp 지정: {temp_dir} 유지")

    if not arguments.keep_temp and os.path.isdir(TEMP_ROOT) and not os.listdir(TEMP_ROOT):
        os.rmdir(TEMP_ROOT)

    print_block_summary(rows, split, pool_size)

    print("")
    print("[block] 다음 단계: python eval_block.py")

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
