#!/usr/bin/env python3
"""MIMO-GS density sweep trainer.

Trains the *unmodified* ``train.py`` on nested random subsets of the training
set and repacks every result into a single self-contained checkpoint that can
be reloaded without the original run directory::

    outputs/density/mimogs/model_{6,12,25,50,100}.pth

Nothing in the repository is modified.  Each fraction gets a throwaway dataset
directory under ``./.density_tmp/`` holding a subsampled ``train.mat`` plus
verbatim copies of the original ``test.mat`` and ``bs_info.yml``; ``train.py``
is then launched as a subprocess with ``--source_path`` pointing at it.

The 100% point is *not* retrained -- it is repacked from the existing
``outputs/20260811_062015`` run, which was already a 50-epoch full-data run.

Zero-argument runnable::

    python train_density.py

``./.density_tmp/`` is scratch space and should not be committed.  ``.gitignore``
is an existing file and is deliberately left untouched, so add the entry by hand
(or rely on the directory being removed automatically after every fraction).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

SEED = 0
EPOCHS = 50
FRACTIONS: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
FRACTION_TAG: Dict[float, str] = {
    0.0625: "6",
    0.125: "12",
    0.25: "25",
    0.5: "50",
    1.0: "100",
}

TEMP_ROOT = os.path.join(REPO_ROOT, ".density_tmp")
MIMOGS_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "density", "mimogs")

# The 100% MIMO-GS point already exists: a 50-epoch full-data run.
REFERENCE_RUN_DIR = os.path.join(REPO_ROOT, "outputs", "20260811_062015")

# Relative loss decrease over the tail of a run above which we warn about a
# possibly unconverged 50-epoch schedule.
CONVERGENCE_TOLERANCE = 0.05

# ``DeepMIMODataset`` normalizes positions by ``abs().max() + DATASET_EPS``.
DATASET_EPS = 1e-6

_FAILURE_TAIL_LINES = 40


# ---------------------------------------------------------------------------
# Dataset conventions (shared with train_density_MLP.py)
# ---------------------------------------------------------------------------


def default_dataset_dir() -> str:
    """Return the ``source_path`` default declared in ``arguments/__init__.py``."""
    from argparse import ArgumentParser

    from arguments import ModelParams

    parser = ArgumentParser()
    ModelParams(parser)
    namespace = parser.parse_args([])
    source_path = str(namespace.source_path)

    if not os.path.isabs(source_path):
        source_path = os.path.join(REPO_ROOT, source_path)

    return os.path.abspath(source_path)


def load_train_mat(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``train.mat`` and return ``(positions, magnitude)``."""
    train_mat_path = os.path.join(dataset_dir, "train.mat")

    if not os.path.isfile(train_mat_path):
        raise SystemExit(f"[density] train.mat is missing: {train_mat_path}")

    contents = sio.loadmat(train_mat_path)
    positions = np.asarray(contents["positions"])
    magnitude = np.asarray(contents["magnitude"])

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SystemExit(f"[density] positions must be (N,3); got {positions.shape}")

    if magnitude.ndim != 3 or magnitude.shape[0] != positions.shape[0]:
        raise SystemExit(
            f"[density] magnitude must be (N,Nr,Nt) matching positions; got "
            f"{magnitude.shape} vs {positions.shape}"
        )

    return positions, magnitude


def dataset_scale_factor(positions: np.ndarray) -> float:
    """Reproduce ``DeepMIMODataset``'s auto-normalization scale factor exactly."""
    max_val = torch.tensor(positions, dtype=torch.float32).abs().max()
    return float(max_val) + DATASET_EPS


def extreme_sample_index(positions: np.ndarray) -> int:
    """Index of the sample carrying the largest ``max(|x|,|y|,|z|)``.

    Forcing this sample into every subset pins the dataset's auto-normalization
    scale factor, so all fractions share one coordinate scale.
    """
    return int(np.argmax(np.abs(positions).max(axis=1)))


def subsampling_permutation(num_samples: int) -> np.ndarray:
    """The one fixed permutation every fraction is carved out of."""
    return np.random.RandomState(SEED).permutation(num_samples)


def compute_keep_indices(
    num_samples: int,
    fraction: float,
    extreme_index: int,
    permutation: np.ndarray,
) -> np.ndarray:
    """Return the sorted kept indices for one fraction.

    ``round(fraction * N)`` entries are taken off the front of the shared
    permutation.  If the extreme sample did not land inside that prefix, it
    replaces the *last* kept entry, so the subset size never changes and the
    nesting property is preserved.
    """
    kept_count = int(round(float(fraction) * num_samples))
    kept_count = max(1, min(num_samples, kept_count))

    kept = permutation[:kept_count].copy()

    if extreme_index not in set(int(index) for index in kept):
        kept[-1] = extreme_index

    return np.sort(kept)


def build_keep_index_map(
    num_samples: int,
    extreme_index: int,
    fractions: Sequence[float] = FRACTIONS,
) -> Dict[float, np.ndarray]:
    """Build every fraction's index set and verify the nesting invariant."""
    permutation = subsampling_permutation(num_samples)

    index_map = {
        float(fraction): compute_keep_indices(
            num_samples, float(fraction), extreme_index, permutation
        )
        for fraction in fractions
    }

    ordered = sorted(index_map)

    for smaller, larger in zip(ordered, ordered[1:]):
        smaller_set = set(int(index) for index in index_map[smaller])
        larger_set = set(int(index) for index in index_map[larger])

        if not smaller_set.issubset(larger_set):
            missing = sorted(smaller_set - larger_set)[:10]
            raise AssertionError(
                f"[density] nesting violated: fraction {smaller} is not a subset "
                f"of {larger} (e.g. missing {missing})"
            )

        if extreme_index not in smaller_set:
            raise AssertionError(
                f"[density] extreme sample {extreme_index} missing from fraction {smaller}"
            )

    return index_map


def materialize_subset(
    dataset_dir: str,
    destination_dir: str,
    keep_indices: np.ndarray,
    positions: np.ndarray,
    magnitude: np.ndarray,
    expected_max_abs: float,
) -> Dict[str, object]:
    """Write a throwaway dataset directory holding the subsampled train.mat.

    ``test.mat`` and ``bs_info.yml`` are copied byte-for-byte; the originals are
    never touched.
    """
    if os.path.isdir(destination_dir):
        shutil.rmtree(destination_dir)

    os.makedirs(destination_dir, exist_ok=True)

    subset_positions = np.ascontiguousarray(positions[keep_indices])
    subset_magnitude = np.ascontiguousarray(magnitude[keep_indices])

    if subset_positions.shape[0] != int(keep_indices.shape[0]):
        raise AssertionError("[density] subset row count does not match the index set")

    sio.savemat(
        os.path.join(destination_dir, "train.mat"),
        {"positions": subset_positions, "magnitude": subset_magnitude},
        do_compression=False,
    )

    for name in ("test.mat", "bs_info.yml"):
        source = os.path.join(dataset_dir, name)

        if not os.path.isfile(source):
            raise SystemExit(f"[density] required dataset file is missing: {source}")

        shutil.copy2(source, os.path.join(destination_dir, name))

    # Read the file back so the assertion covers what training will actually see.
    written = sio.loadmat(os.path.join(destination_dir, "train.mat"))
    written_positions = np.asarray(written["positions"])
    written_max_abs = float(np.abs(written_positions).max())

    if written_max_abs != float(expected_max_abs):
        raise AssertionError(
            f"[density] normalization scale drifted in {destination_dir}: "
            f"max|positions| = {written_max_abs!r} != {float(expected_max_abs)!r}"
        )

    if written_positions.shape[0] != int(keep_indices.shape[0]):
        raise AssertionError("[density] written train.mat has the wrong sample count")

    return {
        "path": destination_dir,
        "n_train": int(keep_indices.shape[0]),
        "max_abs_position": written_max_abs,
        "normalization_scale_factor": dataset_scale_factor(written_positions),
    }


def read_bs_info(dataset_dir: str) -> Dict[str, object]:
    """Load ``bs_info.yml`` so the renderer can be rebuilt without the dataset dir."""
    with open(os.path.join(dataset_dir, "bs_info.yml"), "r", encoding="utf-8") as handle:
        info = yaml.safe_load(handle)

    return {
        "dataset_name": info.get("dataset_name", "mimo"),
        "bs_position": list(info["bs1"]["position"]),
        "bs_orientation": list(info["bs1"]["orientation"]),
    }


def resolve_array_shapes(
    model_params: Dict[str, object], beam_rows: int, beam_cols: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Resolve the Rx/Tx UPA shapes exactly the way ``Scene`` resolves them."""
    from types import SimpleNamespace

    from scene import Scene

    namespace = SimpleNamespace(**model_params)

    return (
        tuple(Scene._resolve_array_shape(namespace, "rx", beam_rows)),
        tuple(Scene._resolve_array_shape(namespace, "tx", beam_cols)),
    )


# ---------------------------------------------------------------------------
# Subprocess plumbing (shared with train_density_MLP.py)
# ---------------------------------------------------------------------------


def subprocess_environment() -> Dict[str, str]:
    """Environment with the repo root on ``PYTHONPATH``.

    ``evaluation/train_MLP.py`` imports ``arguments`` / ``scene`` as top-level
    modules, which only resolves when the repo root is importable.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        REPO_ROOT if not existing else REPO_ROOT + os.pathsep + existing
    )
    return environment


def run_and_tee(command: Sequence[str], cwd: str) -> Tuple[int, List[str]]:
    """Run a command, stream its output live, and return every emitted line.

    tqdm redraws its bar with carriage returns, so the stream is split on both
    ``\\r`` and ``\\n``; every redraw becomes its own record and the loss
    trajectory can be recovered from the postfix.
    """
    print(f"[density] $ {' '.join(command)}", flush=True)

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    lines: List[str] = []
    pending = ""

    assert process.stdout is not None

    while True:
        chunk = process.stdout.read(4096)

        if not chunk:
            break

        text = chunk.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()

        pending += text
        parts = re.split(r"[\r\n]", pending)
        pending = parts.pop()
        lines.extend(part for part in parts if part.strip())

    if pending.strip():
        lines.append(pending)

    process.stdout.close()
    returncode = process.wait()

    sys.stdout.write("\n")
    sys.stdout.flush()

    return returncode, lines


def print_failure_tail(lines: Sequence[str]) -> None:
    """Print the tail of a failed subprocess so the cause is visible."""
    tail = list(lines)[-_FAILURE_TAIL_LINES:]

    print(f"[density] --- 실패한 서브프로세스 출력 마지막 {len(tail)}줄 ---")

    for line in tail:
        print(f"[density] | {line}")

    print("[density] --- 출력 끝 ---")


def parse_loss_trajectory(lines: Sequence[str]) -> List[float]:
    """Recover the loss trajectory from ``train.py``'s tqdm postfix."""
    pattern = re.compile(r"Loss=([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")
    trajectory: List[float] = []

    for line in lines:
        for match in pattern.finditer(line):
            try:
                trajectory.append(float(match.group(1)))
            except ValueError:
                continue

    return trajectory


def find_run_dir(lines: Sequence[str], before: Sequence[str]) -> Optional[str]:
    """Locate the run directory ``train.py`` created for this launch."""
    pattern = re.compile(r"Output path:\s*(\S.*?)\s*$")

    for line in reversed(list(lines)):
        match = pattern.search(line)

        if match:
            candidate = match.group(1)

            if not os.path.isabs(candidate):
                candidate = os.path.join(REPO_ROOT, candidate)

            if os.path.isdir(candidate):
                return os.path.abspath(candidate)

    # Fallback: whichever outputs/ directory appeared while the run was alive.
    outputs_root = os.path.join(REPO_ROOT, "outputs")
    after = set(os.listdir(outputs_root)) if os.path.isdir(outputs_root) else set()
    created = sorted(after - set(before))

    if len(created) == 1:
        return os.path.join(outputs_root, created[0])

    return None


def snapshot_outputs() -> List[str]:
    outputs_root = os.path.join(REPO_ROOT, "outputs")
    return sorted(os.listdir(outputs_root)) if os.path.isdir(outputs_root) else []


# ---------------------------------------------------------------------------
# Convergence heuristic (shared with train_density_MLP.py)
# ---------------------------------------------------------------------------


def convergence_check(trajectory: Sequence[float]) -> Tuple[bool, str]:
    """Flag a run whose loss is still falling fast over its final 10%.

    The tail window is split in half and the two halves' means are compared, so
    a single noisy sample cannot trigger the warning.  This only reports; the
    epoch count is never adjusted automatically.
    """
    values = [float(value) for value in trajectory if np.isfinite(value)]

    if len(values) < 4:
        return False, "손실 기록이 부족하여 수렴 판정 생략"

    window_size = max(4, int(np.ceil(0.1 * len(values))))
    window = values[-window_size:]
    half = len(window) // 2

    early = float(np.mean(window[:half]))
    late = float(np.mean(window[half:]))

    if not np.isfinite(early) or abs(early) < 1e-30:
        return False, "손실이 0에 가까워 상대 변화 판정 생략"

    relative_drop = (early - late) / abs(early)

    if relative_drop > CONVERGENCE_TOLERANCE:
        return True, (
            f"마지막 10% 구간에서 손실이 여전히 {relative_drop * 100.0:.1f}% 감소 "
            f"({early:.6g} -> {late:.6g}), 50 에폭 수렴 미달 가능"
        )

    return False, f"마지막 10% 구간 상대 감소 {relative_drop * 100.0:.1f}% (임계 5% 이하)"


# ---------------------------------------------------------------------------
# Repacking
# ---------------------------------------------------------------------------


def output_path_for(output_dir: str, fraction: float) -> str:
    return os.path.join(output_dir, f"model_{FRACTION_TAG[float(fraction)]}.pth")


def existing_repack_is_loadable(path: str, required_keys: Sequence[str]) -> bool:
    """Resume support: an existing repack counts only if it actually loads."""
    if not os.path.isfile(path):
        return False

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - any failure means "redo it"
        print(f"[density] 기존 파일을 읽을 수 없어 다시 생성합니다: {path} ({error})")
        return False

    if not isinstance(payload, dict) or any(key not in payload for key in required_keys):
        print(f"[density] 기존 파일의 형식이 달라 다시 생성합니다: {path}")
        return False

    return True


def repack_mimogs_checkpoint(
    run_dir: str,
    destination: str,
    fraction: float,
    dataset_dir: str,
    n_train: int,
    normalization_scale_factor: float,
    max_abs_position: float,
    final_loss: Optional[float],
    train_seconds: Optional[float],
    loss_trajectory: Sequence[float],
    epochs: int = EPOCHS,
) -> str:
    """Repack a ``train.py`` run into one self-contained checkpoint."""
    checkpoint_path = os.path.join(run_dir, "model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"[density] model.pth is missing in {run_dir}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_params = dict(raw["model_params"])
    opt_params = dict(raw["opt_params"])

    beam_rows, beam_cols = beam_grid_from_dataset(dataset_dir)
    rx_shape, tx_shape = resolve_array_shapes(model_params, beam_rows, beam_cols)
    bs_info = read_bs_info(dataset_dir)

    run_args_path = os.path.join(run_dir, "run_args.txt")
    run_args_text = ""

    if os.path.isfile(run_args_path):
        with open(run_args_path, "r", encoding="utf-8") as handle:
            run_args_text = handle.read()

    config = {
        # Everything train.py itself recorded, so the model can be rebuilt.
        "model_params": model_params,
        "opt_params": opt_params,
        "run_args_text": run_args_text,
        # Renderer geometry.
        "beam_rows": int(beam_rows),
        "beam_cols": int(beam_cols),
        "beam_grid": (int(beam_rows), int(beam_cols)),
        "rx_shape": tuple(int(value) for value in rx_shape),
        "tx_shape": tuple(int(value) for value in tx_shape),
        "bs_position": bs_info["bs_position"],
        "bs_orientation": bs_info["bs_orientation"],
        "dataset_name": bs_info["dataset_name"],
        # Data provenance.  ``dataset_path`` is the pristine dataset (test.mat
        # lives there); ``train_source_path`` is the throwaway subset directory
        # and is normally deleted right after this repack.
        "dataset_path": os.path.abspath(dataset_dir),
        "train_source_path": str(model_params.get("source_path", "")),
        "n_train": int(n_train),
        "normalization_scale_factor": float(normalization_scale_factor),
        "max_abs_position": float(max_abs_position),
        "iteration": int(raw.get("iteration", 0)),
    }

    payload = {
        "capture": raw["gaussians"],
        "config": config,
        "fraction": float(fraction),
        "seed": SEED,
        "epochs": int(epochs),
        "source_run": os.path.basename(os.path.normpath(run_dir)),
        "final_loss": None if final_loss is None else float(final_loss),
        "train_seconds": None if train_seconds is None else float(train_seconds),
        "loss_trajectory": [float(value) for value in loss_trajectory],
    }

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(payload, destination)

    print(f"[density] 재패킹 완료 -> {destination}")

    return destination


def beam_grid_from_dataset(dataset_dir: str) -> Tuple[int, int]:
    """Read the ``(Nr, Nt)`` beam grid straight off the dataset."""
    contents = sio.loadmat(os.path.join(dataset_dir, "train.mat"))
    magnitude = np.asarray(contents["magnitude"])

    if magnitude.ndim != 3:
        raise SystemExit(f"[density] magnitude must be (N,Nr,Nt); got {magnitude.shape}")

    return int(magnitude.shape[1]), int(magnitude.shape[2])


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------


def smoke_test_mimogs(path: str) -> None:
    """Rebuild the model from ``config`` alone and render three test locations.

    Nothing here touches the original run directory: only the repacked file and
    the pristine dataset's ``test.mat`` are read.
    """
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

    # Three test locations, normalized the way DeepMIMODataset normalizes them.
    test_mat = sio.loadmat(os.path.join(config["dataset_path"], "test.mat"))
    test_positions = torch.tensor(np.asarray(test_mat["positions"]), dtype=torch.float32)
    test_scale = float(test_positions.abs().max()) + DATASET_EPS
    rx_pos = (test_positions[:3] / test_scale).to(device)

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

    expected = (int(config["beam_rows"]), int(config["beam_cols"]))

    if rendered.shape[0] != 3:
        raise AssertionError(f"[smoke] expected 3 rendered locations, got {rendered.shape[0]}")

    for index in range(rendered.shape[0]):
        single = rendered[index]

        if tuple(single.shape) != expected:
            raise AssertionError(
                f"[smoke] location {index}: shape {tuple(single.shape)} != {expected}"
            )

        if not bool(torch.isfinite(single).all()):
            raise AssertionError(f"[smoke] location {index}: non-finite values in the render")

    print(
        f"[smoke] OK {os.path.basename(path)} | fraction={payload['fraction']} | "
        f"render {tuple(rendered.shape)} | finite | "
        f"range [{float(rendered.min()):.4g}, {float(rendered.max()):.4g}]"
    )


def run_smoke_test_subprocess(script_path: str, checkpoint_path: str) -> bool:
    """Run the smoke test in a fresh interpreter so the reload is truly standalone."""
    command = [sys.executable, script_path, "--smoke_test", checkpoint_path]
    returncode, lines = run_and_tee(command, cwd=REPO_ROOT)

    if returncode != 0:
        print_failure_tail(lines)

    return returncode == 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_korean_summary(rows: Sequence[Dict[str, object]], title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)
    header = (
        f"{'비율':>8}  {'n_train':>8}  {'최종 loss':>14}  {'소요 시간':>12}  {'상태':<10}  파일"
    )
    print(header)
    print("-" * 100)

    for row in rows:
        fraction = float(row["fraction"])
        n_train = row.get("n_train")
        final_loss = row.get("final_loss")
        seconds = row.get("train_seconds")
        status = str(row.get("status", ""))
        path = str(row.get("path") or "-")

        n_train_text = "-" if n_train is None else f"{int(n_train):,}"
        loss_text = "-" if final_loss is None else f"{float(final_loss):.8f}"

        if seconds is None:
            time_text = "-"
        else:
            time_text = f"{float(seconds) / 60.0:.1f} 분"

        display_path = path

        if path != "-" and os.path.isabs(path):
            common = os.path.commonpath([os.path.abspath(path), REPO_ROOT])

            if common == REPO_ROOT:
                display_path = os.path.relpath(path, REPO_ROOT)

        print(
            f"{fraction * 100.0:>7.2f}%  {n_train_text:>8}  {loss_text:>14}  "
            f"{time_text:>12}  {status:<10}  {display_path}"
        )

    print("-" * 100)

    warnings = [row for row in rows if row.get("convergence_warning")]

    if warnings:
        print("[수렴 경고] 아래 실행은 50 에폭 시점에서 아직 손실이 유의미하게 감소 중입니다.")

        for row in warnings:
            print(f"  - 비율 {float(row['fraction']) * 100.0:.2f}% : {row.get('convergence_note', '')}")

        print("  (에폭 수는 자동으로 바꾸지 않았습니다. 필요하면 직접 늘려 주세요.)")
    else:
        print("[수렴 경고] 없음 (마지막 10% 구간 손실 감소가 모두 5% 미만이거나 판정 대상 아님)")

    failures = [row for row in rows if str(row.get("status", "")) != "성공"]

    if failures:
        print("")
        print("[실패/건너뜀]")

        for row in failures:
            print(
                f"  - 비율 {float(row['fraction']) * 100.0:.2f}% : "
                f"{row.get('status')} / {row.get('note', '')}"
            )

    print("=" * 100)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_fraction_argument(values: Optional[Sequence[str]]) -> List[float]:
    """Accept ``--fractions 0.25`` and ``--fractions 0.0625,0.125``."""
    if not values:
        return list(FRACTIONS)

    tokens: List[str] = []

    for value in values:
        tokens.extend(token for token in str(value).replace(",", " ").split() if token)

    selected: List[float] = []

    for token in tokens:
        try:
            fraction = float(token)
        except ValueError as error:
            raise SystemExit(f"[density] --fractions 값을 해석할 수 없습니다: {token!r}") from error

        matched = [known for known in FRACTIONS if abs(known - fraction) < 1e-12]

        if not matched:
            raise SystemExit(
                f"[density] 지원하지 않는 비율입니다: {fraction}. "
                f"가능한 값: {', '.join(str(value) for value in FRACTIONS)}"
            )

        selected.append(float(matched[0]))

    return sorted(set(selected))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MIMO-GS density sweep: train train.py on nested subsets and repack",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset directory (default: the source_path default in arguments/__init__.py)",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        default=None,
        help="Subset of 0.0625 0.125 0.25 0.5 1.0 (comma or space separated)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MIMOGS_OUTPUT_DIR,
        help="Where the repacked model_XX.pth files land",
    )
    parser.add_argument(
        "--reference_run",
        type=str,
        default=REFERENCE_RUN_DIR,
        help="Existing full-data 50-epoch run repacked as the 100%% point",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep ./.density_tmp/<name>/ instead of deleting it after each fraction",
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
        smoke_test_mimogs(arguments.smoke_test)
        return 0

    dataset_dir = os.path.abspath(arguments.dataset) if arguments.dataset else default_dataset_dir()

    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[density] 데이터셋 디렉터리가 없습니다: {dataset_dir}")

    output_dir = os.path.abspath(arguments.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    selected_fractions = parse_fraction_argument(arguments.fractions)

    positions, magnitude = load_train_mat(dataset_dir)
    num_samples = int(positions.shape[0])
    extreme_index = extreme_sample_index(positions)
    full_max_abs = float(np.abs(positions).max())
    full_scale_factor = dataset_scale_factor(positions)

    # Nesting/extreme-sample invariants are checked over the full ladder, not
    # just the selected subset, so a pilot run still validates the whole design.
    index_map = build_keep_index_map(num_samples, extreme_index, FRACTIONS)

    print("=" * 100)
    print("[density] MIMO-GS 학습 데이터 밀도 스윕")
    print("=" * 100)
    print(f"  dataset            : {dataset_dir}")
    print(f"  N (train.mat)      : {num_samples:,}")
    print(f"  extreme sample idx : {extreme_index} (max|coord| = {full_max_abs:.6f})")
    print(f"  scale factor       : {full_scale_factor:.6f}")
    print(f"  epochs             : {arguments.epochs}")
    print(f"  seed               : {SEED}")
    print(f"  fractions          : {selected_fractions}")
    print(f"  output_dir         : {output_dir}")
    print("  nesting / scale invariants: OK")
    print("")

    rows: List[Dict[str, object]] = []
    had_failure = False

    for fraction in selected_fractions:
        tag = FRACTION_TAG[fraction]
        keep_indices = index_map[fraction]
        destination = output_path_for(output_dir, fraction)

        row: Dict[str, object] = {
            "fraction": fraction,
            "n_train": int(keep_indices.shape[0]),
            "final_loss": None,
            "train_seconds": None,
            "status": "실패",
            "path": None,
            "note": "",
            "convergence_warning": False,
            "convergence_note": "",
        }

        print("-" * 100)
        print(f"[density] 비율 {fraction * 100.0:.2f}% (n_train={int(keep_indices.shape[0]):,}) 시작")

        if not arguments.overwrite and existing_repack_is_loadable(
            destination, ("capture", "config", "fraction")
        ):
            print(f"[density] 이미 존재하여 건너뜁니다: {destination}")
            row["status"] = "성공"
            row["path"] = destination
            row["note"] = "기존 결과 재사용 (resume)"

            existing = torch.load(destination, map_location="cpu", weights_only=False)
            row["final_loss"] = existing.get("final_loss")
            row["train_seconds"] = existing.get("train_seconds")
            rows.append(row)
            continue

        if fraction == 1.0:
            reference_run = os.path.abspath(arguments.reference_run)

            if not os.path.isfile(os.path.join(reference_run, "model.pth")):
                row["note"] = f"기준 실행을 찾을 수 없음: {reference_run}"
                print(f"[density] {row['note']}")
                had_failure = True
                rows.append(row)
                continue

            print(f"[density] 100% 지점은 학습하지 않고 기존 실행을 재패킹합니다: {reference_run}")

            try:
                repack_mimogs_checkpoint(
                    run_dir=reference_run,
                    destination=destination,
                    fraction=fraction,
                    dataset_dir=dataset_dir,
                    n_train=num_samples,
                    normalization_scale_factor=full_scale_factor,
                    max_abs_position=full_max_abs,
                    final_loss=None,
                    train_seconds=None,
                    loss_trajectory=[],
                    epochs=int(arguments.epochs),
                )
            except Exception as error:  # noqa: BLE001
                row["note"] = f"재패킹 실패: {error}"
                print(f"[density] {row['note']}")
                had_failure = True
                rows.append(row)
                continue

            row["convergence_note"] = "학습 로그 없음 (기존 실행 재패킹)"
            row["note"] = "기존 50 에폭 전체 데이터 실행 재패킹"
        else:
            temp_dir = os.path.join(TEMP_ROOT, f"frac_{tag}")

            try:
                subset_info = materialize_subset(
                    dataset_dir=dataset_dir,
                    destination_dir=temp_dir,
                    keep_indices=keep_indices,
                    positions=positions,
                    magnitude=magnitude,
                    expected_max_abs=full_max_abs,
                )
            except Exception as error:  # noqa: BLE001
                row["note"] = f"임시 데이터셋 생성 실패: {error}"
                print(f"[density] {row['note']}")
                had_failure = True
                rows.append(row)
                continue

            print(
                f"[density] 임시 데이터셋: {temp_dir} "
                f"(n_train={subset_info['n_train']:,}, "
                f"scale={float(subset_info['normalization_scale_factor']):.6f})"
            )

            before = snapshot_outputs()
            started = time.perf_counter()

            # ``num_epochs`` is a ModelParams field, so 50 epochs is expressed
            # through train.py's own CLI without touching the file.
            command = [
                sys.executable,
                os.path.join(REPO_ROOT, "train.py"),
                "--source_path",
                temp_dir,
                "--num_epochs",
                str(int(arguments.epochs)),
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
                had_failure = True

                if not arguments.keep_temp:
                    shutil.rmtree(temp_dir, ignore_errors=True)

                rows.append(row)
                continue

            run_dir = find_run_dir(lines, before)

            if run_dir is None:
                row["note"] = "train.py 가 만든 실행 디렉터리를 찾지 못함"
                print(f"[density] {row['note']}")
                had_failure = True

                if not arguments.keep_temp:
                    shutil.rmtree(temp_dir, ignore_errors=True)

                rows.append(row)
                continue

            print(f"[density] 실행 디렉터리: {run_dir}")

            try:
                repack_mimogs_checkpoint(
                    run_dir=run_dir,
                    destination=destination,
                    fraction=fraction,
                    dataset_dir=dataset_dir,
                    n_train=int(subset_info["n_train"]),
                    normalization_scale_factor=float(subset_info["normalization_scale_factor"]),
                    max_abs_position=float(subset_info["max_abs_position"]),
                    final_loss=row["final_loss"],
                    train_seconds=elapsed,
                    loss_trajectory=trajectory,
                    epochs=int(arguments.epochs),
                )
            except Exception as error:  # noqa: BLE001
                row["note"] = f"재패킹 실패: {error}"
                print(f"[density] {row['note']}")
                had_failure = True

                if not arguments.keep_temp:
                    shutil.rmtree(temp_dir, ignore_errors=True)

                rows.append(row)
                continue

            row["note"] = f"run dir: {os.path.relpath(run_dir, REPO_ROOT)}"

            warned, note = convergence_check(trajectory)
            row["convergence_warning"] = warned
            row["convergence_note"] = note

            if arguments.keep_temp:
                print(f"[density] --keep_temp 지정: {temp_dir} 유지")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[density] 임시 디렉터리 삭제: {temp_dir}")

        if run_smoke_test_subprocess(os.path.abspath(__file__), destination):
            row["status"] = "성공"
            row["path"] = destination
        else:
            row["status"] = "스모크실패"
            row["note"] = (row["note"] + " / " if row["note"] else "") + "독립 재로드 스모크 테스트 실패"
            had_failure = True

        rows.append(row)

    if not arguments.keep_temp and os.path.isdir(TEMP_ROOT) and not os.listdir(TEMP_ROOT):
        os.rmdir(TEMP_ROOT)

    print_korean_summary(rows, "[density] MIMO-GS 밀도 스윕 요약")

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
