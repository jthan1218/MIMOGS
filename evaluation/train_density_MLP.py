#!/usr/bin/env python3
"""Position-MLP density sweep trainer.

Trains the *unmodified* ``evaluation/train_MLP.py`` (``mlp_medium`` only:
hidden 512, depth 6) on the same nested random subsets ``train_density.py``
builds, and repacks every result into a self-contained checkpoint::

    outputs/density/MLP/model_{6,12,25,50,100}.pth

The subsampling rule, the temp-dir materialization, the subprocess plumbing and
the convergence heuristic are all imported from ``train_density.py`` so the two
sweeps are carved out of bit-identical index sets.

``evaluation/train_MLP.py`` imports ``arguments`` / ``scene`` as top-level
modules but does not put the repo root on ``sys.path``, so it is launched as
``python -m evaluation.train_MLP`` from the repo root (with the repo root also
pushed onto ``PYTHONPATH``).  Running it as ``python evaluation/train_MLP.py``
fails with ``ModuleNotFoundError: No module named 'arguments'``.

Zero-argument runnable::

    python train_density_MLP.py

``./.density_tmp/`` is scratch space and should not be committed.  ``.gitignore``
is an existing file and is deliberately left untouched, so add the entry by hand
(or rely on the directory being removed automatically after every fraction).
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
    FRACTION_TAG,
    FRACTIONS,
    REPO_ROOT,
    SEED,
    TEMP_ROOT,
    build_keep_index_map,
    convergence_check,
    dataset_scale_factor,
    default_dataset_dir,
    existing_repack_is_loadable,
    extreme_sample_index,
    load_train_mat,
    materialize_subset,
    output_path_for,
    parse_fraction_argument,
    print_failure_tail,
    print_korean_summary,
    run_and_tee,
)


EPOCHS = 50
CONFIG_NAME = "mlp_medium"
EXPECTED_HIDDEN = 512
EXPECTED_DEPTH = 6

MLP_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "density", "MLP")

# train_MLP.py writes ``<outputs_root>/<config name>/model.pth``.  A per-fraction
# outputs root keeps the five runs from overwriting each other and leaves the raw
# run dirs on disk next to the repacked checkpoints.
MLP_RUN_ROOT = os.path.join(MLP_OUTPUT_DIR, "_runs")

TRAIN_MLP_MODULE = "evaluation.train_MLP"


# ---------------------------------------------------------------------------
# CLI capability check
# ---------------------------------------------------------------------------


def verify_train_mlp_cli() -> Optional[str]:
    """Confirm train_MLP.py can express "mlp_medium only, 50 epochs" as-is.

    Returns a Korean error message when it cannot, so the caller can stop
    instead of editing the file.
    """
    try:
        from evaluation.train_MLP import CONFIGS
    except Exception as error:  # noqa: BLE001
        return (
            f"evaluation/train_MLP.py 를 import 할 수 없습니다: {error}. "
            f"저장소 루트에서 실행 중인지 확인하세요."
        )

    if CONFIG_NAME not in CONFIGS:
        return (
            f"evaluation/train_MLP.py 의 CONFIGS 에 '{CONFIG_NAME}' 가 없습니다. "
            f"존재하는 설정: {sorted(CONFIGS)}"
        )

    config = CONFIGS[CONFIG_NAME]

    if int(config.get("hidden", -1)) != EXPECTED_HIDDEN or int(config.get("depth", -1)) != EXPECTED_DEPTH:
        return (
            f"'{CONFIG_NAME}' 의 구조가 요구 사항(hidden={EXPECTED_HIDDEN}, "
            f"depth={EXPECTED_DEPTH})과 다릅니다: {config}"
        )

    returncode, lines = run_and_tee(
        [sys.executable, "-m", TRAIN_MLP_MODULE, "--help"], cwd=REPO_ROOT
    )

    if returncode != 0:
        print_failure_tail(lines)
        return "evaluation/train_MLP.py --help 실행에 실패했습니다."

    help_text = "\n".join(lines)
    missing = [
        flag
        for flag in ("--configs", "--epochs", "--source_path", "--outputs_root")
        if flag not in help_text
    ]

    if missing:
        return (
            f"evaluation/train_MLP.py 의 CLI 에 필요한 옵션이 없습니다: {missing}. "
            f"파일을 수정하지 않고는 요구 사항을 표현할 수 없어 중단합니다."
        )

    return None


# ---------------------------------------------------------------------------
# Repacking
# ---------------------------------------------------------------------------


def repack_mlp_checkpoint(
    run_dir: str,
    destination: str,
    fraction: float,
    dataset_dir: str,
    n_train: int,
    normalization_scale_factor: float,
    train_seconds: Optional[float],
    epochs: int = EPOCHS,
) -> Tuple[str, List[float]]:
    """Repack a train_MLP.py run into one self-contained checkpoint."""
    checkpoint_path = os.path.join(run_dir, "model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"[density-mlp] model.pth is missing in {run_dir}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = dict(raw["config"])
    hidden = int(config["hidden"])
    depth = int(config["depth"])
    num_outputs = int(config["num_outputs"])
    beam_rows = int(config["beam_rows"])
    beam_cols = int(config["beam_cols"])

    if hidden != EXPECTED_HIDDEN or depth != EXPECTED_DEPTH:
        raise AssertionError(
            f"[density-mlp] unexpected architecture: hidden={hidden}, depth={depth}"
        )

    if num_outputs != beam_rows * beam_cols:
        raise AssertionError(
            f"[density-mlp] num_outputs {num_outputs} != {beam_rows}x{beam_cols}"
        )

    trajectory = [float(entry["train_loss"]) for entry in raw.get("trajectory", [])]

    payload = {
        "state_dict": raw["state_dict"],
        "arch": {
            "hidden": hidden,
            "depth": depth,
            # Read off the run instead of assumed, so a future PE change is caught.
            "num_frequencies": int(config["num_frequencies"]),
            "include_input": bool(config["include_input"]),
            "num_outputs": num_outputs,
        },
        "fraction": float(fraction),
        "seed": SEED,
        "epochs": int(epochs),
        "final_loss": trajectory[-1] if trajectory else None,
        "normalization_scale_factor": float(normalization_scale_factor),
        # Extras needed to reshape and to reach test.mat without the run dir.
        "beam_rows": beam_rows,
        "beam_cols": beam_cols,
        "dataset_path": os.path.abspath(dataset_dir),
        "train_source_path": str(raw.get("training", {}).get("source_path", "")),
        "n_train": int(n_train),
        "train_seconds": float(train_seconds) if train_seconds is not None else raw.get("train_seconds"),
        "source_run": os.path.relpath(run_dir, REPO_ROOT),
        "loss_trajectory": trajectory,
        "test_trajectory": raw.get("trajectory", []),
        "parameters": int(raw.get("parameters", 0)),
    }

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(payload, destination)

    print(f"[density-mlp] 재패킹 완료 -> {destination}")

    return destination, trajectory


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------


def smoke_test_mlp(path: str) -> None:
    """Rebuild PositionMLP from ``arch`` alone and forward three test positions."""
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

    test_mat = sio.loadmat(os.path.join(payload["dataset_path"], "test.mat"))
    test_positions = torch.tensor(np.asarray(test_mat["positions"]), dtype=torch.float32)
    test_scale = float(test_positions.abs().max()) + DATASET_EPS
    positions = (test_positions[:3] / test_scale).to(device)

    with torch.inference_mode():
        predicted = model(positions)

    if tuple(predicted.shape) != (3, int(arch["num_outputs"])):
        raise AssertionError(
            f"[smoke] forward returned {tuple(predicted.shape)}, expected (3, {arch['num_outputs']})"
        )

    maps = predicted.reshape(3, beam_rows, beam_cols)

    for index in range(maps.shape[0]):
        single = maps[index]

        if tuple(single.shape) != (beam_rows, beam_cols):
            raise AssertionError(
                f"[smoke] location {index}: shape {tuple(single.shape)} != ({beam_rows}, {beam_cols})"
            )

        if not bool(torch.isfinite(single).all()):
            raise AssertionError(f"[smoke] location {index}: non-finite values in the output")

    print(
        f"[smoke] OK {os.path.basename(path)} | fraction={payload['fraction']} | "
        f"output {tuple(maps.shape)} | finite | "
        f"range [{float(maps.min()):.4g}, {float(maps.max()):.4g}]"
    )


def run_smoke_test_subprocess(script_path: str, checkpoint_path: str) -> bool:
    """Run the smoke test in a fresh interpreter so the reload is truly standalone."""
    command = [sys.executable, script_path, "--smoke_test", checkpoint_path]
    returncode, lines = run_and_tee(command, cwd=REPO_ROOT)

    if returncode != 0:
        print_failure_tail(lines)

    return returncode == 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Position-MLP density sweep: train evaluation/train_MLP.py on nested subsets",
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
        default=MLP_OUTPUT_DIR,
        help="Where the repacked model_XX.pth files land",
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
        smoke_test_mlp(arguments.smoke_test)
        return 0

    dataset_dir = os.path.abspath(arguments.dataset) if arguments.dataset else default_dataset_dir()

    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[density-mlp] 데이터셋 디렉터리가 없습니다: {dataset_dir}")

    output_dir = os.path.abspath(arguments.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    selected_fractions = parse_fraction_argument(arguments.fractions)

    cli_problem = verify_train_mlp_cli()

    if cli_problem is not None:
        print("")
        print("=" * 100)
        print("[density-mlp] 중단: evaluation/train_MLP.py 를 수정하지 않고는 요구 조건을 만족할 수 없습니다.")
        print(f"[density-mlp] 사유: {cli_problem}")
        print("=" * 100)
        return 1

    positions, magnitude = load_train_mat(dataset_dir)
    num_samples = int(positions.shape[0])
    extreme_index = extreme_sample_index(positions)
    full_max_abs = float(np.abs(positions).max())
    full_scale_factor = dataset_scale_factor(positions)

    index_map = build_keep_index_map(num_samples, extreme_index, FRACTIONS)

    print("=" * 100)
    print("[density-mlp] Position-MLP 학습 데이터 밀도 스윕 (mlp_medium: hidden 512 / depth 6)")
    print("=" * 100)
    print(f"  dataset            : {dataset_dir}")
    print(f"  N (train.mat)      : {num_samples:,}")
    print(f"  extreme sample idx : {extreme_index} (max|coord| = {full_max_abs:.6f})")
    print(f"  scale factor       : {full_scale_factor:.6f}")
    print(f"  epochs             : {arguments.epochs}")
    print(f"  seed               : {SEED}")
    print(f"  fractions          : {selected_fractions}")
    print(f"  output_dir         : {output_dir}")
    print(f"  run dirs           : {MLP_RUN_ROOT}/frac_<tag>/{CONFIG_NAME}/")
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
        print(
            f"[density-mlp] 비율 {fraction * 100.0:.2f}% "
            f"(n_train={int(keep_indices.shape[0]):,}) 시작"
        )

        if not arguments.overwrite and existing_repack_is_loadable(
            destination, ("state_dict", "arch", "fraction")
        ):
            print(f"[density-mlp] 이미 존재하여 건너뜁니다: {destination}")
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            row["status"] = "성공"
            row["path"] = destination
            row["note"] = "기존 결과 재사용 (resume)"
            row["final_loss"] = existing.get("final_loss")
            row["train_seconds"] = existing.get("train_seconds")
            rows.append(row)
            continue

        temp_dir = os.path.join(TEMP_ROOT, f"mlp_frac_{tag}")

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
            print(f"[density-mlp] {row['note']}")
            had_failure = True
            rows.append(row)
            continue

        print(
            f"[density-mlp] 임시 데이터셋: {temp_dir} "
            f"(n_train={subset_info['n_train']:,}, "
            f"scale={float(subset_info['normalization_scale_factor']):.6f})"
        )

        run_root = os.path.join(MLP_RUN_ROOT, f"frac_{tag}")
        os.makedirs(run_root, exist_ok=True)

        started = time.perf_counter()

        # ``--epochs`` and ``--configs`` are train_MLP.py's own CLI options, so
        # 50 epochs / mlp_medium-only is expressed without touching the file.
        command = [
            sys.executable,
            "-m",
            TRAIN_MLP_MODULE,
            "--configs",
            CONFIG_NAME,
            "--epochs",
            str(int(arguments.epochs)),
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
            had_failure = True

            if not arguments.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

            rows.append(row)
            continue

        run_dir = os.path.join(run_root, CONFIG_NAME)

        if not os.path.isfile(os.path.join(run_dir, "model.pth")):
            row["note"] = f"train_MLP.py 결과를 찾지 못함: {run_dir}/model.pth"
            print(f"[density-mlp] {row['note']}")
            had_failure = True

            if not arguments.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

            rows.append(row)
            continue

        print(f"[density-mlp] 실행 디렉터리: {run_dir}")

        try:
            _, trajectory = repack_mlp_checkpoint(
                run_dir=run_dir,
                destination=destination,
                fraction=fraction,
                dataset_dir=dataset_dir,
                n_train=int(subset_info["n_train"]),
                normalization_scale_factor=float(subset_info["normalization_scale_factor"]),
                train_seconds=elapsed,
                epochs=int(arguments.epochs),
            )
        except Exception as error:  # noqa: BLE001
            row["note"] = f"재패킹 실패: {error}"
            print(f"[density-mlp] {row['note']}")
            had_failure = True

            if not arguments.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

            rows.append(row)
            continue

        row["final_loss"] = trajectory[-1] if trajectory else None
        row["note"] = f"run dir: {os.path.relpath(run_dir, REPO_ROOT)}"

        warned, note = convergence_check(trajectory)
        row["convergence_warning"] = warned
        row["convergence_note"] = note

        if arguments.keep_temp:
            print(f"[density-mlp] --keep_temp 지정: {temp_dir} 유지")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[density-mlp] 임시 디렉터리 삭제: {temp_dir}")

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

    print_korean_summary(rows, "[density-mlp] Position-MLP 밀도 스윕 요약")

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
