"""Literature-anchored beam-selection baselines used by ``eval_net_rate.py``.

A site-specific predictor that consumes ONLY the training split:

* :class:`BeamClassifier` / :func:`train_beam_classifier` -- the position-aided
  beam predictor of Morais et al. (IEEE ICC, 2023 / arXiv:2205.09054): a fully
  connected network on the min-max normalized, coarsely quantized 2-D position,
  with a softmax head over the transmit codebook.

It exposes the SAME interface as the other schemes in ``eval_net_rate``: a
full ``(N_test, N_t)`` transmit-beam ordering, so that ``[:, :L_t]`` is the
selected set at any budget.  It does not depend on the SNR, so the ordering is
built once and reused across the whole sweep.

The classifier deliberately does NOT use this repository's Fourier positional
encoding: the published model consumes the raw normalized coordinate, and the
point of the baseline is to reproduce it rather than to improve on it.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------
# Position-aided NN beam classifier (Morais et al., ICC 2023)
# ----------------------------------------------------------------------
CLASSIFIER_HIDDEN = 256
CLASSIFIER_LAYERS = 3
CLASSIFIER_EPOCHS = 60
CLASSIFIER_BATCH = 32
CLASSIFIER_LR = 0.01
CLASSIFIER_MILESTONES = (20, 40)
CLASSIFIER_GAMMA = 0.2
CLASSIFIER_VAL_FRACTION = 0.25
POSITION_BINS = 200
CLASSIFIER_NAME = "model.pth"


def normalize_and_quantize(
    positions_xy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bins: int = POSITION_BINS,
) -> np.ndarray:
    """Min-max the 2-D position with TRAIN statistics, then quantize.

    The paper feeds the network a coarsely quantized coordinate rather than a
    continuous one: each normalized axis is rounded onto a ``bins``-step grid
    (resolution ``1/bins``).  Values are clipped to ``[0, 1]`` first so a test
    point outside the training bounding box cannot leave the grid.
    """
    span = np.maximum(upper - lower, np.finfo(np.float64).tiny)
    normalized = (positions_xy.astype(np.float64) - lower) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.round(normalized * float(bins)) / float(bins)


class BeamClassifier(nn.Module):
    """Fully connected beam classifier: 2-D position -> 64-way softmax.

    Architecture exactly as specified in Morais et al. (ICC 2023): three
    hidden layers of 256 ReLU units on the raw (normalized, quantized) 2-D
    coordinate.  No Fourier / positional encoding is applied -- that is a
    deliberate departure from the other coordinate models in this repository,
    because the point of this baseline is to reproduce the published network.
    """

    def __init__(
        self,
        num_beams: int,
        in_dim: int = 2,
        hidden: int = CLASSIFIER_HIDDEN,
        layers: int = CLASSIFIER_LAYERS,
    ):
        super().__init__()
        self.num_beams = int(num_beams)

        blocks: list = []
        width = int(in_dim)
        for _ in range(int(layers)):
            blocks.append(nn.Linear(width, int(hidden)))
            blocks.append(nn.ReLU())
            width = int(hidden)
        blocks.append(nn.Linear(width, self.num_beams))
        self.net = nn.Sequential(*blocks)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """``(B, 2)`` normalized positions -> ``(B, num_beams)`` logits."""
        return self.net(positions)


def _cache_signature(
    source_path: str, num_train: int, num_beams: int, epochs: int
) -> str:
    return (
        f"morais-icc2023|{os.path.abspath(source_path)}|{num_train}|"
        f"{num_beams}|{epochs}|{CLASSIFIER_BATCH}|{CLASSIFIER_LR}|{POSITION_BINS}"
    )


def train_beam_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_beams: int,
    device: torch.device,
    cache_dir: Optional[str] = None,
    source_path: str = "",
    epochs: int = CLASSIFIER_EPOCHS,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[BeamClassifier, Dict[str, float]]:
    """Train the classifier on TRAIN only, selecting on a held-out 25% split.

    The original TRAIN split is divided 75/25 into fit/validation; after the
    full schedule the parameters from the epoch with the highest validation
    top-1 accuracy are restored.  Nothing from the test split is touched.
    """
    signature = _cache_signature(
        source_path, int(train_features.shape[0]), int(num_beams), int(epochs)
    )
    cache_path = os.path.join(cache_dir, CLASSIFIER_NAME) if cache_dir else ""

    model = BeamClassifier(num_beams=num_beams, in_dim=int(train_features.shape[1]))
    model = model.to(device)

    if cache_path and os.path.isfile(cache_path):
        payload = torch.load(cache_path, map_location=device, weights_only=False)
        if payload.get("signature") == signature:
            model.load_state_dict(payload["state_dict"])
            model.eval()
            if verbose:
                print(
                    f"[baseline_models] restored beam classifier from "
                    f"{cache_path} (val-selected epoch "
                    f"{int(payload['stats']['best_epoch'])}, val top-1 "
                    f"{payload['stats']['best_val_top1']:.4f})"
                )
            return model, dict(payload["stats"])
        if verbose:
            print(
                "[baseline_models] cached beam classifier does not match this "
                "dataset/config; retraining."
            )

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    features = train_features.to(device)
    labels = train_labels.to(device)
    count = int(features.shape[0])

    # 75/25 fit/validation split of the TRAIN set, drawn once with a fixed seed.
    shuffled = torch.randperm(count, generator=generator)
    num_validation = max(1, int(round(CLASSIFIER_VAL_FRACTION * count)))
    validation_index = shuffled[:num_validation].to(device)
    fit_index = shuffled[num_validation:].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(CLASSIFIER_MILESTONES), gamma=CLASSIFIER_GAMMA
    )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_top1 = -1.0
    best_epoch = 0
    fit_count = int(fit_index.numel())

    for epoch in range(1, int(epochs) + 1):
        model.train()
        permutation = fit_index[
            torch.randperm(fit_count, generator=generator).to(device)
        ]
        total_loss = 0.0
        for start in range(0, fit_count, CLASSIFIER_BATCH):
            batch = permutation[start : start + CLASSIFIER_BATCH]
            loss = F.cross_entropy(model(features[batch]), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch.numel())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            predicted = model(features[validation_index]).argmax(dim=1)
            val_top1 = float(
                (predicted == labels[validation_index]).float().mean().item()
            )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"[baseline_models]   epoch {epoch:>3d}/{epochs}  "
                f"loss {total_loss / max(fit_count, 1):.4f}  "
                f"val top-1 {val_top1:.4f}  (best {best_val_top1:.4f} "
                f"@ epoch {best_epoch})"
            )

    model.load_state_dict(best_state)
    model.eval()

    stats = {
        "best_epoch": float(best_epoch),
        "best_val_top1": float(best_val_top1),
        "num_fit": float(fit_count),
        "num_validation": float(num_validation),
    }

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {"signature": signature, "state_dict": model.state_dict(),
             "stats": stats},
            cache_path,
        )
        if verbose:
            print(f"[baseline_models] cached beam classifier to {cache_path}")

    return model, stats


@torch.no_grad()
def classifier_orders(
    model: BeamClassifier,
    test_features: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Transmit-beam ordering by descending predicted class probability."""
    model.eval()
    chunks = []
    for start in range(0, test_features.shape[0], batch_size):
        logits = model(test_features[start : start + batch_size])
        chunks.append(torch.argsort(logits, dim=1, descending=True))
    return torch.cat(chunks, dim=0)


def topk_beam_accuracy(
    ordering: torch.Tensor, true_best_beam: torch.Tensor, k_values=(1, 4)
) -> Dict[int, float]:
    """Fraction of locations whose true best beam is inside the top-k picks."""
    accuracy: Dict[int, float] = {}
    for k in k_values:
        k_eff = min(int(k), int(ordering.shape[1]))
        hit = (ordering[:, :k_eff] == true_best_beam.unsqueeze(1)).any(dim=1)
        accuracy[int(k)] = float(hit.float().mean().item())
    return accuracy
