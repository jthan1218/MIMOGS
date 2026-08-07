"""Throwaway verification of the beam-coordinate mod-2 wrap fix.

Not part of the project -- untracked scratch, safe to delete.

Three checks:
  A. fused CUDA vs PyTorch reference, with primitive centres deliberately
     parked in the wrap-sensitive band (forward + all five gradients).
  B. explicit wrap behaviour: a centre just inside the positive edge must now
     put large weight on the beam at the opposite (negative) edge.
  C. interior no-op: a centre well inside the grid must be unaffected.

Run:  python verify_wrap.py
"""
from __future__ import annotations

import torch

from mimogs_rasterizer import beam_splat, cuda_extension_available
from mimogs_rasterizer.reference import topk_gaussian_beam_weights, wrap_beam_delta

DEV = "cuda"
RX_H = RX_V = 4          # 16 rx beams, matches the ASU dataset
TX_H = TX_V = 8          # 64 tx beams
K = 4                    # matches max_active_{rx,tx}_beams in run_args
FLOOR = 1e-4


def grid(h: int, v: int) -> torch.Tensor:
    """The renderer's beam grid: index b -> (u_bins[b % h], v_bins[b // h])."""
    u = 2.0 * torch.fft.fftfreq(h, d=1.0, device=DEV)
    w = 2.0 * torch.fft.fftfreq(v, d=1.0, device=DEV)
    return torch.stack([u.repeat(v), w.repeat_interleave(h)], dim=-1).contiguous()


def unwrapped_topk(uv, precision, centers, k):
    """The OLD convention: identical maths, raw difference. For contrast only."""
    dx = centers[:, 0] - uv[..., 0, None]
    dy = centers[:, 1] - uv[..., 1, None]
    p00, p01, p11 = precision[..., 0, None], precision[..., 1, None], precision[..., 2, None]
    logits = -0.5 * (p00 * dx.square() + 2.0 * p01 * dx * dy + p11 * dy.square())
    top, idx = torch.topk(logits, k=k, dim=-1, largest=True, sorted=True)
    w = torch.exp(top - top.amax(dim=-1, keepdim=True))
    return w / w.sum(dim=-1, keepdim=True), idx


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


# ----------------------------------------------------------------------
banner("A. fused CUDA vs reference -- centres parked near the grid edges")
# ----------------------------------------------------------------------
print(f"cuda extension available: {cuda_extension_available()}")

rx_centers, tx_centers = grid(RX_H, RX_V), grid(TX_H, TX_V)

# u values chosen to straddle the wrap boundary (grid half-bin = 1/4 = 0.25,
# so anything above 0.75 is wrap-sensitive on the 4x4 rx array).
edge_u = [0.9, 0.95, -0.95, 0.99, 0.76, 0.5, 0.1, -0.3]
edge_v = [0.95, -0.9, 0.9, -0.99, 0.2, -0.76, -0.1, 0.4]
N = len(edge_u)
B = 3

rx_uv = torch.tensor([[[u, v] for u, v in zip(edge_u, edge_v)]], device=DEV)
rx_uv = rx_uv.repeat(B, 1, 1).contiguous()
# de-correlate the batch entries a little so the batch axis is exercised
rx_uv = (rx_uv + torch.linspace(-0.02, 0.02, B, device=DEV)[:, None, None]).contiguous()
tx_uv = torch.tensor([[v, u] for u, v in zip(edge_u, edge_v)], device=DEV).contiguous()

torch.manual_seed(0)


def precision(shape):
    a = torch.rand(shape, device=DEV) * 60 + 20
    b = torch.rand(shape, device=DEV) * 60 + 20
    off = (torch.rand(shape, device=DEV) - 0.5) * 2 * torch.sqrt(a * b) * 0.5
    return torch.stack([a, off, b], dim=-1).contiguous()


rx_prec, tx_prec = precision((B, N)), precision((N,))
gain = (torch.rand(B, N, device=DEV) + 0.2).contiguous()

tensors = [rx_uv, rx_prec, tx_uv, tx_prec, gain]
for t in tensors:
    t.requires_grad_(True)

loss_w = torch.randn(B, RX_H * RX_V, TX_H * TX_V, device=DEV)

out_cuda = beam_splat(rx_uv, rx_prec, tx_uv, tx_prec, gain, rx_centers, tx_centers,
                      K, K, FLOOR, use_cuda_extension=True)
g_cuda = torch.autograd.grad((out_cuda * loss_w).sum(), tensors)

out_ref = beam_splat(rx_uv, rx_prec, tx_uv, tx_prec, gain, rx_centers, tx_centers,
                     K, K, FLOOR, use_cuda_extension=False)
g_ref = torch.autograd.grad((out_ref * loss_w).sum(), tensors)

rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-20)).item()
print(f"  shapes: rx_uv {tuple(rx_uv.shape)}, tx_uv {tuple(tx_uv.shape)}, out {tuple(out_cuda.shape)}")
print(f"  forward            max rel err  {rel(out_cuda, out_ref):.3e}")
for name, a, b in zip(["rx_uv", "rx_precision", "tx_uv", "tx_precision", "gain"], g_cuda, g_ref):
    print(f"  grad {name:13s} max rel err  {rel(a, b):.3e}")

# ----------------------------------------------------------------------
banner("B. wrap behaviour -- centre just inside the positive edge")
# ----------------------------------------------------------------------
u_bins = (2.0 * torch.fft.fftfreq(RX_H, d=1.0, device=DEV)).tolist()
print(f"  rx u-bins ({RX_H} elements): {u_bins}   half-bin = {1.0/RX_H}")

probe_u, probe_v = 0.95, 0.0
probe = torch.tensor([[probe_u, probe_v]], device=DEV)
probe_prec = torch.tensor([[80.0, 0.0, 80.0]], device=DEV)

print(f"\n  probe centre u = {probe_u}, v = {probe_v}")
print(f"    raw     distance to bin -1.00 : {abs(-1.0 - probe_u):.3f}")
print(f"    wrapped distance to bin -1.00 : "
      f"{wrap_beam_delta(torch.tensor([-1.0 - probe_u])).abs().item():.3f}")
print(f"    raw/wrapped distance to bin 0.50 : {abs(0.5 - probe_u):.3f}")

w_new, i_new = topk_gaussian_beam_weights(probe, probe_prec, rx_centers, K, 0.0)
w_old, i_old = unwrapped_topk(probe, probe_prec, rx_centers, K)

order = torch.argsort(w_new[0], descending=True)
print(f"\n  WRAPPED (current code) top-{K}:")
for s in order.tolist():
    b = int(i_new[0, s])
    print(f"    beam {b:3d}  centre u={rx_centers[b,0]:+.2f} v={rx_centers[b,1]:+.2f}   "
          f"weight {w_new[0, s].item():.6f}")
print(f"  UNWRAPPED (old convention) top-{K}:")
for s in range(K):
    b = int(i_old[0, s])
    print(f"    beam {b:3d}  centre u={rx_centers[b,0]:+.2f} v={rx_centers[b,1]:+.2f}   "
          f"weight {w_old[0, s].item():.6f}")

# the wrapped neighbour is the bin at u = -1.0, v = 0.0 -> index 2 on a 4x4 grid
wrapped_neighbour = int((( rx_centers[:, 0] == -1.0) & (rx_centers[:, 1] == 0.0)).nonzero()[0])
sel_new = i_new[0].tolist()
sel_old = i_old[0].tolist()
w_of = lambda w, i, b: (w[0][i[0].tolist().index(b)].item() if b in i[0].tolist() else 0.0)
print(f"\n  wrapped neighbour = beam {wrapped_neighbour} (u=-1.00, v=0.00)")
print(f"    selected by WRAPPED code   : {wrapped_neighbour in sel_new}   "
      f"weight {w_of(w_new, i_new, wrapped_neighbour):.6f}")
print(f"    selected by UNWRAPPED code : {wrapped_neighbour in sel_old}   "
      f"weight {w_of(w_old, i_old, wrapped_neighbour):.6f}")
print(f"  --> PASS" if w_of(w_new, i_new, wrapped_neighbour) > 0.1 else "  --> FAIL")

# same probe through the fused CUDA kernel, to prove the binary agrees
o = beam_splat(probe[None], probe_prec[None], torch.zeros(1, 2, device=DEV),
               torch.tensor([[80.0, 0.0, 80.0]], device=DEV), torch.ones(1, 1, device=DEV),
               rx_centers, tx_centers, 1, 1, 0.0, use_cuda_extension=True)
argmax_beam = int(o.sum(-1).argmax())
print(f"  fused CUDA argmax rx beam for this probe: {argmax_beam} "
      f"(u={rx_centers[argmax_beam,0]:+.2f}) --> "
      f"{'PASS' if argmax_beam == wrapped_neighbour else 'FAIL'}")

# ----------------------------------------------------------------------
banner("C. interior no-op -- wrap must not change anything away from the edges")
# ----------------------------------------------------------------------
inner = torch.tensor([[0.05, -0.10], [0.20, 0.15], [-0.25, 0.05], [0.0, 0.30]], device=DEV)
inner_prec = precision((inner.shape[0],))
w_wrapped, i_wrapped = topk_gaussian_beam_weights(inner, inner_prec, rx_centers, K, 0.0)
w_raw, i_raw = unwrapped_topk(inner, inner_prec, rx_centers, K)

# sort both by index so the comparison is order-independent
sw, si = torch.sort(i_wrapped, dim=-1)
sr, ri = torch.sort(i_raw, dim=-1)
same_idx = bool((sw == sr).all())
max_wdiff = (torch.gather(w_wrapped, -1, si) - torch.gather(w_raw, -1, ri)).abs().max().item()
# Restricted to the beams that are actually SELECTED: far-side beams do get
# their delta shifted by 2, but they are never in the top-k, so "no-op" means
# no shift on the retained beams.
sel_du = torch.gather(rx_centers[:, 0].expand(inner.shape[0], -1), -1, i_wrapped) - inner[..., 0, None]
sel_dv = torch.gather(rx_centers[:, 1].expand(inner.shape[0], -1), -1, i_wrapped) - inner[..., 1, None]
max_delta_shift = max(
    (wrap_beam_delta(sel_du) - sel_du).abs().max().item(),
    (wrap_beam_delta(sel_dv) - sel_dv).abs().max().item(),
)

print(f"  interior centres: {[[round(x,2) for x in r] for r in inner.tolist()]}")
print(f"  max |wrap(d) - d| over the SELECTED (centre, beam) deltas : {max_delta_shift:.3e}")
print(f"  identical top-{K} beam sets                      : {same_idx}")
print(f"  max weight difference                            : {max_wdiff:.3e}")
print(f"  --> {'PASS (wrap is a no-op in the interior)' if same_idx and max_wdiff < 1e-6 else 'FAIL'}")
print()
