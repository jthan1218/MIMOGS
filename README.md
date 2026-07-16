MIMO-GS
MIMO channel rendering: A 3D-Gaussian Splatting Approach

## Files

- `gaussian_renderer/fast_renderer.py`: batched projection and rendering.
- `mimogs_rasterizer/reference.py`: differentiable PyTorch reference/fallback.
- `mimogs_rasterizer/csrc/`: optional fused CUDA forward/backward operator.
- `scene/gaussian_model.py`: checkpoint-compatible batched gain evaluation.
- `train_fast.py`: batched training entry point.
- `benchmark_fast_renderer.py`: forward and training-step benchmark.
- `tests/test_fast_renderer.py`: equivalence tests against the original path.

## Installation

Create the original environment first.  Building a custom CUDA operator also
requires a CUDA compiler (`nvcc`) compatible with the installed PyTorch CUDA
runtime.  From the project root:

```bash
conda env create -f environment.yml
conda activate mimogs

# Optional but recommended for the fused CUDA path.
python -m pip install ninja pytest
python -m pip install -v ./mimogs_rasterizer
```

If the extension is not built or cannot be imported, `render_fast` falls back
to the PyTorch sparse implementation automatically.

<!-- ## Benchmark

```bash
python benchmark_fast_renderer.py \
  --gaussians 25000 \
  --batch 8 \
  --warmup 20 \
  --repeats 100 \
  --backward
```

Benchmark the following separately:

- original one-query loop;
- optimized PyTorch sparse renderer;
- selected path with the fused CUDA extension, when available.

Use the same GPU, batch size, Gaussian count, and beam dimensions as the paper.
Report both latency per batch and throughput in locations/s.  CUDA timings must
be taken after synchronization, as done in the benchmark script. -->

## Training

```bash
python train_fast.py
```

Some code snippets are borrowed from [WRF-GS](https://github.com/wenchaozheng/WRF-GS).
