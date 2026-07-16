#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxTopK = 8;

__device__ __forceinline__ float clamp_logit(float x) {
  return fminf(0.0f, fmaxf(-80.0f, x));
}

__device__ __forceinline__ void insert_topk(
    float value, int index, float* values, int* indices, int k) {
  int min_slot = 0;
  float min_value = values[0];
#pragma unroll
  for (int i = 1; i < kMaxTopK; ++i) {
    if (i >= k) break;
    if (values[i] < min_value) {
      min_value = values[i];
      min_slot = i;
    }
  }
  if (value > min_value) {
    values[min_slot] = value;
    indices[min_slot] = index;
  }
}

__global__ void topk_tx_kernel(
    const float* __restrict__ uv,
    const float* __restrict__ precision,
    const float* __restrict__ centers,
    int n_gaussians,
    int n_beams,
    int k,
    float weight_floor,
    int* __restrict__ out_indices,
    float* __restrict__ out_weights) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_gaussians) return;

  float top_values[kMaxTopK];
  int top_indices[kMaxTopK];
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    top_values[i] = -FLT_MAX;
    top_indices[i] = 0;
  }

  const float ux = uv[2 * n + 0];
  const float uy = uv[2 * n + 1];
  const float p00 = precision[3 * n + 0];
  const float p01 = precision[3 * n + 1];
  const float p11 = precision[3 * n + 2];

  for (int b = 0; b < n_beams; ++b) {
    const float dx = centers[2 * b + 0] - ux;
    const float dy = centers[2 * b + 1] - uy;
    const float mahal = p00 * dx * dx + 2.0f * p01 * dx * dy + p11 * dy * dy;
    insert_topk(clamp_logit(-0.5f * mahal), b, top_values, top_indices, k);
  }

  float denom = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k) break;
    float w = expf(top_values[i]);
    if (weight_floor > 0.0f && w < weight_floor) w = 0.0f;
    top_values[i] = w;
    denom += w;
  }
  denom = fmaxf(denom, 1.0e-12f);

#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k) break;
    out_indices[n * k + i] = top_indices[i];
    out_weights[n * k + i] = top_values[i] / denom;
  }
}

__global__ void topk_rx_kernel(
    const float* __restrict__ uv,
    const float* __restrict__ precision,
    const float* __restrict__ centers,
    int batch_size,
    int n_gaussians,
    int n_beams,
    int k,
    float weight_floor,
    int* __restrict__ out_indices,
    float* __restrict__ out_weights) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * n_gaussians;
  if (linear >= total) return;

  float top_values[kMaxTopK];
  int top_indices[kMaxTopK];
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    top_values[i] = -FLT_MAX;
    top_indices[i] = 0;
  }

  const float ux = uv[2 * linear + 0];
  const float uy = uv[2 * linear + 1];
  const float p00 = precision[3 * linear + 0];
  const float p01 = precision[3 * linear + 1];
  const float p11 = precision[3 * linear + 2];

  for (int b = 0; b < n_beams; ++b) {
    const float dx = centers[2 * b + 0] - ux;
    const float dy = centers[2 * b + 1] - uy;
    const float mahal = p00 * dx * dx + 2.0f * p01 * dx * dy + p11 * dy * dy;
    insert_topk(clamp_logit(-0.5f * mahal), b, top_values, top_indices, k);
  }

  float denom = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k) break;
    float w = expf(top_values[i]);
    if (weight_floor > 0.0f && w < weight_floor) w = 0.0f;
    top_values[i] = w;
    denom += w;
  }
  denom = fmaxf(denom, 1.0e-12f);

#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k) break;
    out_indices[linear * k + i] = top_indices[i];
    out_weights[linear * k + i] = top_values[i] / denom;
  }
}

// Beam-pair maps are small (4x16 in the current experiments), while the
// Gaussian count is large.  A direct global atomicAdd for every Gaussian and
// retained beam pair therefore creates severe contention on only a few output
// cells.  This block-reduced kernel first accumulates one Gaussian chunk into
// shared memory and issues only one global atomicAdd per output cell and block.
__global__ void splat_block_reduced_kernel(
    const int* __restrict__ rx_indices,
    const float* __restrict__ rx_weights,
    const int* __restrict__ tx_indices,
    const float* __restrict__ tx_weights,
    const float* __restrict__ gain,
    int n_gaussians,
    int n_rx_beams,
    int n_tx_beams,
    int k_rx,
    int k_tx,
    int chunks_per_batch,
    float* __restrict__ output) {
  extern __shared__ float local_map[];
  const int n_pairs = n_rx_beams * n_tx_beams;
  const int batch = blockIdx.x / chunks_per_batch;
  const int chunk = blockIdx.x - batch * chunks_per_batch;

  for (int pair = threadIdx.x; pair < n_pairs; pair += blockDim.x) {
    local_map[pair] = 0.0f;
  }
  __syncthreads();

  const int n = chunk * blockDim.x + threadIdx.x;
  if (n < n_gaussians) {
    const int linear = batch * n_gaussians + n;
    const float g = gain[linear];
#pragma unroll
    for (int i = 0; i < kMaxTopK; ++i) {
      if (i >= k_rx) break;
      const int ri = rx_indices[linear * k_rx + i];
      const float rw = rx_weights[linear * k_rx + i];
#pragma unroll
      for (int j = 0; j < kMaxTopK; ++j) {
        if (j >= k_tx) break;
        const int tj = tx_indices[n * k_tx + j];
        const float tw = tx_weights[n * k_tx + j];
        atomicAdd(&local_map[ri * n_tx_beams + tj], g * rw * tw);
      }
    }
  }
  __syncthreads();

  for (int pair = threadIdx.x; pair < n_pairs; pair += blockDim.x) {
    atomicAdd(&output[batch * n_pairs + pair], local_map[pair]);
  }
}

// Generic fallback for unusually large beam grids that do not fit into the
// configured shared-memory budget.
__global__ void splat_global_atomic_kernel(
    const int* __restrict__ rx_indices,
    const float* __restrict__ rx_weights,
    const int* __restrict__ tx_indices,
    const float* __restrict__ tx_weights,
    const float* __restrict__ gain,
    int batch_size,
    int n_gaussians,
    int n_rx_beams,
    int n_tx_beams,
    int k_rx,
    int k_tx,
    float* __restrict__ output) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * n_gaussians;
  if (linear >= total) return;

  const int n = linear % n_gaussians;
  const int batch = linear / n_gaussians;
  const float g = gain[linear];
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k_rx) break;
    const int ri = rx_indices[linear * k_rx + i];
    const float rw = rx_weights[linear * k_rx + i];
#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) {
      if (j >= k_tx) break;
      const int tj = tx_indices[n * k_tx + j];
      const float tw = tx_weights[n * k_tx + j];
      atomicAdd(&output[(batch * n_rx_beams + ri) * n_tx_beams + tj], g * rw * tw);
    }
  }
}

__global__ void backward_rx_gain_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ rx_uv,
    const float* __restrict__ rx_precision,
    const float* __restrict__ gain,
    const float* __restrict__ rx_centers,
    const int* __restrict__ rx_indices,
    const float* __restrict__ rx_weights,
    const int* __restrict__ tx_indices,
    const float* __restrict__ tx_weights,
    int batch_size,
    int n_gaussians,
    int n_rx_beams,
    int n_tx_beams,
    int k_rx,
    int k_tx,
    float* __restrict__ grad_rx_uv,
    float* __restrict__ grad_rx_precision,
    float* __restrict__ grad_gain) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * n_gaussians;
  if (linear >= total) return;

  const int n = linear % n_gaussians;
  const int batch = linear / n_gaussians;
  const float g = gain[linear];

  float grad_w[kMaxTopK];
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) grad_w[i] = 0.0f;
  float grad_g = 0.0f;

#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k_rx) break;
    const int ri = rx_indices[linear * k_rx + i];
    const float rw = rx_weights[linear * k_rx + i];
#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) {
      if (j >= k_tx) break;
      const int tj = tx_indices[n * k_tx + j];
      const float tw = tx_weights[n * k_tx + j];
      const float go = grad_output[(batch * n_rx_beams + ri) * n_tx_beams + tj];
      grad_g += go * rw * tw;
      grad_w[i] += g * go * tw;
    }
  }
  grad_gain[linear] = grad_g;

  float weighted_grad = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k_rx) break;
    weighted_grad += grad_w[i] * rx_weights[linear * k_rx + i];
  }

  const float ux = rx_uv[2 * linear + 0];
  const float uy = rx_uv[2 * linear + 1];
  const float p00 = rx_precision[3 * linear + 0];
  const float p01 = rx_precision[3 * linear + 1];
  const float p11 = rx_precision[3 * linear + 2];

  float gux = 0.0f, guy = 0.0f;
  float gp00 = 0.0f, gp01 = 0.0f, gp11 = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i >= k_rx) break;
    const float w = rx_weights[linear * k_rx + i];
    const int ri = rx_indices[linear * k_rx + i];
    const float dx = rx_centers[2 * ri + 0] - ux;
    const float dy = rx_centers[2 * ri + 1] - uy;
    const float mahal = p00 * dx * dx + 2.0f * p01 * dx * dy + p11 * dy * dy;
    const float raw_logit = -0.5f * mahal;
    float dlogit = w * (grad_w[i] - weighted_grad);
    if (raw_logit <= -80.0f) dlogit = 0.0f;

    gux += dlogit * (p00 * dx + p01 * dy);
    guy += dlogit * (p01 * dx + p11 * dy);
    gp00 += dlogit * (-0.5f * dx * dx);
    gp01 += dlogit * (-dx * dy);
    gp11 += dlogit * (-0.5f * dy * dy);
  }

  grad_rx_uv[2 * linear + 0] = gux;
  grad_rx_uv[2 * linear + 1] = guy;
  grad_rx_precision[3 * linear + 0] = gp00;
  grad_rx_precision[3 * linear + 1] = gp01;
  grad_rx_precision[3 * linear + 2] = gp11;
}

__global__ void backward_tx_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ tx_uv,
    const float* __restrict__ tx_precision,
    const float* __restrict__ gain,
    const float* __restrict__ tx_centers,
    const int* __restrict__ rx_indices,
    const float* __restrict__ rx_weights,
    const int* __restrict__ tx_indices,
    const float* __restrict__ tx_weights,
    int batch_size,
    int n_gaussians,
    int n_rx_beams,
    int n_tx_beams,
    int k_rx,
    int k_tx,
    float* __restrict__ grad_tx_uv,
    float* __restrict__ grad_tx_precision) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_gaussians) return;

  const float ux = tx_uv[2 * n + 0];
  const float uy = tx_uv[2 * n + 1];
  const float p00 = tx_precision[3 * n + 0];
  const float p01 = tx_precision[3 * n + 1];
  const float p11 = tx_precision[3 * n + 2];

  float gux = 0.0f, guy = 0.0f;
  float gp00 = 0.0f, gp01 = 0.0f, gp11 = 0.0f;

  for (int batch = 0; batch < batch_size; ++batch) {
    const int linear = batch * n_gaussians + n;
    const float g = gain[linear];
    float grad_w[kMaxTopK];
#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) grad_w[j] = 0.0f;

#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) {
      if (j >= k_tx) break;
      const int tj = tx_indices[n * k_tx + j];
#pragma unroll
      for (int i = 0; i < kMaxTopK; ++i) {
        if (i >= k_rx) break;
        const int ri = rx_indices[linear * k_rx + i];
        const float rw = rx_weights[linear * k_rx + i];
        const float go = grad_output[(batch * n_rx_beams + ri) * n_tx_beams + tj];
        grad_w[j] += g * go * rw;
      }
    }

    float weighted_grad = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) {
      if (j >= k_tx) break;
      weighted_grad += grad_w[j] * tx_weights[n * k_tx + j];
    }

#pragma unroll
    for (int j = 0; j < kMaxTopK; ++j) {
      if (j >= k_tx) break;
      const float w = tx_weights[n * k_tx + j];
      const int tj = tx_indices[n * k_tx + j];
      const float dx = tx_centers[2 * tj + 0] - ux;
      const float dy = tx_centers[2 * tj + 1] - uy;
      const float mahal = p00 * dx * dx + 2.0f * p01 * dx * dy + p11 * dy * dy;
      const float raw_logit = -0.5f * mahal;
      float dlogit = w * (grad_w[j] - weighted_grad);
      if (raw_logit <= -80.0f) dlogit = 0.0f;

      gux += dlogit * (p00 * dx + p01 * dy);
      guy += dlogit * (p01 * dx + p11 * dy);
      gp00 += dlogit * (-0.5f * dx * dx);
      gp01 += dlogit * (-dx * dy);
      gp11 += dlogit * (-0.5f * dy * dy);
    }
  }

  grad_tx_uv[2 * n + 0] = gux;
  grad_tx_uv[2 * n + 1] = guy;
  grad_tx_precision[3 * n + 0] = gp00;
  grad_tx_precision[3 * n + 1] = gp01;
  grad_tx_precision[3 * n + 2] = gp11;
}

}  // namespace

std::vector<torch::Tensor> beam_splat_forward_cuda(
    torch::Tensor rx_uv,
    torch::Tensor rx_precision,
    torch::Tensor tx_uv,
    torch::Tensor tx_precision,
    torch::Tensor gain,
    torch::Tensor rx_centers,
    torch::Tensor tx_centers,
    int64_t k_rx,
    int64_t k_tx,
    double weight_floor) {
  TORCH_CHECK(rx_uv.dim() == 3 && rx_uv.size(2) == 2, "rx_uv must be (B,N,2)");
  TORCH_CHECK(rx_precision.dim() == 3 &&
              rx_precision.size(0) == rx_uv.size(0) &&
              rx_precision.size(1) == rx_uv.size(1) &&
              rx_precision.size(2) == 3,
              "rx_precision must be (B,N,3)");
  TORCH_CHECK(tx_uv.dim() == 2 && tx_uv.size(1) == 2, "tx_uv must be (N,2)");
  TORCH_CHECK(tx_precision.dim() == 2 &&
              tx_precision.size(0) == tx_uv.size(0) &&
              tx_precision.size(1) == 3,
              "tx_precision must be (N,3)");
  TORCH_CHECK(gain.dim() == 2 &&
              gain.size(0) == rx_uv.size(0) &&
              gain.size(1) == rx_uv.size(1),
              "gain must be (B,N)");
  TORCH_CHECK(tx_uv.size(0) == rx_uv.size(1), "Rx/Tx Gaussian counts differ");
  TORCH_CHECK(rx_centers.dim() == 2 && rx_centers.size(1) == 2, "rx_centers must be (R,2)");
  TORCH_CHECK(tx_centers.dim() == 2 && tx_centers.size(1) == 2, "tx_centers must be (T,2)");
  TORCH_CHECK(k_rx > 0 && k_rx <= kMaxTopK && k_rx <= rx_centers.size(0), "invalid k_rx");
  TORCH_CHECK(k_tx > 0 && k_tx <= kMaxTopK && k_tx <= tx_centers.size(0), "invalid k_tx");

  const int B = static_cast<int>(rx_uv.size(0));
  const int N = static_cast<int>(rx_uv.size(1));
  const int R = static_cast<int>(rx_centers.size(0));
  const int T = static_cast<int>(tx_centers.size(0));
  const int KR = static_cast<int>(k_rx);
  const int KT = static_cast<int>(k_tx);

  auto float_opts = rx_uv.options();
  auto int_opts = rx_uv.options().dtype(torch::kInt32);
  auto output = torch::zeros({B, R, T}, float_opts);
  auto rx_indices = torch::empty({B, N, KR}, int_opts);
  auto rx_weights = torch::empty({B, N, KR}, float_opts);
  auto tx_indices = torch::empty({N, KT}, int_opts);
  auto tx_weights = torch::empty({N, KT}, float_opts);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  topk_tx_kernel<<<(N + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      tx_uv.data_ptr<float>(), tx_precision.data_ptr<float>(),
      tx_centers.data_ptr<float>(), N, T, KT, static_cast<float>(weight_floor),
      tx_indices.data_ptr<int>(), tx_weights.data_ptr<float>());
  topk_rx_kernel<<<(B * N + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      rx_uv.data_ptr<float>(), rx_precision.data_ptr<float>(),
      rx_centers.data_ptr<float>(), B, N, R, KR, static_cast<float>(weight_floor),
      rx_indices.data_ptr<int>(), rx_weights.data_ptr<float>());
  const int n_pairs = R * T;
  const int chunks_per_batch = (N + kThreads - 1) / kThreads;
  // 16 KiB comfortably covers the intended 4x16 and common larger beam maps.
  // The generic path remains available for very large grids.
  constexpr int kSharedPairLimit = 4096;
  if (n_pairs <= kSharedPairLimit) {
    const size_t shared_bytes = static_cast<size_t>(n_pairs) * sizeof(float);
    splat_block_reduced_kernel<<<B * chunks_per_batch, kThreads, shared_bytes, stream>>>(
        rx_indices.data_ptr<int>(), rx_weights.data_ptr<float>(),
        tx_indices.data_ptr<int>(), tx_weights.data_ptr<float>(),
        gain.data_ptr<float>(), N, R, T, KR, KT, chunks_per_batch,
        output.data_ptr<float>());
  } else {
    splat_global_atomic_kernel<<<(B * N + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        rx_indices.data_ptr<int>(), rx_weights.data_ptr<float>(),
        tx_indices.data_ptr<int>(), tx_weights.data_ptr<float>(),
        gain.data_ptr<float>(), B, N, R, T, KR, KT, output.data_ptr<float>());
  }

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "MIMO-GS CUDA forward launch failed: ",
              cudaGetErrorString(err));
  return {output, rx_indices, rx_weights, tx_indices, tx_weights};
}

std::vector<torch::Tensor> beam_splat_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor rx_uv,
    torch::Tensor rx_precision,
    torch::Tensor tx_uv,
    torch::Tensor tx_precision,
    torch::Tensor gain,
    torch::Tensor rx_centers,
    torch::Tensor tx_centers,
    torch::Tensor rx_indices,
    torch::Tensor rx_weights,
    torch::Tensor tx_indices,
    torch::Tensor tx_weights) {
  const int B = static_cast<int>(rx_uv.size(0));
  const int N = static_cast<int>(rx_uv.size(1));
  const int R = static_cast<int>(rx_centers.size(0));
  const int T = static_cast<int>(tx_centers.size(0));
  const int KR = static_cast<int>(rx_indices.size(2));
  const int KT = static_cast<int>(tx_indices.size(1));

  auto grad_rx_uv = torch::zeros_like(rx_uv);
  auto grad_rx_precision = torch::zeros_like(rx_precision);
  auto grad_tx_uv = torch::zeros_like(tx_uv);
  auto grad_tx_precision = torch::zeros_like(tx_precision);
  auto grad_gain = torch::zeros_like(gain);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  backward_rx_gain_kernel<<<(B * N + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(), rx_uv.data_ptr<float>(),
      rx_precision.data_ptr<float>(), gain.data_ptr<float>(),
      rx_centers.data_ptr<float>(), rx_indices.data_ptr<int>(),
      rx_weights.data_ptr<float>(), tx_indices.data_ptr<int>(),
      tx_weights.data_ptr<float>(), B, N, R, T, KR, KT,
      grad_rx_uv.data_ptr<float>(), grad_rx_precision.data_ptr<float>(),
      grad_gain.data_ptr<float>());

  backward_tx_kernel<<<(N + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(), tx_uv.data_ptr<float>(),
      tx_precision.data_ptr<float>(), gain.data_ptr<float>(),
      tx_centers.data_ptr<float>(), rx_indices.data_ptr<int>(),
      rx_weights.data_ptr<float>(), tx_indices.data_ptr<int>(),
      tx_weights.data_ptr<float>(), B, N, R, T, KR, KT,
      grad_tx_uv.data_ptr<float>(), grad_tx_precision.data_ptr<float>());

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "MIMO-GS CUDA backward launch failed: ",
              cudaGetErrorString(err));
  return {grad_rx_uv, grad_rx_precision, grad_tx_uv, grad_tx_precision, grad_gain};
}
