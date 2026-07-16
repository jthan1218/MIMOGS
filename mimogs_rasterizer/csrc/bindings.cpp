#include <torch/extension.h>

#include <vector>

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
    double weight_floor);

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
    torch::Tensor tx_weights);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOAT(x)

std::vector<torch::Tensor> forward(
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
  CHECK_INPUT(rx_uv);
  CHECK_INPUT(rx_precision);
  CHECK_INPUT(tx_uv);
  CHECK_INPUT(tx_precision);
  CHECK_INPUT(gain);
  CHECK_INPUT(rx_centers);
  CHECK_INPUT(tx_centers);
  return beam_splat_forward_cuda(
      rx_uv, rx_precision, tx_uv, tx_precision, gain, rx_centers,
      tx_centers, k_rx, k_tx, weight_floor);
}

std::vector<torch::Tensor> backward(
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
  CHECK_INPUT(grad_output);
  CHECK_INPUT(rx_uv);
  CHECK_INPUT(rx_precision);
  CHECK_INPUT(tx_uv);
  CHECK_INPUT(tx_precision);
  CHECK_INPUT(gain);
  CHECK_INPUT(rx_centers);
  CHECK_INPUT(tx_centers);
  CHECK_CUDA(rx_indices);
  CHECK_CONTIGUOUS(rx_indices);
  TORCH_CHECK(rx_indices.scalar_type() == at::kInt, "rx_indices must be int32");
  CHECK_INPUT(rx_weights);
  CHECK_CUDA(tx_indices);
  CHECK_CONTIGUOUS(tx_indices);
  TORCH_CHECK(tx_indices.scalar_type() == at::kInt, "tx_indices must be int32");
  CHECK_INPUT(tx_weights);
  return beam_splat_backward_cuda(
      grad_output, rx_uv, rx_precision, tx_uv, tx_precision, gain,
      rx_centers, tx_centers, rx_indices, rx_weights, tx_indices,
      tx_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "MIMO-GS sparse beam splat forward (CUDA)");
  m.def("backward", &backward, "MIMO-GS sparse beam splat backward (CUDA)");
}
