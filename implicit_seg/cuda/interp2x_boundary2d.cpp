#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> interp2x_boundary2d_cuda_forward(
    const torch::Tensor& input, const float balance_value);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> interp2x_boundary2d_forward(
    const torch::Tensor& input, const float balance_value) {
      CHECK_INPUT(input);

      return interp2x_boundary2d_cuda_forward(input, balance_value);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &interp2x_boundary2d_forward, "forward (CUDA)");
}