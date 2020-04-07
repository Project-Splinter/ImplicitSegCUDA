#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> upsample2x_bilinear2d_cuda_forward(
    const torch::Tensor& input, const float balance_value);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> upsample2x_bilinear2d_forward(
    const torch::Tensor& input, const float balance_value) {
      CHECK_INPUT(input);

      return upsample2x_bilinear2d_cuda_forward(input, balance_value);
}



// // std::vector<torch::Tensor> softras_render_cuda_backward(
// //     torch::Tensor grad_alpha2d,
// //     torch::Tensor alpha3d,
// //     torch::Tensor alpha2d);

// // C++ interface


// std::vector<torch::Tensor> softras_render_forward(
//     torch::Tensor alpha3d) {
//   CHECK_INPUT(alpha3d);

//   return softras_render_cuda_forward(alpha3d);
// }

// // std::vector<torch::Tensor> softras_render_backward(
// //     torch::Tensor grad_alpha2d,
// //     torch::Tensor alpha3d,
// //     torch::Tensor alpha2d) {
// //   CHECK_INPUT(grad_alpha2d);
// //   CHECK_INPUT(alpha3d);
// //   CHECK_INPUT(alpha2d);

// //   return softras_render_cuda_backward(grad_alpha2d, alpha3d, alpha2d);
// // }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &softras_render_forward, "forward (CUDA)");
//   // m.def("backward", &softras_render_backward, "backward (CUDA)");
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &upsample2x_bilinear2d_forward, "forward (CUDA)");
}