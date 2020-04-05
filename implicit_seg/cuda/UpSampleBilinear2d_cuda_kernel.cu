#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void upsample2x_bilinear2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4> input,
    torch::PackedTensorAccessor32<scalar_t,4> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int bn = output.size(0);
    const int c = output.size(1);
    const int h = output.size(2);
    const int w = output.size(3);

    if (i >= bn * c * h * w) {
        return;
    }
    
    const int x = i % w;
    const int y = (i / w) % h;
    const int ci = (i / (h * w)) % c;
    const int bi = i / (c * h * w);

    const bool skip_x = x % 2 == 0;
    const bool skip_y = y % 2 == 0; 

    if (skip_x && skip_y){
        output[bi][ci][y][x] = input[bi][ci][y/2][x/2];
        return;
    }else if (skip_x){
        output[bi][ci][y][x] = (
            input[bi][ci][(y-1)/2][x/2] + 
            input[bi][ci][(y+1)/2][x/2]
        ) / 2.;
        return;
    }else if (skip_y){
        output[bi][ci][y][x] = (
            input[bi][ci][y/2][(x-1)/2] + 
            input[bi][ci][y/2][(x+1)/2]
        ) / 2.;
        return;
    }else{
        output[bi][ci][y][x] = (
            input[bi][ci][(y-1)/2][(x-1)/2] + 
            input[bi][ci][(y-1)/2][(x+1)/2] + 
            input[bi][ci][(y+1)/2][(x-1)/2] + 
            input[bi][ci][(y+1)/2][(x+1)/2]
        ) / 4.;
        return;
    }
}

} // namespace

torch::Tensor upsample2x_bilinear2d_cuda_forward(
    const torch::Tensor& input) {

    int bn = input.size(0);
    int c = input.size(1);
    int h = input.size(2) * 2 - 1;
    int w = input.size(3) * 2 - 1;

    auto output = torch::empty({bn, c, h, w}, input.type());

    const int num_kernels = bn * c * h * w;
    const int num_threads = 1024;
    const dim3 blocks((num_kernels + num_threads - 1) / num_threads);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "upsample2x_bilinear2d_cuda_forward", ([&] {
            auto idata = input.packed_accessor32<scalar_t, 4>();
            auto odata = output.packed_accessor32<scalar_t, 4>();

            upsample2x_bilinear2d_cuda_forward_kernel<scalar_t><<<blocks, num_threads>>>(
                idata, odata);
    }));

    return output;
}
