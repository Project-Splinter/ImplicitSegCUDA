#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void upsample2x_bilinear2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4> input,
    torch::PackedTensorAccessor32<scalar_t,4> output,
    torch::PackedTensorAccessor32<bool,4> is_boundary,
    const float balance_value) {
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
        is_boundary[bi][ci][y][x] = false;
        return;
        
    }else if (skip_x){
        auto v1 = input[bi][ci][(y-1)/2][x/2];
        auto v2 = input[bi][ci][(y+1)/2][x/2];
        output[bi][ci][y][x] = (v1 + v2) / 2.;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        if (flag1 == flag2){is_boundary[bi][ci][y][x] = false;}
        else{is_boundary[bi][ci][y][x] = true;}
        return;

    }else if (skip_y){
        auto v1 = input[bi][ci][y/2][(x-1)/2];
        auto v2 = input[bi][ci][y/2][(x+1)/2];
        output[bi][ci][y][x] = (v1 + v2) / 2.;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        if (flag1 == flag2){is_boundary[bi][ci][y][x] = false;}
        else{is_boundary[bi][ci][y][x] = true;}
        return;

    }else{
        auto v1 = input[bi][ci][(y-1)/2][(x-1)/2];
        auto v2 = input[bi][ci][(y-1)/2][(x+1)/2]; 
        auto v3 = input[bi][ci][(y+1)/2][(x-1)/2]; 
        auto v4 = input[bi][ci][(y+1)/2][(x+1)/2];
        output[bi][ci][y][x] = (v1 + v2 + v3 + v4) / 4.0;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        bool flag3 = v3 > balance_value;
        bool flag4 = v4 > balance_value;
        if (flag1 == flag2 && flag2 == flag3 && flag3 == flag4){
            is_boundary[bi][ci][y][x] = false;
        }else{is_boundary[bi][ci][y][x] = true;}
        return;
    }
}

} // namespace

std::vector<torch::Tensor> upsample2x_bilinear2d_cuda_forward(
    const torch::Tensor& input, const float balance_value) {

    int bn = input.size(0);
    int c = input.size(1);
    int h = input.size(2) * 2 - 1;
    int w = input.size(3) * 2 - 1;

    auto output = torch::empty({bn, c, h, w}, input.type());
    auto is_boundary = torch::empty(
        {bn, c, h, w}, torch::ScalarType::Bool).to(input.device());

    const int num_kernels = bn * c * h * w;
    const int num_threads = 1024;
    const dim3 blocks((num_kernels + num_threads - 1) / num_threads);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "upsample2x_bilinear2d_cuda_forward", ([&] {
            upsample2x_bilinear2d_cuda_forward_kernel<scalar_t>
                <<<blocks, num_threads>>>(
                    input.packed_accessor32<scalar_t, 4>(), 
                    output.packed_accessor32<scalar_t, 4>(),
                    is_boundary.packed_accessor32<bool, 4>(),
                    balance_value);
    }));

    return {output, is_boundary};
}
