from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='implicit_seg',
    ext_modules=[
        CUDAExtension('implicit_seg.cuda.UpSampleBilinear2d_cuda', [
            './implicit_seg/cuda/UpSampleBilinear2d_cuda.cpp',
            './implicit_seg/cuda/UpSampleBilinear2d_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })