from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='implicit_seg',
    ext_modules=[
        CUDAExtension('implicit_seg.cuda.interp2x_boundary2d', [
            './implicit_seg/cuda/interp2x_boundary2d.cpp',
            './implicit_seg/cuda/interp2x_boundary2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })