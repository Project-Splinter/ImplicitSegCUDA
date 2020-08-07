from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

INSTALL_REQUIREMENTS = ['torch', 'torchvision', 'matplotlib']

ext_modules=[
    CUDAExtension('implicit_seg.cuda.interp2x_boundary2d', [
        'implicit_seg/cuda/interp2x_boundary2d.cpp',
        'implicit_seg/cuda/interp2x_boundary2d_kernel.cu',
    ]),

    CUDAExtension('implicit_seg.cuda.interp2x_boundary3d', [
        'implicit_seg/cuda/interp2x_boundary3d.cpp',
        'implicit_seg/cuda/interp2x_boundary3d_kernel.cu',
    ]),
]

setup(
    name='implicit_seg',
    url='https://github.com/Project-Splinter/ImplicitSegCUDA',
    description='A Pytorch Segmentation module through implicit way (support 2d and 3d)', 
    version='0.0.2',
    author='Ruilong Li',
    author_email='ruilongl@usc.edu',    
    license='MIT License',
    packages=['implicit_seg', 'implicit_seg.cuda', 'implicit_seg.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)