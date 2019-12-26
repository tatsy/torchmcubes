from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mcubes_module',
    ext_modules=[
        CUDAExtension('mcubes_module', [
            'cxx/pscan.cu',
            'cxx/mcubes.cpp',
            'cxx/mcubes_cpu.cpp',
            'cxx/mcubes_cuda.cu',
            'cxx/grid_interp_cpu.cpp',
            'cxx/grid_interp_cuda.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
