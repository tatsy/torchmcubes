from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

try:
    from torch.utils.cpp_extension import CUDAExtension

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
            ],
            extra_compile_args=['-DWITH_CUDA'])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
except:
    print('CUDA environment was not successfully loaded!')
    print('Build only CPU module!')

    from torch.utils.cpp_extension import CppExtension

    setup(
        name='mcubes_module',
        ext_modules=[
            CppExtension('mcubes_module', [
                'cxx/mcubes.cpp',
                'cxx/mcubes_cpu.cpp',
                'cxx/grid_interp_cpu.cpp',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
