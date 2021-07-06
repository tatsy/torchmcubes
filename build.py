from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension

def build(setup_kwargs):
    if torch.cuda.is_available():
        from torch.utils.cpp_extension import CUDAExtension

        setup_kwargs.update({
            'ext_modules': [
                CUDAExtension(
                    'mcubes_module',
                    [
                        'cxx/pscan.cu',
                        'cxx/mcubes.cpp',
                        'cxx/mcubes_cpu.cpp',
                        'cxx/mcubes_cuda.cu',
                        'cxx/grid_interp_cpu.cpp',
                        'cxx/grid_interp_cuda.cu',
                    ],
                    extra_compile_args=['-DWITH_CUDA'],
                )
            ],
            'cmdclass': {'build_ext': BuildExtension}})

    else:
        print('CUDA environment is unavailable!')
        print('Build only CPU module!')

        from torch.utils.cpp_extension import CppExtension

        setup_kwargs.update({
          'ext_modules': [
              CppExtension('mcubes_module', [
                  'cxx/mcubes.cpp',
                  'cxx/mcubes_cpu.cpp',
                  'cxx/grid_interp_cpu.cpp',
              ])
          ],
          'cmdclass': {'build_ext': BuildExtension}})
