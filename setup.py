from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

setup_kwargs = {
    'name': 'torchmcubes',
    'version': '0.1.0',
    'description': 'torchmcubes: marching cubes for PyTorch',
    'license': 'MIT',
    'author': 'Tatsuya Yatagawa',
    'author_email': 'tatsy.mail@gmail.com',
    'packages': [
        'torchmcubes'
    ],
    'classifiers': [
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
}

try:
    from torch.utils.cpp_extension import CUDAExtension

    setup_kwargs.update({
        'ext_modules': [
            CUDAExtension(
                'torchmcubes_module',
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
        'cmdclass': {
            'build_ext': BuildExtension
        }
    })
    setup(**setup_kwargs)

except:
    print('CUDA environment was not successfully loaded!')
    print('Build only CPU module!')

    from torch.utils.cpp_extension import CppExtension

    setup_kwargs.update({
        'ext_modules': [
            CppExtension('torchmcubes_module', [
                'cxx/mcubes.cpp',
                'cxx/mcubes_cpu.cpp',
                'cxx/grid_interp_cpu.cpp',
            ])
        ],
        'cmdclass': {
            'build_ext': BuildExtension
        }
    })
    setup(**setup_kwargs)
