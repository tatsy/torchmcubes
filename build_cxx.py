import platform

from setuptools import setup
from setuptools.errors import CCompilerError, PackageDiscoveryError
from torch.utils.cpp_extension import BuildExtension


class MyBuildExtension(BuildExtension):
    def run(self):
        try:
            super(MyBuildExtension, self).run()
        except FileNotFoundError:
            raise Exception("File not found. Could not compile C extension")

    def build_extension(self, ext):
        # common settings
        for e in self.extensions:
            pass

        # OS specific settings
        if platform.system() == "Darwin":
            for e in self.extensions:
                e.extra_compile_args.extend([
                    "-Xpreprocessor",
                    "-fopenmp",
                    "-mmacosx-version-min=10.15",
                ])
                e.extra_link_args.extend([
                    "-lomp",
                ])

        elif platform.system() == "Linux":
            for e in self.extensions:
                e.extra_compile_args.extend([
                    "-fopenmp",
                ])
                e.extra_link_args.extend([
                    "-fopenmp",
                ])

        # compiler specific settings
        if self.compiler.compiler_type == "unix":
            for e in self.extensions:
                e.extra_compile_args.extend([
                    "-std=c++17",
                    "-pthread",
                ])

        elif self.compiler.compiler_type == "msvc":
            for e in self.extensions:
                e.extra_compile_args.extend(["/utf-8", "/std:c++17", "/openmp"])
                e.define_macros.extend([
                    ("_CRT_SECURE_NO_WARNINGS", 1),
                    ("_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING", 1),
                ])

        # building
        try:
            super(MyBuildExtension, self).build_extension(ext)
        except (CCompilerError, PackageDiscoveryError, ValueError):
            raise Exception("Could not compile C extension")


def build(setup_kwargs):
    from torch.utils.cpp_extension import CUDAExtension

    try:
        setup_kwargs.update({
            "ext_modules": [
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
            "cmdclass": {
                "build_ext": BuildExtension
            }
        })
    except:
        from torch.utils.cpp_extension import CppExtension

        print('CUDA environment was not successfully loaded!')
        print('Build only CPU module!')

        setup_kwargs.update({
            'ext_modules': [
                CppExtension('torchmcubes_module', [
                    'cxx/mcubes.cpp',
                    'cxx/mcubes_cpu.cpp',
                    'cxx/grid_interp_cpu.cpp',
                ])
            ],
            'cmdclass': {
                'build_ext': MyBuildExtension
            }
        })
