cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# General settings
set(BUILD_TARGET torchmcubes_module)

# CUDA settings
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  add_definitions(-DWITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
  string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
  message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
else()
  message(WARNING "NO CUDA INSTALLATION FOUND, TRYING TO INSTALL CPU VERSION ONLY!")
endif()

# Recent PyTorch headers require C++20 (torch/all.h and ATen/ATen.h emit
# #error otherwise; torch 2.14 requires it, torch 2.9 still accepted C++17).
# Older torch versions compile fine with C++20 as well.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

# Find packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

find_package(Torch CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(cxx)
install(TARGETS ${BUILD_TARGET} LIBRARY DESTINATION .)
