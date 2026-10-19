cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

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

  # NVTX workaround for CUDA 12
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12
     AND NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
    message(STATUS "CUDA::nvToolsExt not found; aliasing it to CUDA::nvtx3 for PyTorch")
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
    target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
  endif()

  message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_ID})")
  message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(WARNING "NO CUDA INSTALLATION FOUND, TRYING TO INSTALL CPU VERSION ONLY!")
endif()

# Recent PyTorch headers require C++20 (e.g., torch 2.14 requires it)
# Older torch versions compile fine with C++20 as well.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
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
