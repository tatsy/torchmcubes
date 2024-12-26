foreach (DIR IN LISTS TORCH_INCLUDE_DIRS)
  if (EXISTS "${DIR}")
    include_directories(${DIR})
  else()
    message(WARNING "Directory ${DIR} does not exist.")
  endif()
endforeach()

Python_add_library(${BUILD_TARGET}
  mcubes.cpp
  mcubes_cpu.cpp
  grid_interp_cpu.cpp)

if (CMAKE_CUDA_COMPILER)
  target_sources(${BUILD_TARGET} PRIVATE
    cuda_utils.h
    helper_math.h
    macros.h
    mcubes_cuda.cu
    grid_interp_cuda.cu)
endif()

target_link_libraries(${BUILD_TARGET} PRIVATE
  pybind11::headers
  ${TORCH_PYTHON_LIBRARY}
)

foreach (LIB IN LISTS TORCH_LIBRARIES)
  if (EXISTS ${LIB} OR ${LIB} MATCHES "torch.*")
    target_link_libraries(${BUILD_TARGET} PRIVATE ${LIB})
  else()
    message(WARNING "Library ${LIB} does not exist.")
  endif()
endforeach()

if (OPENMP_FOUND)
  target_link_libraries(${BUILD_TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(${BUILD_TARGET} PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET "hidden")

target_compile_definitions(${BUILD_TARGET} PRIVATE VERSION_INFO=${PROJECT_VERSION})

pybind11_extension(${BUILD_TARGET})
pybind11_strip(${BUILD_TARGET})
