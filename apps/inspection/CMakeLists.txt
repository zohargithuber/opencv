ocv_add_application(opencv_inspection
    MODULES opencv_core opencv_imgproc opencv_geometry opencv_imgcodecs opencv_dnn
    SRCS main.cpp inspection.cpp missing_part.cpp scratch_diff.cpp scratch_dnn.cpp dual_inspector.cpp stripe_pool.cpp clahe.cpp)

# clahe: cuda -- built only when a CUDA compiler is found (Jetson, or x86 with an NVIDIA GPU).
# Its own static library, so OpenCV's C++ flags never reach nvcc.
option(INSPECTION_WITH_CUDA "Build the CUDA CLAHE backend of opencv_inspection" ON)
if(INSPECTION_WITH_CUDA AND TARGET opencv_inspection)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED INSPECTION_CUDA_ARCHITECTURES)
      set(INSPECTION_CUDA_ARCHITECTURES "87;86;75") # Orin, Ampere desktop, Turing
    endif()
    add_library(opencv_inspection_cuda STATIC clahe_cuda.cu)
    set_target_properties(opencv_inspection_cuda PROPERTIES
                          CUDA_ARCHITECTURES "${INSPECTION_CUDA_ARCHITECTURES}" CUDA_STANDARD 14)
    target_link_libraries(opencv_inspection_cuda PRIVATE CUDA::cudart)
    target_link_libraries(opencv_inspection PRIVATE opencv_inspection_cuda CUDA::cudart)
    target_compile_definitions(opencv_inspection PRIVATE INSPECTION_WITH_CUDA=1)
    message(STATUS "opencv_inspection: CUDA CLAHE backend enabled (${INSPECTION_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "opencv_inspection: no CUDA compiler -- clahe: cuda not available")
  endif()
endif()
