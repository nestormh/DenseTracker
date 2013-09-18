set(CCFILES
#   dense_trajectory_release_v1.1/DenseTrack.cpp
  dense_trajectory_release_v1.1/opencv/IplImagePyramid.cpp
  dense_trajectory_release_v1.1/opencv/IplImageWrapper.cpp
  dense_trajectory_release_v1.1/opencv/functions.cpp
  densetracker.cpp
  main.cpp
)

set (HFILES
  dense_trajectory_release_v1.1/geometry/Box.hpp
  dense_trajectory_release_v1.1/geometry/Size.hpp
  dense_trajectory_release_v1.1/geometry/Point.hpp
  dense_trajectory_release_v1.1/numeric/functions.hpp
  dense_trajectory_release_v1.1/opencv/IplImagePyramid.hpp
  dense_trajectory_release_v1.1/opencv/functions.hpp
  dense_trajectory_release_v1.1/opencv/IplImageWrapper.hpp

)

set (CUHFILES

)

set(CUFILES
)

set(HFILES_PUBLIC
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)

set(CUDAGDB_OPTS -g -G)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_executable(testDenseTracker
  ${CUFILES} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
    OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS}
#   OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS} ${CUDA_BIND_COMPILER}
#   ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS} ${CUDAGDB_OPTS}
)
  
include_directories(testDenseTracker
#     GPULibSVM
#     ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
#     emon.a
    ${CUDA_INCLUDE_DIRS}
    dense_trajectory_release_v1.1
)

# set(CUDA_CURAND_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so)

target_link_libraries(testDenseTracker
#   ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${BOOST_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
)
CUDA_ADD_CUBLAS_TO_TARGET(testDenseTracker)
