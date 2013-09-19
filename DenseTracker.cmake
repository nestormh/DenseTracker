set(DENSETRACKER_PATH ${CMAKE_CURRENT_LIST_DIR}/src)

find_package(OpenCV  REQUIRED )

set(DENSETRACKER_CCFILES
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/IplImagePyramid.cpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/IplImageWrapper.cpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/functions.cpp
    ${DENSETRACKER_PATH}/densetracker.cpp
)

set (DENSETRACKER_HFILES
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/geometry/Box.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/geometry/Size.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/geometry/Point.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/numeric/functions.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/IplImagePyramid.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/functions.hpp
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1/opencv/IplImageWrapper.hpp
)

set (DENSETRACKER_INCLUDE_DIRS
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${DENSETRACKER_PATH}/dense_trajectory_release_v1.1
)

set (DENSETRACKER_LIBRARIES
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${BOOST_LIBRARIES}
)