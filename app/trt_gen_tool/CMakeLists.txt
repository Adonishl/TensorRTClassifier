# check the cmake version
cmake_minimum_required(VERSION 3.5)
# Project name
project(TrtGenTool)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")

add_definitions(-D_GLIBCXX_USE_C99)

set(TENSORRT_INCLUDE /root/tensorrt/trt_include)
include_directories(${TENSORRT_INCLUDE})

set(REPO_ROOT ${TrtGenTool_SOURCE_DIR}/../..)
set(INCLUDE_ROOT ${REPO_ROOT}/include)
include_directories( ${INCLUDE_ROOT})

set(CUDA_VERSION cuda-8.0)
set(CUDA_INSTALL_DIR /usr/local/${CUDA_VERSION})

include_directories(${CUDA_INSTALL_DIR}/include)
include_directories(/usr/local/include)
set(CUDNN_LIB_PATH ${CUDA_INSTALL_DIR}/lib64)
set(TARGET_LIB_PATH ${CUDA_INSTALL_DIR}/targets/x86_64-linux/lib64)
link_directories(${CUDNN_LIB_PATH})
set(SYS_LIB /usr/local/lib)
link_directories(${SYS_LIB})

link_libraries(gflags)
find_package(gflags REQUIRED)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
include_directories( ${CUDA_INCLUDE_DIRS} )

set(TENSORRT_LIBRARIES "-lnvinfer -lnvparsers -lnvinfer_plugin")
link_libraries(${TENSORRT_LIBRARIES})
link_libraries(cudnn)
link_libraries(cublas)
link_libraries(cudart_static)
link_libraries(nvToolsExt)
link_libraries(cudart)
link_libraries(rt)
link_libraries(dl)
link_libraries(pthread)

cuda_add_executable(TrtGenTool main.cpp)
