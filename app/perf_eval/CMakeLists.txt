# check the cmake version
cmake_minimum_required(VERSION 3.5)
if (UNIX)
  add_definitions(-Wno-sign-compare -Wno-enum-compare)
endif(UNIX)
# Project name
project(PerfEval)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PerfEval_SOURCE_DIR}/cmake_module ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")

set(CUDA_ROOT /usr/local/cuda)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
include_directories( ${CUDA_INCLUDE_DIRS} )
add_definitions(-D_GLIBCXX_USE_C99)
set(TENSORRT_INCLUDE /root/tensorrt/trt_include)
include_directories(${TENSORRT_INCLUDE})
set(CUDA_VERSION cuda-8.0)
set(CUDA_INSTALL_DIR /usr/local/${CUDA_VERSION})
include_directories(${CUDA_INSTALL_DIR}/include)
include_directories(/usr/local/include)
set(CUDNN_LIB_PATH ${CUDA_INSTALL_DIR}/lib64)
set(TARGET_LIB_PATH ${CUDA_INSTALL_DIR}/targets/x86_64-linux/lib64)
link_directories(${CUDNN_LIB_PATH})
set(SYS_LIB /usr/local/lib)
link_directories(${SYS_LIB})
set(TENSORRT_LIBRARIES "-lnvinfer -lnvparsers -lnvinfer_plugin")
link_libraries(${TENSORRT_LIBRARIES})
add_definitions(-DUSE_TENSORRT)
link_libraries(cudnn)
link_libraries(cublas)
link_libraries(cudart_static)
link_libraries(nvToolsExt)
link_libraries(cudart)
link_libraries(rt)
link_libraries(dl)
link_libraries(pthread)


set(REPO_ROOT ${PerfEval_SOURCE_DIR}/../..)
set(INCLUDE_ROOT ${REPO_ROOT}/include)

#add include folder to the project
include_directories( ${INCLUDE_ROOT})

#add opencv lib
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_libraries( ${OpenCV_LIBS})

link_libraries( ssl )
link_libraries( crypto )
link_libraries( boost_system )
link_libraries( boost_filesystem )

#gflags
link_libraries( gflags )
find_package(gflags REQUIRED)

# For convenience we define the sources as a variable. You can add 
# header files and cpp/c files and CMake will sort them out
file(GLOB_RECURSE SRCS
  ${REPO_ROOT}/src/classifier.cpp 
  )

cuda_add_library(ClassifierLib ${SRCS})
cuda_add_executable(PerfEval main.cpp)
target_link_libraries(PerfEval ClassifierLib)
