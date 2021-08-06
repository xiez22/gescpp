cmake_minimum_required(VERSION 3.17)
project(gescpp)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "/Users/xz/miniforge3/envs/arm38/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")

add_executable(gescpp main.cpp)
target_link_libraries(gescpp "${TORCH_LIBRARIES}")
set_property(TARGET gescpp PROPERTY CXX_STANDARD 20d)
