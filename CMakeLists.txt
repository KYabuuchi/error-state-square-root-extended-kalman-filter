cmake_minimum_required(VERSION 3.0)
project(sqrt-ekf)
# Eigen3
find_package(Eigen3 REQUIRED)
message("Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
# main
add_executable(main main.cpp)
target_link_libraries(main)
