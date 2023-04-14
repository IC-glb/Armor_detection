cmake_minimum_required(VERSION 3.20)
project(Complete_detection)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR D:\\opencv\\build\\x64\\mingw\\lib)

find_package(OpenCV REQUIRED)



include_directories(include)
include_directories(D:\\opencv\\build\\x64)
include_directories(C:\\eigen-3.4.0)
include_directories(D:\\clion tect\\Complete_detection\\include)


add_executable(Complete_detection main.cpp)
add_executable(armor_detection  armor_detection.cpp)
target_link_libraries(Complete_detection ${OpenCV_LIBS})
target_link_libraries(armor_detection ${OpenCV_LIBS})