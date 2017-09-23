cmake_minimum_required(VERSION 2.8.8)

set(PROJECT_NAME CatEM)

project (${PROJECT_NAME})

find_package(OpenCV REQUIRED)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})