cmake_minimum_required(VERSION 3.1)
project(RealSenseCG)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(EXECUTABLE_NAME RealSenseCG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/main.cc

  ${PROJECT_SOURCE_DIR}/src/shaderprogram.h
  ${PROJECT_SOURCE_DIR}/src/shaderprogram.cc

  ${PROJECT_SOURCE_DIR}/src/realsenseimage.cc
  ${PROJECT_SOURCE_DIR}/src/realsenseimage.h

  ${PROJECT_SOURCE_DIR}/src/window.cc
  ${PROJECT_SOURCE_DIR}/src/window.h)


add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLBINDING REQUIRED)

if(SDL2_FOUND)
  include_directories(${SDL2_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${SDL2_LIBRARY} realsense)
else()
  message("SDL2 NOT FOUND")
endif()

if(GLBINDING_FOUND)
  include_directories(${GLBINDING_INCLUDE_DIRS})
  target_link_libraries(${EXECUTABLE_NAME} ${GLBINDING_LIBRARIES})
else()
  message("GLBINDING NOT FOUND")
endif()

if(OPENGL_FOUND)
  include_directories(${OPENGL_INCLUDE_DIRS})
  target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_LIBRARIES})
else()
  message("OpenGL NOT FOUND")
endif()
