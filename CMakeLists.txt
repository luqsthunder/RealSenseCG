
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
  ${PROJECT_SOURCE_DIR}/src/window.h
  
  ${PROJECT_SOURCE_DIR}/src/camera.h
  ${PROJECT_SOURCE_DIR}/src/camera.cc
  
  ${PROJECT_SOURCE_DIR}/src/shaderprogram.h
  ${PROJECT_SOURCE_DIR}/src/shaderprogram.cc
  
  ${PROJECT_SOURCE_DIR}/src/oclpointcloud.h
  ${PROJECT_SOURCE_DIR}/src/oclpointcloud.cc

  ${PROJECT_SOURCE_DIR}/src/RSCGutils.h
  
  ${PROJECT_SOURCE_DIR}/src/rect.h
  ${PROJECT_SOURCE_DIR}/src/rect.cc
  
  ${PROJECT_SOURCE_DIR}/src/graphprobs.h
  ${PROJECT_SOURCE_DIR}/src/graphprobs.cc)


 set(SHADERS ${PROJECT_SOURCE_DIR}/Shaders/simpleshader/shader.frag.glsl
             ${PROJECT_SOURCE_DIR}/Shaders/simpleshader/shader.vert.glsl
			  
			 ${PROJECT_SOURCE_DIR}/Shaders/SimplePointCloud/shader.frag.glsl
			 ${PROJECT_SOURCE_DIR}/Shaders/SimplePointCloud/shader.frag.glsl)
  
add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})

find_package(SFML 2 COMPONENTS network system REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLBINDING REQUIRED)
find_package(OpenCL REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

if(GLM_FOUND)
  include_directories(${GLM_INCLUDE_DIRS})
endif()

if(EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})
else()
  message("GLBINDING NOT FOUND")
endif()

if(SDL2_FOUND)
  include_directories(${SDL2_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${SDL2_LIBRARY})
else()
  message("SDL2 NOT FOUND")
endif()

if(WIN32 AND MSVC)
	find_package(RSSDK REQUIRED)
	
	source_group("src" FILES ${SOURCE_FILES})
	source_group("Shaders" FILES ${SHADERS})
	
	include_directories(${RSSDK_INCLUDE_DIRS})
	target_link_libraries(${EXECUTABLE_NAME} ${RSSDK_LIBRARIES})
endif()

if(OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${OpenCL_LIBRARIES})
else()
  message("OpenCL NOT FOUND")
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
