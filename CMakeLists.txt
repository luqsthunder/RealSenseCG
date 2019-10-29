cmake_minimum_required(VERSION 3.10)
project(LibrasCapture)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(EXECUTABLE_NAME LibrasCapture)

set(CMAKE_AUTOMOC ON)

set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cc

    ${PROJECT_SOURCE_DIR}/src/csv.h

    ${PROJECT_SOURCE_DIR}/src/camera.h
    ${PROJECT_SOURCE_DIR}/src/camera.cc

    ${PROJECT_SOURCE_DIR}/src/datasetdirwidget.h
    ${PROJECT_SOURCE_DIR}/src/datasetdirwidget.cc
)

add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)
find_package(KinectSDKv2)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 14)

if(KinectSDK2_FOUND)
    include_directories(${KinectSDK2_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} ${KinectSDK2_LIBRARIES})
endif()

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIR})
    target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})
else()
    message("OPENCV2 NOT FOUND")
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} ${Boost_LIBRARIES})
else()
    message("Boost NOT FOUND")
endif()

if(Qt5_FOUND)
    include_directories(${Qt5Widgets_INCLUDES} ${Qt5Core_INCLUDES}
        ${Qt5Gui_INCLUDES})
    target_link_libraries(${EXECUTABLE_NAME}
        ${Qt5Widgets_LIBRARIES}
        ${Qt5Gui_LIBRARIES}
        ${Qt5Core_LIBRARIES})
else()
    message("Qt5 NOT FOUND")
endif()

if(WIN32 AND MSVC)
    source_group("src" FILES ${SOURCE_FILES})
endif()
