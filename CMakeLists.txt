cmake_minimum_required(VERSION 3.16)
project(ObjectDetectionApp VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必需的包
find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 自动包含Qt的MOC、UIC和RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 源文件
set(SOURCES
    main.cpp
    MainWindow.cpp
    MainWindow.h
    ObjectDetector.cpp
    ObjectDetector.h
)

# UI文件
set(UI_FILES
    MainWindow.ui
)

# 创建可执行文件
add_executable(ObjectDetectionApp ${SOURCES} ${UI_FILES}
    FrameProcessor.h
    FrameProcessor.cpp)

# 链接库
target_link_libraries(ObjectDetectionApp PRIVATE
    Qt6::Core
    Qt6::Widgets
    ${OpenCV_LIBS}
)

# 设置包含目录
target_include_directories(ObjectDetectionApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 创建 models 目录（如果不存在）
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models")
endif()





