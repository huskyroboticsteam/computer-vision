cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Find Torch
find_package(Torch REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(object_detection object_detection.cpp)

# Link against Torch and OpenCV
target_link_libraries(object_detection "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(object_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set C++ standard
set_property(TARGET object_detection PROPERTY CXX_STANDARD 17)
