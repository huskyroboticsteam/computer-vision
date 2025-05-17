cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Find Torch
find_package(Torch REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(test test.cpp)

# Link against Torch and OpenCV
target_link_libraries(test "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set C++ standard
set_property(TARGET test PROPERTY CXX_STANDARD 17)
