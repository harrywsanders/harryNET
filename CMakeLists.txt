cmake_minimum_required(VERSION 3.10)
project(harryNET VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# harryNET library with source files
add_library(harryNET STATIC src/tensor.cpp)
target_include_directories(harryNET PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add flags for optimization
target_compile_options(harryNET PRIVATE -O3)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(harryNET PUBLIC OpenMP::OpenMP_CXX)
endif()

# Enable testing
enable_testing()

# Build benchmark executable if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark.cpp")
    add_executable(benchmark benchmark/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE harryNET)
endif()

# Build tests if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp")
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE harryNET)
    add_test(NAME test_tensor COMMAND test_tensor)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_autograd.cpp")
    add_executable(test_autograd tests/test_autograd.cpp)
    target_link_libraries(test_autograd PRIVATE harryNET)
    add_test(NAME test_autograd COMMAND test_autograd)
endif()

# Build main executable if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(harrynet_main src/main.cpp)
    target_link_libraries(harrynet_main PRIVATE harryNET)
    
    # Set the main executable as the default target
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT harrynet_main)
endif()

# Install targets
install(TARGETS harryNET DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)