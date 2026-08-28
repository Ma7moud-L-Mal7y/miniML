cmake_minimum_required(VERSION 3.14)
project(minml LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include path for headers (equivalent to -Iinclude)
include_directories(include)

# 1. Create the core library target from all source files in src/
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
add_library(minml_lib ${LIB_SOURCES})

# 2. Fetch and configure Google Test automatically
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# Prevent GTest from overriding parent project runtime/compiler settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CTest runner
enable_testing()

# 3. Discover all test source files in test/ and create test executables
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "test/*.cpp")

foreach(TEST_SRC ${TEST_SOURCES})
    # Extract filename without directory or extension (e.g., test/Matrix_test.cpp -> Matrix_test)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    
    # Define test executable
    add_executable(${TEST_NAME} ${TEST_SRC})
    
    # Link test with core minml library and GTest main entry point
    target_link_libraries(${TEST_NAME} PRIVATE minml_lib gtest_main)
    
    # Register with CTest framework
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()