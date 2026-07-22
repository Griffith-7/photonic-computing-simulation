cmake_minimum_required(VERSION 3.20)
project(onn_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

add_library(onn_core STATIC
    src/core/types.cpp
    src/photonic/mzi.cpp
    src/photonic/phase_shifter.cpp
    src/photonic/reck.cpp
    src/photonic/clements.cpp
    src/photonic/svd_mapper.cpp
    src/physical/drift.cpp
    src/physical/attenuation.cpp
    src/physical/shot_noise.cpp
    src/physical/crosstalk.cpp
    src/physical/physical_layer.cpp
    src/physical/drift_compensation.cpp
    src/network/linear_layer.cpp
    src/network/activation.cpp
    src/network/network.cpp
    src/network/trainer.cpp
    src/data/mnist_loader.cpp
)
target_include_directories(onn_core PUBLIC src)
target_link_libraries(onn_core PUBLIC Eigen3::Eigen)

add_executable(onn_benchmark src/main.cpp)
target_link_libraries(onn_benchmark PRIVATE onn_core)

enable_testing()
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(onn_tests
    tests/test_mzi.cpp
    tests/test_mesh.cpp
    tests/test_physical.cpp
    tests/test_network.cpp
    tests/test_e2e.cpp
)
target_link_libraries(onn_tests PRIVATE onn_core GTest::gtest_main)
add_test(NAME onn_tests COMMAND onn_tests)

option(ONN_BUILD_PYTHON "Build Python bindings" OFF)
if(ONN_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(onn_python src/python/bindings.cpp)
    target_link_libraries(onn_python PRIVATE onn_core)
    target_include_directories(onn_python PRIVATE src)
endif()
