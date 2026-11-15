cmake_minimum_required(VERSION 3.15)
project(manille_mobile CXX)

# The C++ the app runs (rules engine, world samplers, rule agent and its
# play-out), built on the desktop so it can be tested without a phone. The
# Android build compiles the same files through
# android/app/src/main/cpp/CMakeLists.txt.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
enable_testing()

# Vendored, header-only, MIT (third_party/ismcsolver/LICENSE): only the game
# interface the engine implements.
set(ISMCTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ismcsolver/include)

add_library(manille_mobile_core STATIC
    mobile/manille_mobile.cpp
    manille_game.cpp
    encoding.cpp
    rule_agent.cpp
    rule_playout.cpp
    fast_policy.cpp
    prior/trump_rule.cpp
    sampler/uniform.cpp
    sampler/trump_constrained.cpp
)

function(manille_configure target)
    target_include_directories(${target} PRIVATE ${ISMCTS_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2)
    else()
        target_compile_options(${target} PRIVATE -O3)
    endif()
endfunction()

manille_configure(manille_mobile_core)
target_link_libraries(manille_mobile_core PRIVATE Threads::Threads)

foreach(test mobile_api trump_rule trump_constrained_sampler)
    add_executable(manille_test_${test} test_${test}.cpp)
    manille_configure(manille_test_${test})
    target_link_libraries(manille_test_${test} PRIVATE manille_mobile_core Threads::Threads)
    add_test(NAME ${test} COMMAND manille_test_${test})
endforeach()

# The research build (research/README.md): the Python module manille_ismcts
# that the DMC training, SPARTA and the evaluation harness use -- the same
# engine, encoders and rule agent as the app, plus the ISMCTS solvers and
# libtorch inference for the search players. Off by default: the app and its
# tests need neither pybind11 nor libtorch.
#   cmake -B build -S cpp -DMANILLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
#         -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
option(MANILLE_PYTHON "Build the manille_ismcts Python module for research/" OFF)
if(MANILLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(manille_ismcts
        bindings.cpp nn_policy.cpp solver_wrapper.cpp batched_ismcts.cpp
        manille_game.cpp encoding.cpp rule_agent.cpp rule_playout.cpp fast_policy.cpp
        prior/trump_rule.cpp sampler/uniform.cpp sampler/trump_constrained.cpp)
    manille_configure(manille_ismcts)
    set_property(TARGET manille_ismcts PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    # libtorch from the Python environment's torch package unless given
    if(NOT DEFINED TORCH_DIR)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
            OUTPUT_VARIABLE TORCH_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    target_include_directories(manille_ismcts PRIVATE
        ${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include)
    target_link_directories(manille_ismcts PRIVATE ${TORCH_DIR}/lib)
    target_link_libraries(manille_ismcts PRIVATE torch torch_cpu c10 Threads::Threads)
    if(EXISTS ${TORCH_DIR}/lib/libtorch_cuda.so)
        target_link_libraries(manille_ismcts PRIVATE torch_cuda c10_cuda)
    endif()
    set_target_properties(manille_ismcts PROPERTIES
        BUILD_RPATH "${TORCH_DIR}/lib" INSTALL_RPATH "${TORCH_DIR}/lib")
endif()
