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
