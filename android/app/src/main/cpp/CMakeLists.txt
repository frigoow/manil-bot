cmake_minimum_required(VERSION 3.22.1)
project(manille_android LANGUAGES CXX)

add_library(manille_mobile SHARED
    jni_bridge.cpp
    ../../../../../cpp/mobile/manille_mobile.cpp
    ../../../../../cpp/manille_game.cpp
    ../../../../../cpp/encoding.cpp
    ../../../../../cpp/rule_agent.cpp
    ../../../../../cpp/rule_playout.cpp
    ../../../../../cpp/fast_policy.cpp
    ../../../../../cpp/prior/trump_rule.cpp
    ../../../../../cpp/sampler/uniform.cpp
    ../../../../../cpp/sampler/trump_constrained.cpp
)

target_compile_features(manille_mobile PRIVATE cxx_std_17)
target_compile_options(manille_mobile PRIVATE -O3)
target_include_directories(manille_mobile PRIVATE
    ../../../../../cpp
    ../../../../../third_party/ismcsolver/include
)

find_library(log-lib log)
target_link_libraries(manille_mobile PRIVATE ${log-lib})
