set(TARGET llama-bonsai-logits)
add_executable(${TARGET} bonsai-logits.cpp)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
