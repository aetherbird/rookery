cmake_minimum_required(VERSION 3.24)

option(ROOKERY_CPU_ONLY "Build CPU-only rookery without CUDA" OFF)

# Default targets: Volta (sm_70) + Ampere (sm_86).
# Override via -DCMAKE_CUDA_ARCHITECTURES="52" (Maxwell), "52;60;61" (Maxwell+Pascal), etc.
# NOTE: Must be set BEFORE enabling CUDA, otherwise CMake defaults to sm_52.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;86" CACHE STRING "CUDA target architectures")
endif()

project(rookery LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOKERY_HAS_CUDA OFF)
if(NOT ROOKERY_CPU_ONLY)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(ROOKERY_HAS_CUDA ON)
    else()
        message(STATUS "CUDA compiler not found — building rookery in CPU-only mode")
    endif()
else()
    message(STATUS "ROOKERY_CPU_ONLY=ON — building rookery in CPU-only mode")
endif()

find_package(Threads REQUIRED)

add_library(rookery_http STATIC
    third_party/cpp-httplib/httplib.cpp
)
target_include_directories(rookery_http PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_options(rookery_http PRIVATE -O3 -w)
target_link_libraries(rookery_http PUBLIC Threads::Threads)

set(ROOKERY_CORE_SOURCES
    src/gguf/reader.cpp
    src/gguf/sharded.cpp
    src/tokenizer/bpe.cpp
    src/engram/engram_table.cpp
    src/engram/engram_sidecar.cpp
    src/cpu/cpu_kernels.cpp
    src/cpu/bailingmoe3_cpu.cpp
    src/cpu/gemma4_cpu.cpp
    src/cpu/qwen2_cpu.cpp
    src/cpu/qwen35_cpu.cpp
)

if(ROOKERY_HAS_CUDA)
    list(APPEND ROOKERY_CORE_SOURCES
        src/gguf/loader.cu
        src/core/device.cu
        src/engram/engram_kernels.cu
        src/kernels/rmsnorm_rope.cu
        src/kernels/hc_fused.cu
        src/kernels/gemv_quant.cu
        src/kernels/gemv_kquant.cu
        src/kernels/maxwell_kquant.cu
        src/kernels/bailingmoe3_pascal.cu
        src/kernels/moe_fused.cu
        src/kernels/attn_flash.cu
        src/kernels/msa_sparse.cu
        src/kernels/kda_gdn.cu
        src/kernels/mla_attn.cu
        src/kernels/inkling2.cu
        src/kernels/qwen_next.cu
        src/kernels/qwen35_pascal.cu
        src/kernels/qwen4exp.cu
        src/kernels/gemma4.cu
        src/kernels/raon1_mini.cu
        src/kernels/sensenova_loop.cu
        src/models/maxwell_moe.cu
        src/models/glm5_next.cu
        src/models/inkling2_loop.cu
        src/models/qwen35_moe.cu
        src/models/qwen4exp.cu
        src/models/bailingmoe3.cu
        src/models/gemma4.cu
        src/models/raon1_mini.cu
        src/models/sensenova_loop.cu
    )
else()
    list(APPEND ROOKERY_CORE_SOURCES
        src/core/device_cpu.cpp
    )
endif()

add_library(rookery_core STATIC ${ROOKERY_CORE_SOURCES})

target_include_directories(rookery_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_options(rookery_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Wno-deprecated-gpu-targets -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
)

target_link_libraries(rookery_core PUBLIC Threads::Threads)

if(ROOKERY_HAS_CUDA)
    target_compile_definitions(rookery_core PUBLIC ROOKERY_ENABLE_CUDA=1)
    target_link_libraries(rookery_core PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
endif()

add_executable(rookery-server src/main.cpp)
target_link_libraries(rookery-server PRIVATE rookery_core rookery_http)

add_executable(rookery_inspect tests/test_gguf.cpp)
target_link_libraries(rookery_inspect PRIVATE rookery_core)

add_executable(rookery_bailingmoe3_test tests/test_bailingmoe3_forward.cpp)
target_link_libraries(rookery_bailingmoe3_test PRIVATE rookery_core)

add_executable(rookery_gemma4_test tests/test_gemma4_forward.cpp)
target_link_libraries(rookery_gemma4_test PRIVATE rookery_core)

add_executable(rookery_qwen2_test tests/test_qwen2_forward.cpp)
target_link_libraries(rookery_qwen2_test PRIVATE rookery_core)

add_executable(rookery_unit_test tests/test_server_units.cpp)
target_link_libraries(rookery_unit_test PRIVATE rookery_core)

if(ROOKERY_HAS_CUDA)
    add_executable(rookery_maxwell_test tests/test_maxwell_forward.cpp)
    target_link_libraries(rookery_maxwell_test PRIVATE rookery_core)

    add_executable(rookery_glm5_test tests/test_glm5_forward.cpp)
    target_link_libraries(rookery_glm5_test PRIVATE rookery_core)

    add_executable(rookery_inkling2_test tests/test_inkling2_forward.cpp)
    target_link_libraries(rookery_inkling2_test PRIVATE rookery_core)

    add_executable(rookery_qwen35_test tests/test_qwen35_forward.cpp)
    target_link_libraries(rookery_qwen35_test PRIVATE rookery_core)

    add_executable(rookery_qwen4exp_test tests/test_qwen4exp_forward.cpp)
    target_link_libraries(rookery_qwen4exp_test PRIVATE rookery_core)

    add_executable(rookery_raon1_test tests/test_raon1_forward.cpp)
    target_link_libraries(rookery_raon1_test PRIVATE rookery_core)

    add_executable(rookery_sensenova_test tests/test_sensenova_forward.cpp)
    target_link_libraries(rookery_sensenova_test PRIVATE rookery_core)

    add_executable(rookery_engram_test tests/test_engram_sidecar.cpp)
    target_link_libraries(rookery_engram_test PRIVATE rookery_core)

    add_executable(rookery-fit-adapter tools/fit_engram_adapter.cpp)
    target_link_libraries(rookery-fit-adapter PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()
