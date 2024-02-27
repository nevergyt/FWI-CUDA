cmake_minimum_required(VERSION 3.27)
project(FWI CUDA)

set(CMAKE_CUDA_STANDARD 26)

include_directories(
        "../common/"
        "./")

add_executable(FWI main.cu
        ./fwi_main.cu
        ../common/fwi_constants.cu
        ../common/fwi_common.cu
        ../common/fwi_kernel.cu
        ../common/fwi_sched.cu
        fwi_propagator.cu
)


set_target_properties(FWI PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
