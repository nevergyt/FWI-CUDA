cmake_minimum_required(VERSION 2.8.8)

set (CMAKE_C_COMPILER "D:/Program Files (x86)/MinGW/mingw64/bin/gcc.exe")
project(FWI C)

#--------------------------------------
#   Common compilation flags, if any.
#--------------------------------------

message(STATUS "Compiler Id ${CMAKE_C_COMPILER_ID}")

#---------------------------------------------------
# Configure compiler flags for Intel/GCC compilers
#--------------------------------------------------
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -qopenmp -fma -align -finline-functions")
else()
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fopenmp")
endif()

add_definitions ("-DSHARED_MEMORY_IMPLEMENTATION")

#--------------------------------------
#   Configure release
#--------------------------------------
set (debug "Debug mode enabled" OFF)

if (debug)
	message(STATUS "Debug mode enabled")
	add_definitions ("-DDEBUG")
else()
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-result")
endif()

#--------------------------------------
#       Configure Input/Output
#--------------------------------------
option (perform-io "Perform local I/O" ON)

if (perform-io)
	message(STATUS "Input/Output is enabled.")
else()
	message(STATUS "Input/Output is disabled.")
	add_definitions("-DDO_NOT_PERFORM_IO")
endif()

#--------------------------------------
#        Use NMVe local storage
#--------------------------------------
option (use-nvme "Use NVMe devices for local I/O" OFF)

if (use-nvme)
	add_definitions("-DUSE_NVME")
	message(STATUS "NMVe devices will be used for local I/O")
else()
	message(STATUS "Default file system will be used for local I/O")
endif()


message(STATUS "CMAKE_C_FLAGS are ${CMAKE_C_FLAGS}")

#--------------------------------------
#    Create model generator utility
#--------------------------------------
include_directories(
	"../common/"
	"../0_version_openmp/"
	"./")
add_executable(generateModel.bin
	"../common/fwi_constants.cu"
	"../common/fwi_common.cu"
	#"../common/fwi_comms.c"
	"../0_version_openmp/fwi_propagator.c"
	"../common/fwi_kernel.cu"
	"../common/fwi_sched.cu"
	"../common/fwi_generatemodel.c")

target_link_libraries(generateModel.bin "-lm")
install (TARGETS generateModel.bin DESTINATION "${CMAKE_SOURCE_DIR}/")

#--------------------------------------
#   Create model scheduler utility
#---------------------------------------
include_directories(
	"../common/"
	"../0_version_openmp/"
	"./")
add_executable(generateSchedule.bin
	"../common/fwi_constants.cu"
	"../common/fwi_common.cu"
	#"../common/fwi_comms.c"
	"../0_version_openmp/fwi_propagator.c"
	"../common/fwi_kernel.cu"
	"../common/fwi_sched.cu"
	"../common/fwi_schedule.cu")

target_link_libraries(generateSchedule.bin "-lm")
install (TARGETS generateSchedule.bin DESTINATION "${CMAKE_SOURCE_DIR}/")
