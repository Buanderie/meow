cmake_minimum_required(VERSION 2.8)
project( MEOW )

set(USE_CUDA FALSE CACHE BOOL "Use nVidia CUDA as GPGPU API")

## CHECK FOR COMPILER CAPABILITIES
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
##

# Need CUDA ?
IF( USE_CUDA )
FIND_PACKAGE(CUDA 5)
IF( CUDA_FOUND )
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30)
    add_definitions(-DWITH_CUDA)
ENDIF()
ENDIF()

# includes
include_directories( ${PROJECT_SOURCE_DIR}/src )
include_directories( /usr/local/include )

# Global shit
file (GLOB_RECURSE FILES_CPP ./src/*.c*)
add_executable( meow ${FILES_CPP})

set( DEPS "${DEPS};pthread" )

# Linking
target_link_libraries( meow ${DEPS})
