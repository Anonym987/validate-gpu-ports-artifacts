##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
add_subdirectory(util)
add_subdirectory(resource)
add_subdirectory(alloc)
add_subdirectory(event)
add_subdirectory(op)
add_subdirectory(strategy)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()
if (UMPIRE_ENABLE_C OR UMPIRE_ENABLE_FORTRAN)
  add_subdirectory(interface/c_fortran)
endif()

set (umpire_headers
  Allocator.hpp
  Allocator.inl
  ResourceManager.hpp
  ResourceManager.inl
  Tracking.hpp
  TypedAllocator.hpp
  TypedAllocator.inl
  Umpire.hpp)

set (umpire_sources
  Allocator.cpp
  ResourceManager.cpp
  Umpire.cpp)

if (UMPIRE_ENABLE_DEVICE_ALLOCATOR AND (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP))
  set(umpire_device_headers
    DeviceAllocator.hpp
    device_allocator_helper.hpp)

  set(umpire_device_sources
    DeviceAllocator.cpp
    device_allocator_helper.cpp)

  if (UMPIRE_ENABLE_CUDA)
    set_source_files_properties(DeviceAllocator.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
    set_source_files_properties(device_allocator_helper.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
    set(umpire_device_depends camp cuda umpire_util)
  elseif (UMPIRE_ENABLE_HIP)
    set(umpire_device_depends camp blt::hip umpire_util)
    set(hip_version_min "5.4.0")

    if (${hip_VERSION} VERSION_LESS ${hip_version_min})
      message(FATAL_ERROR "ROCM version must be ${hip_version_min} or higher for HIP DeviceAllocator functionality!")
    endif ()
  endif()

  blt_add_library(
    NAME umpire_device
    HEADERS ${umpire_device_headers}
    SOURCES ${umpire_device_sources}
    DEPENDS_ON ${umpire_device_depends}
    OBJECT TRUE)

  target_include_directories(
    umpire_device
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  install(FILES
    ${umpire_device_headers}
    DESTINATION include/umpire)

  if (UMPIRE_ENABLE_CUDA)
    set_target_properties(umpire_device PROPERTIES
      CUDA_SEPARABLE_COMPILATION On)
  else ()
    target_link_options(umpire_device PUBLIC -fgpu-rdc)
    target_compile_options(umpire_device PUBLIC -fgpu-rdc)
  endif ()

endif()

if (UMPIRE_ENABLE_SYCL)
  set(umpire_headers
      ${umpire_headers}
      util/sycl_compat.hpp)
endif()

set (umpire_depends
  camp
  umpire_resource
  umpire_strategy
  umpire_alloc
  umpire_op
  umpire_util
  umpire_event
  umpire_tpl_judy)

if (UMPIRE_ENABLE_HIP)
  set (umpire_depends
    ${umpire_depends}
    blt::hip_runtime)
endif()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (umpire_depends
    ${umpire_depends}
    umpire_event
    openmp)
endif ()

blt_list_append( TO umpire_depends ELEMENTS umpire_interface IF UMPIRE_ENABLE_C )

if(UMPIRE_ENABLE_DEVICE_ALLOCATOR AND (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP))
  blt_list_append( TO umpire_depends ELEMENTS umpire_device)
endif()

set (umpire_defines)

if (COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS)
  set (umpire_defines
    ${umpire_defines}
    UMPIRE_WIN_STATIC_BUILD)
endif ()

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  DEPENDS_ON ${umpire_depends}
  DEFINES ${umpire_defines})

if (NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS 3.18)
  target_link_options(
    umpire INTERFACE
    $<BUILD_INTERFACE:$<$<NOT:$<LINK_LANGUAGE:Fortran>>:-rdynamic>>
  )

  target_link_libraries(
    umpire INTERFACE $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>
  )
endif()

if (UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  message(STATUS "Host Shared Memory Enabled")
  find_package(Threads)
  target_link_libraries(
    umpire INTERFACE Threads::Threads -lrt 
  )
else()
  message(STATUS "Host Shared Memory Disabled")
endif()

set_target_properties(umpire PROPERTIES
  CUDA_SEPARABLE_COMPILATION On)

if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_PINNED On)
  set(UMPIRE_ENABLE_UM On)
  if (UMPIRE_ENABLE_DEVICE_CONST)
    set(UMPIRE_ENABLE_CONST On)
  endif()
elseif (UMPIRE_ENABLE_SYCL)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_UM On)
  set(UMPIRE_ENABLE_PINNED On)
else ()
  set(UMPIRE_ENABLE_DEVICE Off)
  set(UMPIRE_ENABLE_PINNED Off)
  set(UMPIRE_ENABLE_CONST Off)
  set(UMPIRE_ENABLE_UM Off)
endif()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set(UMPIRE_ENABLE_DEVICE On)
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  ${CMAKE_BINARY_DIR}/include/umpire/config.hpp)

target_include_directories(
  umpire
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_headers}
  DESTINATION include/umpire)

install(DIRECTORY
  ${CMAKE_BINARY_DIR}/include
  DESTINATION ".")

umpire_add_code_checks(PREFIX umpire)
