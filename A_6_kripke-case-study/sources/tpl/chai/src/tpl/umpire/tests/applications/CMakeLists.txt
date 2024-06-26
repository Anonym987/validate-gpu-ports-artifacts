##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (test_applications_depends umpire)

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (test_applications_depends
      ${test_applications_depends}
      openmp)
endif()

if (UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
  set (test_applications_depends
      ${test_applications_depends}
      ${CMAKE_DL_LIBS})
endif()

blt_add_executable(
  NAME pool_allocator_stress
  SOURCES pool_allocator_stress.cpp
  DEPENDS_ON ${test_applications_depends})
list(APPEND umpire_test_applications pool_allocator_stress)

blt_add_executable(
  NAME poison_memory_checks
  SOURCES poison_memory_checks.cpp
  DEPENDS_ON ${test_applications_depends})
list(APPEND umpire_test_applications poison_memory_checks)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  blt_add_target_link_flags(
    TO pool_allocator_stress
    FLAGS "-rdynamic")
endif ()

blt_add_target_compile_flags(
  TO pool_allocator_stress
  FLAGS ${UMPIRE_DISABLE_DEPRECATED_WARNINGS_FLAG})
blt_add_target_compile_flags(
  TO poison_memory_checks 
  FLAGS ${UMPIRE_DISABLE_DEPRECATED_WARNINGS_FLAG})

install(TARGETS ${umpire_test_applications} RUNTIME DESTINATION bin)
