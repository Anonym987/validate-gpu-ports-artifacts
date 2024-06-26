##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(tools_depends umpire umpire_tpl_CLI11 umpire_tpl_json)

if (UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
  set(tools_depends ${tools_depends} ${CMAKE_DL_LIBS})
endif()

set(replay_headers
  ReplayInterpreter.hpp
  ReplayInterpreter.inl
  ReplayMacros.hpp
  ReplayOperationManager.hpp
  ReplayOptions.hpp
  ReplayFile.hpp)

set(replay_sources
  ReplayInterpreter.cpp
  ReplayOperationManager.cpp
  ReplayFile.cpp)

blt_add_executable(
  NAME replay
  SOURCES replay.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends})
list(APPEND replay_tools replay)

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends})
list(APPEND replay_tools replaydiff)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
  blt_add_target_link_flags(
    TO replay
    FLAGS "-rdynamic")
  blt_add_target_link_flags(
    TO replaydiff
    FLAGS "-rdynamic")
endif ()

blt_add_target_compile_flags(
  TO replay
  FLAGS ${UMPIRE_DISABLE_DEPRECATED_WARNINGS_FLAG})
blt_add_target_compile_flags(
  TO replaydiff
  FLAGS ${UMPIRE_DISABLE_DEPRECATED_WARNINGS_FLAG})

install(TARGETS ${replay_tools} RUNTIME DESTINATION bin)

umpire_add_code_checks(PREFIX tools)
