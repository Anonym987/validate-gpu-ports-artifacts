##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

install(
  FILES analysis/plot_allocations analysis/plot_allocator_traces
  DESTINATION bin)

add_subdirectory(replay)
