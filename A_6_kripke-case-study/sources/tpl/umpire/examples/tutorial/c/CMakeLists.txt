##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (tutorial_depends umpire)

blt_add_executable(
  NAME tut_allocator_c
  SOURCES tut_allocator.c
  DEPENDS_ON ${tutorial_depends})
list(APPEND umpire_tutorials tut_allocator_c)

blt_add_executable(
  NAME tut_resources_c
  SOURCES tut_resources.c
  DEPENDS_ON ${tutorial_depends})
list(APPEND umpire_tutorials tut_resources_c)

blt_add_executable(
  NAME tut_pool_c
  SOURCES tut_pool.c
  DEPENDS_ON ${tutorial_depends})
list(APPEND umpire_tutorials tut_pool_c)
