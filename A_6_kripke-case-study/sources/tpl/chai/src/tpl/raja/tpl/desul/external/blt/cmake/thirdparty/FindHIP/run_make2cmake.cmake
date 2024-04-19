# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
# Computes dependencies using HIPCC
###############################################################################

###############################################################################
# This file converts dependency files generated using hipcc to a format that
# cmake can understand.

# Input variables:
#
# input_file:STRING=<> Dependency file to parse. Required argument
# output_file:STRING=<> Output file to generate. Required argument

if(NOT input_file OR NOT output_file)
    message(FATAL_ERROR "You must specify input_file and output_file on the command line")
endif()

file(READ ${input_file} depend_text)

if (NOT "${depend_text}" STREQUAL "")
    string(REPLACE " /" "\n/" depend_text ${depend_text})
    string(REGEX REPLACE "^.*:" "" depend_text ${depend_text})
    string(REGEX REPLACE "[ \\\\]*\n" ";" depend_text ${depend_text})

    set(dependency_list "")

    foreach(file ${depend_text})
        string(REGEX REPLACE "^ +" "" file ${file})
        if(NOT EXISTS "${file}")
            message(WARNING " Removing non-existent dependency file: ${file}")
            set(file "")
        endif()

        if(NOT IS_DIRECTORY "${file}")
            get_filename_component(file_absolute "${file}" ABSOLUTE)
            list(APPEND dependency_list "${file_absolute}")
        endif()
    endforeach()
endif()

# Remove the duplicate entries and sort them.
list(REMOVE_DUPLICATES dependency_list)
list(SORT dependency_list)

foreach(file ${dependency_list})
    set(hip_hipcc_depend "${hip_hipcc_depend} \"${file}\"\n")
endforeach()

file(WRITE ${output_file} "# Generated by: FindHIP.cmake. Do not edit.\nSET(HIP_HIPCC_DEPEND\n ${hip_hipcc_depend})\n\n")
# vim: ts=4:sw=4:expandtab:smartindent
