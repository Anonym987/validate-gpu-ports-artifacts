#!/bin/bash

EXTRA_PROF_WRAPPER=off cmake .. -C ../host-configs/deep-est-cn-gcc.cmake -DCMAKE_BUILD_TYPE=Release
EXTRA_PROF_SCOREP_INSTRUMENTATION=$(scorep-config --prefix)/lib/scorep/scorep_instrument_function.so EXTRA_PROF_ENERGY=on EXTRA_PROF_GPU=off make -j