#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
#MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3 -Xcompiler -fopenmp -Xcompiler -mavx -arch sm_70  -g -G -ccbin=mpicxx

CXXFLAGS = $(CFLAGS) 

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) \
	$(MINIFE_MATRIX_TYPE) \
	-DMINIFE_RESTRICT=__restrict__ \
        -DHAVE_MPI #-DMPICH_IGNORE_CXX_SEEK
        #-DMINIFE_DEBUG=1
	

LDFLAGS=$(CFLAGS)
LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

export EXTRA_PROF_SCOREP_INSTRUMENTATION=$(shell scorep-config --prefix)/lib/scorep/scorep_instrument_function.so

CXX=nvcc-wrapper.sh
CC=nvcc-wrapper.sh


include make_targets
