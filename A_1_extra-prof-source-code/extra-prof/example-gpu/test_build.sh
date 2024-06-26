export EXTRA_PROF_ADVANCED_INSTRUMENTATION=OFF

if [ ! -z ${ADD_CUDA_PATH+x} ]; then
export PATH=$PATH:$ADD_CUDA_PATH
echo $PATH
fi

rm library.o liblibrary.so lib_extra_prof.so test_exe
../nvcc-wrapper.sh -c -Xcompiler -fPIC -o library.o library.cpp
[ $? -eq 0 ] || exit $?
../nvcc-wrapper.sh -shared -Xcompiler -fPIC -o liblibrary.so library.o library_kernels.cu
[ $? -eq 0 ] || exit $?
../nvcc-wrapper.sh kernels.cu copy_test.cu max_parallel_test.cu ParallelKernelsTest.cpp -g -Xcompiler -fopenmp -o test_exe -L. -L /usr/local/cuda/lib64 -I /usr/local/cuda/include -lcuda -llibrary -ldl
