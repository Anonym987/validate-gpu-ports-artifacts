export EXTRA_PROF_GPU=OFF

if [ ! -z ${ADD_CUDA_PATH+x} ]; then
export PATH=$PATH:$ADD_CUDA_PATH
fi

rm library.o liblibrary.so lib_extra_prof.so test_exe
../g++-wrapper.sh -c -fPIC -o library.o library.cpp
[ $? -eq 0 ] || exit $?
../g++-wrapper.sh -shared -o liblibrary.so library.o 
[ $? -eq 0 ] || exit $?
../g++-wrapper.sh ParallelKernelsTest.cpp -g -fopenmp -o test_exe -L. -llibrary -ldl
