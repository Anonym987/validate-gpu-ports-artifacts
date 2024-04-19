Example CPU
===========
This example can be build with the Extra-Profiler's GCC wrapper by executing the `test_build.sh` script, that takes care of compiling the two dynamic libraries and the application. This example requires the GCC compiler to be in the PATH and support for OpenMP. You can run the example by executing the `test_exe` with an argument, that controls the how many iterations will run:

```sh
./test_exe 4
```

Once the run is complete you should find a `profile.extra-prof.msgpack` file in a new `extra_prof_*` folder.