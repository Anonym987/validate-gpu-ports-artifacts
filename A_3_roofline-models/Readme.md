Roofline Models
===============

This folder contains the roofline models collected on the DEEP-EST system with the [Empirical Roofline Toolkit](https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src).
We collected one on the Booster module using NVCC 11.7.0 with GCC 11.3.0 as host compiler which can be found in [Booster_NVCC_11.7.0](Booster_NVCC_11.7.0).
For that we used the following flags `-x cu -arch=sm_70 -O3`.
On the Cluster module we collected four roofline models:
* With the Intel Compiler 2021.4.0 and the `-O3 -fno-alias -fno-fnalias -xCore-AVX512 -qopt-zmm-usage=high -DERT_INTEL -qopenmp` flags: [Cluster_Intel_2021.4.0](Cluster_Intel_2021.4.0)
* With the GCC 11.3.0 and the `-g -O3 -fopenmp` flags: [Cluster_GCC_11.3.0](Cluster_GCC_11.3.0)
* With the GCC 11.3.0 and the `-g -O3 -mavx -fopenmp` flags: [Cluster_GCC_11.3.0_avx](Cluster_GCC_11.3.0_avx)
* With the GCC 11.3.0 and the `-g -O3 -mavx -ffast-math -fopenmp` flags: [Cluster_GCC_11.3.0_fastmath](Cluster_GCC_11.3.0_fastmath)

Each of these folders contains the configuration file for the Empirical Roofline Toolkit and the results folder, which contains all individual measurements and a PDF with the roofline model.