miniFE Case Study
=================

The miniFE case study comprises five versions of miniFE.

* A CPU-only variant based on OpenMP
* Three versions of a naive GPU port that is based on the CPU-only version with added GPU offloading
* The reference GPU implementation using CUDA

The `*.extra-p` files containing the differential models for these versions are located in this folder.  
However, each of these versions also has its own folder, which contains the results and the sources for that version.
The results folders contain the measurements and the scripts used to collect them. 
The corresponding performance models (`*.extra-p` files) derived from the measurements are also in these folders.
The sources folder contains the source code used for the particular variant and the makefile necessary to compile it.



Compiling
---------
When compiling the variants, we assume for all variants, that the Extra-Profiler wrappers, Score-P, including its
wrappers, the compilers, and the MPI compiler wrappers are in the PATH.
Then, executing `make` in the respective source folder builds the instrumented executable of the respective version.

Running
-------
The experiments are run as slurm jobs. So, the `batch_template.sh` files must be adjusted to fit the target system.
After that, the jobs can be submitted by executing the `start-batch.sh` files in the respective results folder.
This will create and submit all jobs to collect the measurement data for modeling.

Modeling
--------
To model the collected measurements, you should use Extra-P. If the measurements were obtained with Score-P, you can use
the _Open set of CUBE files_ option from the _File_ menu of Extra-P. If the Extra-Profiler was used, you should select
_Open set of Extra-Prof files_ from the _File_ menu. Before the import of the measurements starts, make sure that you
select _weak parallel_ as the _Scaling type_ to ensure correct processing of the weak scaling experiments.
We turned off the use of negative coefficients during modeling for more consistent results; the result of this is shown
in the _Positive Coefficients Model_ set.
For the energy measurements, we applied the _Calculate Total Energy_ aggregation, which adds the _Energy_ metric that
represents the combined energy usage of CPU and GPU.
We stored the resulting models in the folder of the respective variant using the _Save experiment_ action in the _File_
menu.

_For a more detailed explanation of the performance modeling, please refer to the Extra-P documentation._


Differential Modeling
----------------------

For the differential modeling, you need to open the baseline experiment. Then you can select _Compare with
experiment_ from the _File_ menu. The comparison assistant will guid you through the process.

First you load the experiment which you want to compare. Then you name each experiment, so that you have a way to
distinguish them later on. After that the models, that should be compared to each other are mapped. The same is done
with the parameters. As the penultimate step we need to select the mapping provider. We select the _Smart Matcher_ to
use the call-tree mapping process that finds a unified call tree.

Once the comparison is finished, we can create the projection for the hardware-adjusted runtime. We have to load the
matching empirical roofline toolkit (ERT) file for both experiments. Then we can start the projection, the other
settings are configured automatically.

By default, Extra-P always shows the models for both versions separately. When the _Show differences_ option is checked,
it will display the differential performance modeling.

Analysis
--------

You can load the files containing the performance models into Extra-P using the _Open experiment_ action in the _File_
menu.

### Comparing at a specific point

The point of comparison is controlled with the sliders in the bottom left of the Extra-P window.

### Asymptotic comparison

The asymptotic comparison is performed on a model-by-model basis. You can execute it by selecting all the call paths to
compare and then choosing the _Calculate complexity comparison_ from the context menu.

_Note that the complexity comparison involves a symbolic computation, which might result in very long runtimes depending
on the lenght of the model._

### Plots

Plots can be selected through the _Plots_ menu. When the _Comparison Plot_ is used to view the hardware-adjusted time,
it will display the times of both experiment variants in addition to the hardware-adjusted time.
