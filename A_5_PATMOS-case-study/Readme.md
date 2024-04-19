PATMOS Case Study
=================

The PATMOS case study comprises the measurements for the thread-scaling experiment of the CPU-only and the GPU versions
of PATMOS.
In addition, we provide the measurements of the node-scaling experiment of the GPU port.

The comparison results are directly provided in this folder as `.extra-p` files.

We have one results folder per measurement set, which contains the raw measurements and the corresponding performance
model (`*.extra-p` files) derived from the measurements.


Modeling
--------
To model the collected measurements, you should use Extra-P. You should select
_Open set of Extra-Prof files_ from the _File_ menu. Before the import of the measurements starts, make sure that you
select _weak parallel_ as the _Scaling type_ to ensure correct processing of the weak scaling experiments.
We used the median of the measurements for modeling to reduce the influence of noise; the result is shown in
the _Median Model_ set. Furthermore, we created aggregated models using the _Sum_ aggregation to
describe the performance behavior of the callpath, including all children.
We stored the resulting models in the folder of the respective variant using the _Save experiment_ action in the _File_
menu.

_For a more detailed explanation of the performance modeling, please refer to the Extra-P documentation._

Differential Modeling
----------------------

_The differential modeling works analogously to the miniFE case study._

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

_The analysis is performed analogously to the miniFE case study._

You can load the files containing the performance models into Extra-P using the _Open experiment_ action in the _File_
menu.

### Comparing at a specific point

The point of comparison is controlled with the sliders in the bottom left of the Extra-P window.

### Asymptotic comparison

The asymptotic comparison is performed on a model-by-model basis. You can execute it by selecting all the call paths to
compare and then choosing the _Calculate complexity comparison_ from the context menu.

_Note that the complexity comparison involves a symbolic computation, which might result in very long runtimes depending
on the length of the model._

### Plots

Plots can be selected through the _Plots_ menu. When the _Comparison Plot_ is used to view the hardware-adjusted time,
it will display the times of both experiment variants in addition to the hardware-adjusted time.
