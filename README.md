## MKF - Minkowski Functional calculation on 3D scalar fields.
Copyright: the project contributors, June-Oct 2019.

In order to concisely describe 3D spatial structure, measures of: Volume, Surface-area, Curvature and Topology are used.
Some important names relevant to this form of analysis are: Leonard Euler, Carl Friedrich Gauss, Hermann Minkowski, 
Hugo Hadwiger, Georges Matheron & Jean Serra, Joachim Ohser & Frank Mucklich. In other words, various concepts from 
classical analysis, theoretical completeness, algorithmic efficiency, statistical methods and robust estimation find
application here.


#### About/Objectives

The efficient computation of MKF on digital images begins with Matheron&Serra and their basic approach is adopted
in this implementation. Revised algorithms that allow efficient parallel execution on a modern vector processor
(such as a GPU) are developed with attention also paid to storage efficiency. The objective is to permit runtime
analysis of simulation data so that appropriate action can be taken without a separate operator-driven analysis
cycle. This may allow earlier completion of simulations, or perhaps just the prompt reporting of a problem.
There are potentially many situations when it is desireable to restart, pause or terminate a simulation.


#### Notes

OpenACC acceleration works for multicore (i.e. vectorised CPU) but not especially well (compiler unable to detect
parallelisable code despite effort to reduce/eliminate dependancies within packed binary map generation and processing).
OpenACC for GPU target so far produces nothing useful, hence CUDA implementation. OpenACC seems unable to translate
packed binary map generation into GPU code, despite efforts to expose/encourage parallelism. Presumably it could be
made to work with non-packed data but that would defeat memory-efficiency which is an important objective of this
project (GPU memory is already in short supply for applications involving 3D scalar fields). As useful progress has
been made with CUDA (and it has in any case greater potential for optimisation) future effort will be concentrated
there.

Packed Binary Map construction is straightforward using CUDA warp level primitives (SHFL instructions) to do the bit
merge. Surprisingly this seems to be the faster stage of processing even when double precision scalar fields are used.
Perhaps clever scheduling is able to hide the expected memory & instruction latency. The main issue with the PBM code
is managing the algorithm variants that handle data type and organisation, plus additional features such as descriptive
statistic/moment calculation that may be useful (or not). This needs to be abstracted (hopefully without impact upon 
computational efficiency) so that processing and data transfer steps can be overlapped to hide latency.

Construction of the Binary Pattern Frequency Distribution from the Packed Binary Map was first implemented with a fully
privatised distribution in shared memory per thread. This avoids the need for any atomic operations but limits the 
number of warps due to the high shared memory requirement and has noticeable setup and merge costs. Subsequent testing
has shown that shared memory atomic operations at the warp level offer twice the performance, although this figure is
likely dependant on the limited field width (256 elements) tested.
