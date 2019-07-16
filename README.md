## MKF - Minkowski Functional calculation on 3D scalar fields.
Copyright: the project contributors, June-July 2019.

In order to concisely describe 3D spatial structure, measures of: volume, area, curvature and topology are used.
Some eminent names in the "recent" lineage of this form of analysis are: Hermann Minkowski, Hugo Hadwiger, 
Georges Matheron & Jean Serra, Joachim Ohser & Frank Mucklich. (While at the root of it all are the intellectual titans
Leonard Euler and Carl Freidrich Gauss. Caveat Emptor: There is a great deal of mathematics to digest before deep
understanding is realised.)


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
OpenACC for GPU target so far produces nothing useful, hence CUDA implementation. 

CUDA version greatly improved by inclusion of *Synchronise() calls - yields deterministic (albeit incorrect) results in
all cases. There is a consistent undercount of binary patterns and this (weirdly) changes with image content (size of ball).
Suspect this is an internal pattern chunk boundary problem...

