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



