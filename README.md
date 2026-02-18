# Varitional Bayesian inference for finite element problem 
A variational Bayesian network is implemented for predicting unobservable mehcnical response using measurmenable quantities. The training framework is based on a finite element analysis module implemented with tesnforflow functions.

The finite element analysis can be tested using the Cook's membrane example and the deformed shape can be shown as below:



## Dependency
- numpy
- scipy
- matplotlib
- tensorflow 2.13

## Benchmark
The unscented Kalman smoother is used to predict displacement, velocity and acceleration for a single-degree-of-freedom system under random harmonic loading. The detailed description for the target dynamic system is given as follows:
