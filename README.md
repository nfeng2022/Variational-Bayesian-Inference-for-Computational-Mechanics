# Varitional Bayesian inference for finite element problem 
A variational Bayesian network is implemented for predicting unobservable mehcnical response using measurmenable quantities. The training framework is based on a finite element analysis module implemented with tesnforflow functions.

The finite element analysis of the Cook's membrane example can be tested by running
```
fem_test.py
```
The deformed shape can be shown as below:
![sdof](./defromed_shape.pdf)  




## Dependency
- numpy
- scipy
- matplotlib
- tensorflow 2.13

## Benchmark
The unscented Kalman smoother is used to predict displacement, velocity and acceleration for a single-degree-of-freedom system under random harmonic loading. The detailed description for the target dynamic system is given as follows:
