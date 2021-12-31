<img src="https://raw.githubusercontent.com/rust-ml/linfa/1ba884495e4e2c44d2ef68b220da1f525ad518a5/mascot.svg" width="300"> <img src="https://smartcorelib.org/assets/logo/smartcore.png" width="300">

## About
Two heavy hitters have emerged in terms of `scikit-learn` analogous machine learning frameworks for rust: [`linfa`](https://rust-ml.github.io/linfa/) and [`smartcore`](https://smartcorelib.org/). Both provide access to a number of bread-and-butter algorithms that form the backbone of many analyses. This repository provides a comparison between the execution time of algorithms in these two machine learning frameworks. The full report is available [here](criterion/report/index.html), but summary violin plots are provided below.

## Considerations Besides Execution Time
Over the process of creating this benchmark study, a few additional differences between the libraries emerged.

### Documentation
The documentation for `smartcore` is a bit more consistent across algorithms. This may be due to the fact that it is maintained in a single crate.

### Dependencies
While `linfa` requires a BLAS/LAPACK backend (either `openblas`, `netblas`, or `intel-mkl`), `smartcore` does not. This allows `linfa` to take advantage of some additional optimization, but it limits portability.

## Results
### Regression
#### Linear Regression
_No customization needed to equate algorithms._

![](criterion/Linear%20Regression/report/violin.svg)

#### Elastic Net

![](criterion/Elastic%20Net/report/violin.svg)

### Classification
#### Logistic Regression

The `smartcore` implementation has no parameters, but the `linfa` settings were modified to align it with `smartcore` defaults:

- Gradient tolerance set to `1e-8`
- Maximum number of iterations set to `1000`

![](criterion/Logistic%20Regression/report/violin.svg)


### Clustering
#### K-Means

Since the two implementations use different convergence criteria, the number of max iterations was equated at a low value, and only 1 run of the `linfa` algorithm was permitted:

- Max iterations set to `10`
- Number of runs set to `1`

![](criterion/K-Means%20Clustering/report/violin.svg)

#### DBSCAN

![](criterion/DBSCAN%20Clustering/report/violin.svg)
