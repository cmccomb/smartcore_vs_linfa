## About
A comparison between the execution time of algorithms in the [`linfa`](https://rust-ml.github.io/linfa/) and [`smartcore`](https://smartcorelib.org/) machine learning frameworks. The full report is available [here](criterion/report/index.html), but summary violin plots are provided below.

## Considerations Besides Execution Time
### Documentation

### Dependencies
While `linfa` requires a BLAS/LAPACK backend (either `openblas`, `netblas`, or `intel-mkl`), `smartcore` does not. This allows `linfa` to take advantage of some additional optimization, but it limits portability.

## Results
### Linear Regression
_No customization to equate algorithms._
![](criterion/Linear%20Regression/report/violin.svg)

### Logistic Regression
![](criterion/Logistic%20Regression/report/violin.svg)

### K-Means Clustering
![](criterion/K-Means%20Clustering/report/violin.svg)
