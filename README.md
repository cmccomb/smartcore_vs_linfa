<div style="width:600px; height: 200px; display: table; text-align: center; margin:auto;">
    <img src="https://raw.githubusercontent.com/rust-ml/linfa/1ba884495e4e2c44d2ef68b220da1f525ad518a5/mascot.svg" width="150"> 
    <span style="display: table-cell; vertical-align: middle; font-size: 32px;">vs</span>
    <img src="https://smartcorelib.org/assets/logo/smartcore.png" width="150">
</div>

## About
[`linfa`](https://rust-ml.github.io/linfa/) and [`smartcore`](https://smartcorelib.org/) have emerged as two leading `scikit-learn`-analogous machine learning frameworks for Rust. Both provide access to a number of algorithms that form the backbone of machine learning analysis. This repository provides a comparison between the training time of algorithms in these two machine learning frameworks. The algorithms included are:

| Algorithm                     | Smartcore v2.0.0 | Linfa v5.0.0 | Benchmarked here? |
|:------------------------------|:-----------------|:-------------|:------------------|
| Linear Regression             | ✓                | ✓            | ✓                 |
| Ridge Regression              | ✓                |              |                   |
| LASSO Regression              | ✓                |              |                   |
| Decision Tree Regression      | ✓                |              |                   |
| Random Forest Regression      | ✓                |              |                   |
| Support Vector Regression     | ✓                | ✓            | ✓                 |
| KNN Regression                | ✓                |              |                   |
| Elastic Net Regression        | ✓                | ✓            | ✓                 |
| Partial Least Squares         |                  | ✓            |                   |
| Logistic Regression           | ✓                | ✓            | ✓                 |
| Decision Tree Classification  | ✓                | ✓            | ✓                 |
| Random Forest Classification  | ✓                |              |                   |
| Support Vector Classification | ✓                | ✓            | ✓                 |
| KNN Classification            | ✓                |              |                   |
| Gaussian Naive Bayes          | ✓                | ✓            | ✓                 |
| K-Means                       | ✓                | ✓            | ✓                 |
| DBSCAN                        | ✓                | ✓            | ✓                 |
| Hierarchical Clustering       |                  | ✓            |                   |
| Approximated DBSCAN           |                  | ✓            |                   |
| Gaussian Mixture Model        |                  | ✓            |                   |
| PCA                           | ✓                | ✓            | ✓                 |
| ICA                           |                  | ✓            |                   |
| SVD                           | ✓                |              |                   |
| t-SNE                         |                  | ✓            |                   |
| Diffusion Mapping             |                  | ✓            |                   |

The full report is available [here](criterion/report/index.html), but summary violin plots are provided below.

## Considerations Besides Execution Time
Over the process of creating this benchmark study, a few additional differences between the libraries emerged.

### Documentation
The documentation for `smartcore` is a bit more consistent across algorithms. This may be due to the fact that it is maintained in a single crate.

### Dependencies
While `linfa` requires a BLAS/LAPACK backend (either `openblas`, `netblas`, or `intel-mkl`), `smartcore` does not. This allows `linfa` to take advantage of some additional optimization, but it limits portability.

## Results
### Regression
#### [Linear Regression](criterion/Linear%20Regression/report/index.html)
_No customization needed to equate algorithms._

![](criterion/Linear%20Regression/report/violin.svg)

#### [Elastic Net](criterion/Elastic%20Net/report/index.html)

![](criterion/Elastic%20Net/report/violin.svg)


#### [Support Vector Regression](criterion/Support%20Vector%20Regression/report/index.html)

![](criterion/Support%20Vector%20Regression/report/violin.svg)



### Classification
#### [Logistic Regression](criterion/Logistic%20Regression/report/index.html)

The `smartcore` implementation has no parameters, but the `linfa` settings were modified to align it with `smartcore` defaults:

- Gradient tolerance set to `1e-8`
- Maximum number of iterations set to `1000`

![](criterion/Logistic%20Regression/report/violin.svg)

#### [Decision Tree](criterion/Decision%20Tree%20Classification/report/index.html)

![](criterion/Decision%20Tree%20Classification/report/violin.svg)

#### [Gaussian Naive Bayes](criterion/Gaussian%20Naive%20Bayes/report/index.html)

![](criterion/Gaussian%20Naive%20Bayes/report/violin.svg)

#### [Support Vector Classification](criterion/Support%20Vector%20Classification/report/index.html)

![](criterion/Support%20Vector%20Classification/report/violin.svg)



### Clustering
#### [K-Means](criterion/K-Means%20Clustering/report/index.html)

Since the two implementations use different convergence criteria, the number of max iterations was equated at a low value, and only 1 run of the `linfa` algorithm was permitted:

- Max iterations set to `10`
- Number of runs set to `1`

![](criterion/K-Means%20Clustering/report/violin.svg)

#### [DBSCAN](criterion/DBSCAN%20Clustering/report/index.html)

![](criterion/DBSCAN%20Clustering/report/violin.svg)

### Dimensionality Reduction
#### [PCA](criterion/PCA/report/index.html)
![](criterion/PCA/report/violin.svg)
