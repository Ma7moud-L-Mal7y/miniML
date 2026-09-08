# minml

A machine learning library implemented from scratch in modern C++ — no ML frameworks, no external linear algebra dependencies. Every algorithm is built from first principles: the math is derived and implemented directly, not called from a library.

## Overview

minml implements core machine learning algorithms — regression, clustering, decision trees, and neural networks — on top of a custom linear algebra engine written in C++. The project is structured around two goals: numerical correctness and clean systems-level design, with performance work (SIMD, cache-aware memory layout) planned once the core algorithms are complete and tested.

## Technical Highlights

- **Custom linear algebra core** — matrix arithmetic, transposition, determinants, and LU decomposition implemented without external dependencies (no Eigen, no BLAS)
- **Dataset abstraction** — CSV loading, train/test splitting, shuffling, and normalization built on top of `Matrix`, with no external data-handling dependencies
- **Linear regression** — closed-form fitting via the normal equations, with a standard set of regression metrics (MSE, MAE, RMSE, R²)
- **Logistic regression** — gradient-based binary classification on top of the same `Matrix`/`Dataset` foundation
- **K-means clustering** — k-means++ initialization, Lloyd's algorithm, and convergence-based early stopping, with inertia tracked as a fitted metric
- **Decision tree** — from-scratch splitting and tree construction for classification/regression
- **RAII-based memory management** — no manual allocation/deallocation calls; resources are owned and released automatically by their containing objects
- **Class-based (OOP) model design** — each algorithm is implemented as its own self-contained class exposing `fit`/`predict`-style methods, with fitted state (coefficients, centroids, tree structure) held internally via `std::optional` rather than requiring it at construction
- **Binary model serialization** *(planned)* — custom format for saving and loading trained model weights

## Architecture

- Types that own memory (`Matrix`, `Dataset`) are RAII classes with defined constructors, destructors, and copy semantics.
- Model classes (`LinearRegression`, `LogisticRegression`, `KMean`, decision tree) hold their fitted state — coefficients, centroids, tree nodes — in `std::optional` members rather than requiring it at construction. A model can be default-constructed with just its hyperparameters and fit later, without sentinel values or a separate "unfitted" flag; `has_value()` on the relevant member (or an explicit `isFitted`/`isConverged`-style flag where warranted) answers whether it's ready for inference.
- Training data (`X`/`y`) is never stored as model state — it's passed explicitly to `fit`/`predict` as parameters and not retained afterward, keeping each model's memory footprint independent of dataset size once fitting completes.
- Lightweight, purely-aggregate outputs like `trainTest` (a paired train/test `Dataset` split) are plain structs, not classes — no need for encapsulation when there's no invariant to protect beyond what `Dataset` itself already guarantees.
- Evaluation logic is kept out of the model classes: metrics (regression: `computeMSE`, `computeMAE`, `computeRMSE`, `computeR2`; clustering: silhouette score, Davies-Bouldin index) live as free functions operating on `Matrix`/label data, so scoring a model doesn't require the model object itself.
- Shared numerical helpers used across algorithms (e.g. `addOnesCol` for the bias/intercept term, pairwise squared-distance computation) are standalone free functions rather than static class members, reusable across algorithms without inheritance.
- Each algorithm class is independent and self-contained, built and tested in isolation rather than through a shared polymorphic base — model-specific hyperparameters and internals live entirely within their own class.

## Project Structure

```
minml/
├── include/minml/     # Public headers
├── src/                # Implementation files
├── tests/              # Unit tests
├── examples/           # Sample datasets
```

## Roadmap

| Module | Status |
|---|---|
| Linear algebra core (`Matrix`) | In progress |
| Data loading (`Dataset`) | Done |
| Linear regression | Done |
| Logistic regression | Done |
| K-means clustering | Done |
| Decision tree | Done |
| Feedforward neural network (manual backprop) | In progress |
| Model serialization (`model_io`) | Planned |
| Unit test suite | In progress |

## Matrix API

```cpp
Matrix A(3, 3);
Matrix I = Matrix::Identity(3);

Matrix B = A + I;
Matrix C = A * B;
Matrix T = A.transpose();

double det = A.determinant();
luDecomposition lu = A.luDecompose();

double total = A.sum();
Matrix rowMeans = A.mean(0);
```

Full arithmetic (`+`, `-`, `*`, scalar operations), element-wise operations (`hadamard`, `apply`), row operations, and matrix property checks (`isSquare`, `isSymmetric`, `isSingular`) are supported. See [`include/minml/matrix.hpp`](include/minml/matrix.hpp) for the full interface.

## Dataset API

```cpp
Dataset data("data/housing.csv", /*labelNums=*/1);

data.head();
data.head(10);
data.showFeatureNames();

size_t n = data.getSampleNums();
size_t features = data.getFeatureNums();

data.shuffle();
data.normalize();

trainTest split = data.trainTestSplit(0.8 /* splitPoint */);
Matrix X_train = split.trainSample.getX();
Matrix y_train = split.trainSample.getY();
```

`Dataset` loads features and labels directly from a CSV file (or from existing `Matrix` objects), tracks feature/label names, and supports shuffling, normalization, and train/test splitting — either by index range or by a single split point. See [`include/minml/dataset.hpp`](include/minml/dataset.hpp) for the full interface.

## Linear Regression API

```cpp
LinearRegression model(X_train, y_train);
Matrix beta = model.fit();
model.showBeta();

Matrix y_hat = model.predict(X_test);

double mse = metrics::computeMSE(y_test, y_hat);
double rmse = metrics::computeRMSE(y_test, y_hat);
double r2 = metrics::computeR2(y_test, y_hat);
```

`LinearRegression` fits coefficients via the normal equations and exposes `predict` for inference on new data. The `metrics` namespace provides MSE, MAE, RMSE, and R² for evaluating predictions. See [`include/minml/linear_regression.hpp`](include/minml/linear_regression.hpp) for the full interface.

## K-Means API

```cpp
KMean model(/*k=*/3, /*maxIterations=*/1000, /*tolerance=*/1e-8);
model.fit(X_train);

std::vector<size_t> labels = model.predict(X_test);
double inertia = model.getInertia();
bool converged = model.isConverged();
```

`KMean` fits cluster centroids via k-means++ initialization followed by Lloyd's algorithm, iterating until either convergence (centroid movement below `tolerance`) or `maxIterations` is reached. `predict` assigns new points to the nearest fitted centroid. See [`include/minml/kmean.hpp`](include/minml/kmean.hpp) for the full interface.

```

## Author

Created and architected by **Mahmoud Mohamed**, including the project design, module structure, and technical direction outlined above.

## Contributing

Development is a collaborative effort, with **Abdelrahman Ahmed** contributing to implementation across several modules.

## License
MIT