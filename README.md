# minml
 
A machine learning library implemented from scratch in modern C++ — no ML frameworks, no external linear algebra dependencies. Every algorithm is built from first principles: the math is derived and implemented directly, not called from a library.
 
## Overview
 
minml implements core machine learning algorithms — regression, clustering, and neural networks — on top of a custom linear algebra engine written in C++. The project is structured around two goals: numerical correctness and clean systems-level design, with performance work (SIMD, cache-aware memory layout) planned once the core algorithms are complete and tested.
 
## Technical Highlights
 
- **Custom linear algebra core** — matrix arithmetic, transposition, determinants, and LU decomposition implemented without external dependencies (no Eigen, no BLAS)
- **Dataset abstraction** — CSV loading, train/test splitting, shuffling, and normalization built on top of `Matrix`, with no external data-handling dependencies
- **Linear regression** — closed-form fitting via the normal equations, with a standard set of regression metrics (MSE, MAE, RMSE, R²)
- **RAII-based memory management** — no manual allocation/deallocation calls; resources are owned and released automatically by their containing objects
- **Composition over inheritance** — algorithms are implemented as free functions operating on data types, not as a polymorphic class hierarchy, avoiding the runtime and cache costs of virtual dispatch in numerical code
- **Binary model serialization** *(planned)* — custom format for saving and loading trained model weights
## Architecture
 
Design decisions favor patterns used in real numerical/performance-sensitive C++ codebases (e.g. Eigen) over textbook OOP:
 
- Types that own memory (`Matrix`, `Dataset`) are RAII classes with defined constructors, destructors, and copy semantics.
- Model classes hold their state (fitted coefficients, training data) in `std::optional` members rather than requiring it at construction — `LinearRegression` can be default-constructed and fit later, or constructed directly with `X`/`y` and fit immediately, without sentinel values or an "unfitted" flag.
- Lightweight, purely-aggregate outputs like `trainTest` (a paired train/test `Dataset` split) are plain structs, not classes — no need for encapsulation when there's no invariant to protect beyond what `Dataset` itself already guarantees.
- Evaluation logic is kept out of the model class entirely: metrics (`computeMSE`, `computeMAE`, `computeRMSE`, `computeR2`) live as free functions in a `metrics` namespace, so scoring a model doesn't require the model object itself — just two `Matrix`es.
- Shared numerical helpers used across algorithms (e.g. `addOnesCol`, for the bias/intercept term) are standalone free functions rather than static class members, keeping them reusable across future algorithms (logistic regression, etc.) without inheritance.
- As with `Matrix`, algorithm classes (`LinearRegression`, and future `LogisticRegression`, k-means) are independent, non-polymorphic types — there is no `Model` base class or virtual `fit`/`predict` interface, keeping the hot path free of virtual dispatch and each algorithm self-contained and easy to test in isolation.
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
| Logistic regression | Planned |
| K-means clustering | Planned |
| Feedforward neural network (manual backprop) | Planned |
| Model serialization (`model_io`) | Planned |
| Unit test suite | Planned |
 
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
 
## Build
 
```bash
# make test to run tests
```
 
## Author
 
Created and architected by **Mahmoud Mohamed**, including the project design, module structure, and technical direction outlined above.
 
## Contributing
 
Development is a collaborative effort, with **Abdelrahman Ahmed** contributing to implementation across several modules.
 
## License
 