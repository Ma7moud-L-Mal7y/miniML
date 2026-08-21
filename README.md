# minml

A machine learning library implemented from scratch in modern C++ — no ML frameworks, no external linear algebra dependencies. Every algorithm is built from first principles: the math is derived and implemented directly, not called from a library.

## Overview

minml implements core machine learning algorithms — regression, clustering, and neural networks — on top of a custom linear algebra engine written in C++. The project is structured around two goals: numerical correctness and clean systems-level design, with performance work (SIMD, cache-aware memory layout) planned once the core algorithms are complete and tested.

## Technical Highlights

- **Custom linear algebra core** — matrix arithmetic, transposition, determinants, and LU decomposition implemented without external dependencies (no Eigen, no BLAS)
- **RAII-based memory management** — no manual allocation/deallocation calls; resources are owned and released automatically by their containing objects
- **Composition over inheritance** — algorithms are implemented as free functions operating on data types, not as a polymorphic class hierarchy, avoiding the runtime and cache costs of virtual dispatch in numerical code
- **Binary model serialization** *(planned)* — custom format for saving and loading trained model weights

## Architecture

Design decisions favor patterns used in real numerical/performance-sensitive C++ codebases (e.g. Eigen) over textbook OOP:

- Types that own memory (`Matrix`, `Dataset`, `NeuralNet`) are RAII classes with defined constructors, destructors, and copy semantics.
- Algorithms (linear regression, logistic regression, k-means) are free functions rather than subclasses of a `Model` base — this keeps the hot path free of virtual function overhead and keeps each algorithm's implementation self-contained and easy to test in isolation.

## Project Structure

```
minml/
├── include/minml/     # Public headers
├── src/                # Implementation files
├── tests/              # Unit tests
├── examples/           # Example programs for each algorithm
└── data/                # Sample datasets
```

## Roadmap

| Module | Status |
|---|---|
| Linear algebra core (`Matrix`) | In progress |
| Data loading (`Dataset`) | Planned |
| Linear regression | Planned |
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

## Build

```bash
# make test to run tests
```

## Author

Created and architected by **Mahmoud Mohamed**, including the project design, module structure, and technical direction outlined above.

## Contributing

Development is a collaborative effort, with **Abdelrahman Ahmed** contributing to implementation across several modules.
## License
