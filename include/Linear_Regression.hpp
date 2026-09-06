#pragma once

#include "Matrix.hpp"
#include <optional>
#include "metrics.hpp"

class LinearRegression{
public:
    // constructors
    LinearRegression();
    LinearRegression(Matrix& X, Matrix& y);

    // training
    Matrix fit();
    Matrix fit(const Matrix& X, const Matrix& y);

    // predictions
    Matrix predict(const Matrix& X) const;

    // showing
    void graph();
    void showBeta();

private:
    // matrices
    std::optional<Matrix> beta;
    std::optional<Matrix> X;
    std::optional<Matrix> y;
};

// helper functions
Matrix addOnesCol(const Matrix& A);