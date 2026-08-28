#pragma once

#include "Matrix.hpp"
#include <optional>

class LinearRegression{
public:
    // constructors
    LinearRegression();
    LinearRegression(Matrix X, Matrix y);

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

// evaluate
namespace metrics{
    double computeSSE(const Matrix& y, const Matrix& y_hat);
    double computeSST(const Matrix& y, const Matrix& y_hat);
    double computeMSE(const Matrix& y, const Matrix& y_hat);
    double computeMAE(const Matrix& y, const Matrix& y_hat);
    double computeRMSE(const Matrix& y, const Matrix& y_hat);
    double computeR2(const Matrix& y, const Matrix& y_hat);
    double computeAdjustedR2(const Matrix& y, const Matrix& y_hat, size_t features);
    double computeMaxError(const Matrix& y, const Matrix& y_hat);
}

// helper functions
Matrix addOnesCol(const Matrix& A);