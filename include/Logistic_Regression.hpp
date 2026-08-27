#pragma once
#include "matrix.hpp"
#include <optional>

class LogisticRegression{
public:
    //constructors
    LogisticRegression();
    LogisticRegression(Matrix x,Matrix y, size_t maxIterations=1000, double learningRate=0.01);

    // training
    Matrix fit();
    Matrix fit(const Matrix& x, const Matrix& y);

    // predictions
    Matrix predict(const Matrix& X) const;
    Matrix predictClass(const Matrix& X) const;

    // showing
    void graph();
    void showBeta();

private:
    std::optional<Matrix> beta;
    std::optional<Matrix> x;
    std::optional<Matrix> y;
    size_t maxIterations;
    double learningRate;
};
namespace classification_metrics {
    double computeAccuracy(const Matrix& y, const Matrix& y_hat);
    double computeLogLoss(const Matrix& y, const Matrix& y_hat);
}
Matrix sigmoid(const Matrix &z);