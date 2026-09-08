#pragma once
#include "Matrix.hpp"
#include <vector>

namespace regression_metrics{

    double computeSSE(const Matrix& y, const Matrix& y_hat);
    double computeSST(const Matrix& y);
    double computeMSE(const Matrix& y, const Matrix& y_hat);
    double computeMAE(const Matrix& y, const Matrix& y_hat);
    double computeRMSE(const Matrix& y, const Matrix& y_hat);
    double computeR2(const Matrix& y, const Matrix& y_hat);
    double computeAdjustedR2(const Matrix& y, const Matrix& y_hat, size_t features);
    double computeMaxError(const Matrix& y, const Matrix& y_hat);
    double computeMAPE(const Matrix& y, const Matrix& y_hat);

}

struct ConfusionMatrix { double tp, fp, tn, fn; };
namespace classification_metrics {
    ConfusionMatrix confusionMatrix(const Matrix& y, const Matrix& y_hat);
    double computeAccuracy(const ConfusionMatrix& conmatrix);
    double computePrecision(const ConfusionMatrix& conmatrix);
    double computeRecall(const ConfusionMatrix& conmatrix);
    double computeF1Score(const ConfusionMatrix& conmatrix);
    double computeLogLoss(const Matrix& y, const Matrix& y_hat);
}

namespace multiclass_metrics {
    Matrix computeConfusionMatrix(const Matrix& y, const Matrix& y_hat, size_t numClasses);
    double computeAccuracy(const Matrix& cm);
    std::vector<double> computePrecisionPerClass(const Matrix& cm);
    std::vector<double> computeRecallPerClass(const Matrix& cm);
    std::vector<double> computeF1PerClass(const Matrix& cm);
    double computeMacroPrecision(const Matrix& cm);
    double computeMacroRecall(const Matrix& cm);
    double computeMacroF1(const Matrix& cm);
    double computeWeightedPrecision(const Matrix& cm);
    double computeWeightedRecall(const Matrix& cm);
    double computeWeightedF1(const Matrix& cm);
}