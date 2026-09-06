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

struct MulticlassConfusionMatrix {
    std::vector<std::vector<size_t>> matrix;
    size_t numClasses;
};

namespace multiclass_metrics {
    MulticlassConfusionMatrix computeConfusionMatrix(const Matrix& y, const Matrix& y_hat, size_t numClasses);
    double computeAccuracy(const Matrix& y, const Matrix& y_hat);
    std::vector<double> computePrecisionPerClass(const MulticlassConfusionMatrix& cm);
    std::vector<double> computeRecallPerClass(const MulticlassConfusionMatrix& cm);
    std::vector<double> computeF1PerClass(const MulticlassConfusionMatrix& cm);
    double computeMacroPrecision(const MulticlassConfusionMatrix& cm);
    double computeMacroRecall(const MulticlassConfusionMatrix& cm);
    double computeMacroF1(const MulticlassConfusionMatrix& cm);
    double computeWeightedPrecision(const MulticlassConfusionMatrix& cm);
    double computeWeightedRecall(const MulticlassConfusionMatrix& cm);
    double computeWeightedF1(const MulticlassConfusionMatrix& cm);
}