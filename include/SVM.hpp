#pragma once
#include <optional>

#include "Matrix.hpp"


class SVM{
public:
    // constructors
    SVM();
    SVM(std::string kernel);

    // fit
    SVM& fit(const Matrix& trainX, const std::vector<bool>& trainY);

    // predict
    std::vector<int8_t> predict(const Matrix& testX);

    // showing
    void printKernelType();

private:
    // SVM essential parameters
    std::optional<Matrix> trainX;
    std::vector<double> alphas;
    std::vector<size_t> supportVectors;
    std::vector<bool> supportVectorsLabels;
    double bias;
    double C;

    std::string kernelType = "linear";

    // SVM parameters
    //  linear
    std::vector<double> w;

    //  polynomial
    size_t degree;
    double coef0;

    //  RBF & polynomial
    double gamma;
};