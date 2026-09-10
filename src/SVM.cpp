#include "SVM.hpp"

// constructors
SVM::SVM()
{

}

SVM::SVM(std::string kernel)
    : kernelType(kernel)
{
    if(kernel != "linear" && kernel != "poly" && kernel != "rbf"){
        throw std::runtime_error("choose one of the three supported kernels");
    }
}

// fit
SVM& SVM::fit(const Matrix& trainX, const std::vector<bool>& trainY){

}

// predict
std::vector<int8_t> SVM::predict(const Matrix& testX){
    // a_i * y_i K(x_i, x) + b
    size_t n = testX.getRows();
    size_t d = testX.getCols();
    size_t SVLen = supportVectors.size();
    std::vector<int8_t> result(n);

    if(kernelType == "linear"){
        for(size_t i = 0; i < n; i++){
            double sum = bias;
            for(size_t j = 0; j < w.size(); j++){
                sum += w[j] * testX(i, j);
            }
            result[i] = ((sum < 0) ? -1 : 1);
        }
    }
    else if(kernelType == "poly"){
        // K(x_i, x) = pow(gamma * x . x_i + coef0, d)
        for(size_t i = 0; i < n; i++){
            double sum = bias;
            for(size_t j = 0; j < SVLen; j++){
                size_t rowIdx = supportVectors[j];
                double alpha = alphas[rowIdx];
                int8_t label = 2 * supportVectorsLabels[j] - 1;

                double dot = 0.0;
                for(size_t k = 0; k < d; k++){
                    dot += trainX.value()(rowIdx, k) * testX(i, k);
                }

                double loc = alpha * label;
                for(size_t i = 0; i < degree; i++){
                    loc *= gamma * dot + coef0;
                }
                sum += loc;
            }
            result[i] = ((sum < 0) ? -1 : 1);
        }
    }
    else if(kernelType == "rbf"){
        // K(x_i, x) = e^(-gamma * ||x_i - x||^2)
        for(size_t i = 0; i < n; i++){
            double sum = bias;
            for(size_t j = 0; j < SVLen; j++){
                size_t rowIdx = supportVectors[j];
                double alpha = alphas[rowIdx];
                int8_t label = 2 * supportVectorsLabels[j] - 1;

                double distSqr = 0.0;
                for(size_t k = 0; k < d; k++){
                    double diff = trainX.value()(rowIdx, k) - testX(i, k);
                    distSqr += diff * diff;
                }

                sum += alpha * label * exp(-gamma * distSqr);
            }
            result[i] = ((sum < 0) ? -1 : 1);
        }
    }

    return result;
}

// showing
void SVM::printKernelType(){
    std::cout << kernelType << '\n';
}