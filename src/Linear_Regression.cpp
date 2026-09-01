#include "Linear_Regression.hpp"

// cosntructors
LinearRegression::LinearRegression()
    : beta(std::nullopt),
    X(std::nullopt), 
    y(std::nullopt)
{

}

LinearRegression::LinearRegression(Matrix X, Matrix y)
    : beta(std::nullopt),
    X(addOnesCol(X)), 
    y(y)
{

}

// training
Matrix LinearRegression::fit(){
    if(! X.has_value())
        throw std::runtime_error("X is not populated yet");
    if(! y.has_value())
        throw std::runtime_error("y is not populated yet");

    beta = (X.value().transpose() * X.value()).solve(X.value().transpose() * y.value());

    return beta.value();
}

Matrix LinearRegression::fit(const Matrix& X, const Matrix& y){
    this->X = addOnesCol(X);
    this->y = y;
    beta = (this->X.value().transpose() * this->X.value()).solve(this->X.value().transpose() * y);

    return beta.value();
}

// predictions
Matrix LinearRegression::predict(const Matrix& X) const{
    if(!beta.has_value())
        throw std::runtime_error("beta is not populated yet");

    return addOnesCol(X) * beta.value();
}

// showing
void LinearRegression::graph(){

}

void LinearRegression::showBeta(){
    if(!beta.has_value())
        throw std::runtime_error("beta is not populated yet");

    beta.value().show();
}

// evaluate
double metrics::computeSSE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).hadamard(e).sum();
}

double metrics::computeSST(const Matrix& y){
    double y_mean = y.mean();
    Matrix e(y - y_mean);
    return e.hadamard(e).sum();
}

double metrics::computeMSE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).hadamard(e).mean();
}

double metrics::computeMAE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).apply(std::fabs).mean();
}

double metrics::computeRMSE(const Matrix& y, const Matrix& y_hat){
    return sqrt(metrics::computeMSE(y, y_hat));
}

double metrics::computeR2(const Matrix& y, const Matrix& y_hat){
    double SSE = computeSSE(y, y_hat), SST = computeSST(y);
    if(SST < epsilon)
        return 1.0;

    return 1 - (SSE/SST);
}

double metrics::computeAdjustedR2(const Matrix& y, const Matrix& y_hat, size_t features){
    size_t n = y.getRows(), k = features;
    double r2 = metrics::computeR2(y, y_hat);

    return 1 - (1-r2)*((double)(n-1)/(n-k-1));
}

double metrics::computeMaxError(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);

    double maxError = 0.0;
    for(size_t i = 0; i < e.getRows(); i++){
        maxError = std::max(maxError, std::fabs(e(i, 0)));
    }
    return maxError;
}

// helper functions
Matrix addOnesCol(const Matrix& A){
    size_t r = A.getRows(), c = A.getCols();
    Matrix result(r, c + 1);
    for(size_t i = 0; i < r; i++){
        for(size_t j = 0; j < c + 1; j++){
            if(j == c)
                result(i, j) = 1;
            else
                result(i, j) = A(i, j);
        }
    }

    return result;
}