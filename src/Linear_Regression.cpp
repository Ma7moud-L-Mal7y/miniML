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
    X(X), 
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
    this->X = X;
    this->y = y;
    beta = (X.transpose() * X).solve(X.transpose() * y);

    return beta.value();
}

// predictions
Matrix LinearRegression::predict(const Matrix& X) const{
    if(!beta.has_value())
        throw std::runtime_error("beta is not populated yet");

    return X * beta.value();
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
double metrics::computeMSE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).hadamard(e).mean();
}

double metrics::computeMAE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).apply(fabs).mean();
}

double metrics::computeRMSE(const Matrix& y, const Matrix& y_hat){
    return sqrt(metrics::computeMSE(y, y_hat));
}

double metrics::computeR2(const Matrix& y, const Matrix& y_hat){

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