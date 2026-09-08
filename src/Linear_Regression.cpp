#include "Linear_Regression.hpp"

// cosntructors
LinearRegression::LinearRegression()
    : beta(std::nullopt),
    X(std::nullopt), 
    y(std::nullopt)
{

}

LinearRegression::LinearRegression(Matrix& X, Matrix& y)
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