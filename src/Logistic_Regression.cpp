#include "Logistic_Regression.hpp"

//constructors
LogisticRegression::LogisticRegression()
    : beta(std::nullopt),
    x(std::nullopt),
    y(std::nullopt)
{
    
}
LogisticRegression::LogisticRegression(Matrix x,Matrix y,size_t maxIterations,double learningRate)
    : beta(std::nullopt),
    x(x),
    y(y),
    maxIterations(maxIterations),
    learningRate(learningRate)
{

}