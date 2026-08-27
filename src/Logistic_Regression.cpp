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

// training

Matrix LogisticRegression::fit(const Matrix& X, const Matrix& Y) {
    if (Y.getCols() != 1)
        throw std::invalid_argument("LogisticRegression expects a single-column label matrix");
    this->x = X;
    this->y = Y;
    size_t colNums = x.value().getCols();
    size_t sampleNums = x.value().getRows();
    beta = Matrix(colNums, 1);
    
    const double tolerance = 1e-6;
    
    for(size_t i = 0; i < maxIterations; i++) {
        Matrix predictions = sigmoid(x.value() * beta.value());
        Matrix error = predictions - y.value();
        Matrix gradient = (x.value().transpose() * error) * (1.0 / sampleNums);
        double gradNorm = gradient.hadamard(gradient).sum();
        if (gradNorm < tolerance * tolerance) {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        
        beta = beta.value() - gradient * learningRate;
    }
    return beta.value();
}
Matrix LogisticRegression ::fit(){
    if(! x.has_value())
        throw std::runtime_error("x is not populated yet");
    if(! y.has_value())
        throw std::runtime_error("y is not populated yet");
    return fit(x.value(),y.value());
    
}
// predictions
Matrix LogisticRegression::predict(const Matrix& X) const{
    if(!beta.has_value())
        throw std::runtime_error("beta is not populated yet");
    return sigmoid(X* beta.value());
}
Matrix LogisticRegression::predictClass(const Matrix& X) const{
    Matrix predictions=this->predict(X);
    size_t sampleNums=predictions.getRows();
    Matrix classify=Matrix(sampleNums,1);
    for(size_t i=0;i<sampleNums;i++){
        if(predictions(i,0)>0.5){
            classify(i,0)=1;
        }
        else{
            classify(i,0)=0;
        }
    }
    return classify;
}
//show
void LogisticRegression::showBeta(){
    if(!beta.has_value())
        throw std::runtime_error("beta is not populated yet");
    beta.value().show();
}
void LogisticRegression::graph(){
    
}

static double sigmoidFunction(double z){
    return 1.0/(1.0+std::exp(-z));
}
Matrix sigmoid(const Matrix &z){
    return z.apply(sigmoidFunction);
}
