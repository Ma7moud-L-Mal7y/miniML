#include "Logistic_Regression.hpp"
//constructors
LogisticRegression::LogisticRegression()
    : beta(std::nullopt),
    x(std::nullopt),
    y(std::nullopt),
    learningRate(0.01),
    maxIterations(1000)
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
//evaluate
ConfusionMatrix classification_metrics::confusionMatrix(const Matrix& y, const Matrix& y_hat){
    if (y.getRows() != y_hat.getRows() || y.getCols() != 1 || y_hat.getCols() != 1){
        throw std::invalid_argument("y and y_hat must be single-column matrices of the same length");
    }
    double tp=0,fp=0,tn=0,fn=0;
    for(size_t i=0;i<y.getRows();i++){
        bool actual = (y(i,0) == 1.0);
        bool predicted = (y_hat(i,0) == 1.0);
        if(actual && predicted) tp++;
        else if(!actual && !predicted) tn++;
        else if(actual && !predicted) fn++;
        else fp++;
    }
    return ConfusionMatrix{tp,fp,tn,fn};
}
double classification_metrics::computeAccuracy(const ConfusionMatrix& conmatrix){
    double total = conmatrix.tp+conmatrix.tn+conmatrix.fp+conmatrix.fn;
    if (total == 0) return 0.0;
    return (conmatrix.tn+conmatrix.tp)/total;
}
double classification_metrics::computePrecision(const ConfusionMatrix& conmatrix){
    if(conmatrix.tp+conmatrix.fp==0) return 0.0;
    return conmatrix.tp/(conmatrix.tp+conmatrix.fp);
}
double classification_metrics::computeRecall(const ConfusionMatrix& conmatrix){
    if(conmatrix.tp+conmatrix.fn==0) return 0.0;
    return conmatrix.tp/(conmatrix.tp+conmatrix.fn);
}
double classification_metrics::computeF1Score(const ConfusionMatrix& conmatrix){
    if(conmatrix.tp+conmatrix.fn+conmatrix.fp==0) return 0.0;
    return 2*conmatrix.tp/(2*conmatrix.tp+conmatrix.fn+conmatrix.fp);
}
double classification_metrics::computeLogLoss(const Matrix& y, const Matrix& y_hat) {
    if (y.getRows() != y_hat.getRows() || y.getCols() != 1 || y_hat.getCols() != 1){
        throw std::invalid_argument("y and y_hat must be single-column matrices of the same length");
    }
    size_t n=y.getRows();
    double totalLoss=0.0;
    for (size_t i = 0; i < n; i++) {
        double actual = y(i, 0);
        double p = y_hat(i, 0);
        if (p < epsilon) p = epsilon;
        if (p > 1.0 - epsilon) p = 1.0 - epsilon;
        double sampleLoss = -(actual * std::log(p) + (1.0 - actual) * std::log(1.0 - p));
        totalLoss += sampleLoss;
    }
    return totalLoss/n;
}
//helper functions
static double sigmoidFunction(double z){
    return 1.0/(1.0+std::exp(-z));
}
Matrix sigmoid(const Matrix &z){
    return z.apply(sigmoidFunction);
}
