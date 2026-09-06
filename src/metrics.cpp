#include "metrics.hpp"
double regression_metrics::computeSSE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).hadamard(e).sum();
}

double regression_metrics::computeSST(const Matrix& y){
    double y_mean = y.mean();
    Matrix e(y - y_mean);
    return e.hadamard(e).sum();
}

double regression_metrics::computeMSE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).hadamard(e).mean();
}

double regression_metrics::computeMAE(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);
    return (e).apply(std::fabs).mean();
}

double regression_metrics::computeRMSE(const Matrix& y, const Matrix& y_hat){
    return sqrt(regression_metrics::computeMSE(y, y_hat));
}

double regression_metrics::computeR2(const Matrix& y, const Matrix& y_hat){
    double SSE = computeSSE(y, y_hat), SST = computeSST(y);
    if(SST < epsilon)
        return 1.0;

    return 1 - (SSE/SST);
}

double regression_metrics::computeAdjustedR2(const Matrix& y, const Matrix& y_hat, size_t features){
    size_t n = y.getRows(), k = features;
    double r2 = regression_metrics::computeR2(y, y_hat);

    return 1 - (1-r2)*((double)(n-1)/(n-k-1));
}

double regression_metrics::computeMaxError(const Matrix& y, const Matrix& y_hat){
    Matrix e(y - y_hat);

    double maxError = 0.0;
    for(size_t i = 0; i < e.getRows(); i++){
        maxError = std::max(maxError, std::fabs(e(i, 0)));
    }
    return maxError;
}

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