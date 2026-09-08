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
double regression_metrics::computeMAPE(const Matrix& y, const Matrix& y_hat){
    size_t rows = y.getRows();
    double sum = 0.0;
    size_t count = 0;
    for(size_t r = 0; r < rows; ++r){
        if (std::fabs(y(r,0)) < epsilon) continue;
        double error = std::fabs(y(r,0) - y_hat(r,0)) / y(r,0);
        sum += error;
        count++;
    }
    if (count == 0) return 0.0;
    return sum * 100.0 / count;
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

Matrix multiclass_metrics::computeConfusionMatrix(const Matrix& y, const Matrix& y_hat, size_t numClasses){
    Matrix cm(numClasses,numClasses,0);
    size_t rows=y.getRows();
    for(size_t r=0;r<rows;++r){
        size_t predicted=y_hat(r,0);
        size_t actual=y(r,0);
        cm(actual,predicted)++;
    }
    return cm;
}
double multiclass_metrics::computeAccuracy(const Matrix& cm){
    size_t numclasses=cm.getCols();
    size_t total=0,correct=0;
    for(size_t i=0;i<numclasses;i++){
        for(size_t j=0;j<numclasses;j++){
            if(i==j) correct+= cm(i,j);
            total+=cm(i,j);
        }
    }
    if (total == 0) return 0.0;
    return static_cast<double>(correct)/total;
}
std::vector<double> multiclass_metrics::computePrecisionPerClass(const Matrix& cm){
    std::vector<double> precision;
    size_t numClasses=cm.getRows();
    precision.resize(numClasses);
    for(size_t i=0;i<numClasses;i++){
        double sumCol=0,predicted=0;
        for(size_t j=0;j<numClasses;j++){
            sumCol+=cm(j,i);
            if(i==j)predicted+=cm(j,i);
        }
        precision[i] = (sumCol == 0) ? 0.0 : predicted/ sumCol;
    }
    return precision;
}
std::vector<double> multiclass_metrics::computeRecallPerClass(const Matrix& cm){
    std::vector<double> recall;
    size_t numClasses=cm.getRows();
    recall.resize(numClasses);
    for(size_t i=0;i<numClasses;i++){
        double sumRow=0,predicted=0;
        for(size_t j=0;j<numClasses;j++){
            sumRow+=cm(i,j);
            if(i==j)predicted+=cm(i,j);
        }
        recall[i] = (sumRow == 0) ? 0.0 : predicted/ sumRow;
    }
    return recall;
}
std::vector<double> multiclass_metrics::computeF1PerClass(const Matrix& cm){
    std::vector<double> F1score;
    size_t numClasses=cm.getRows();
    F1score.resize(numClasses);
    for(size_t i=0;i<numClasses;i++){
        double sumRow=0,sumCol=0,predicted=0;
        for(size_t j=0;j<numClasses;j++){
            sumRow+=cm(i,j);
            sumCol+=cm(j,i);
            if(i==j)predicted+=cm(i,j);
        }
        F1score[i] = (sumRow+sumCol == 0) ? 0.0 : 2*predicted/ (sumRow+sumCol);
    }
    return F1score;
}
double multiclass_metrics::computeMacroPrecision(const Matrix& cm){
    std::vector<double> precision=computePrecisionPerClass(cm);
    double sum=0;
    for(size_t i=0;i<precision.size();i++){
        sum+=precision[i];
    }
    return sum/precision.size();
}
double multiclass_metrics::computeMacroRecall(const Matrix& cm){
    std::vector<double> Recall=computeRecallPerClass(cm);
    double sum=0;
    for(size_t i=0;i<Recall.size();i++){
        sum+=Recall[i];
    }
    return sum/Recall.size();
}
double multiclass_metrics::computeMacroF1(const Matrix& cm){
    std::vector<double> F1=computeF1PerClass(cm);
    double sum=0;
    for(size_t i=0;i<F1.size();i++){
        sum+=F1[i];
    }
    return sum/F1.size();
}
double multiclass_metrics::computeWeightedPrecision(const Matrix& cm) {
    size_t numClasses = cm.getRows();
    double totalSamples = 0.0;
    double weightedSum = 0.0;

    for (size_t i = 0; i < numClasses; ++i) {
        double rowSum = 0.0;
        double colSum = 0.0;
        double tp = cm(i, i);
        for (size_t j = 0; j < numClasses; ++j) {
            rowSum += cm(i, j);
            colSum += cm(j, i);
        }
        double precision_i = (colSum == 0.0) ? 0.0 : tp / colSum;
        weightedSum += precision_i * rowSum;
        totalSamples += rowSum;
    }
    return (totalSamples == 0.0) ? 0.0 : weightedSum / totalSamples;
}

double multiclass_metrics::computeWeightedRecall(const Matrix& cm) {
    size_t numClasses = cm.getRows();
    double totalSamples = 0.0;
    double totalTP = 0.0;
    for (size_t i = 0; i < numClasses; ++i) {
        totalTP += cm(i, i);
        for (size_t j = 0; j < numClasses; ++j) {
            totalSamples += cm(i, j);
        }
    }
    return (totalSamples == 0.0) ? 0.0 : totalTP / totalSamples;
}

double multiclass_metrics::computeWeightedF1(const Matrix& cm) {
    size_t numClasses = cm.getRows();
    double totalSamples = 0.0;
    double weightedSum = 0.0;
    for (size_t i = 0; i < numClasses; ++i) {
        double rowSum = 0.0;
        double colSum = 0.0;
        double tp = cm(i, i);
        for (size_t j = 0; j < numClasses; ++j) {
            rowSum += cm(i, j);
            colSum += cm(j, i);
        }
        double f1_i = (rowSum + colSum == 0.0) ? 0.0 : (2.0 * tp) / (rowSum + colSum);
        weightedSum += f1_i * rowSum;
        totalSamples += rowSum;
    }
    return (totalSamples == 0.0) ? 0.0 : weightedSum / totalSamples;
}

