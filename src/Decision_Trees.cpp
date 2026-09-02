#include "Decision_Trees.hpp"


double DecisionTree::computeGini(const Matrix& y) const{
    size_t numRows= y.getRows();
    if(numRows==0) return 0.0;
    std::map<double, size_t> counts;
    for(size_t i=0;i<numRows;i++){
        counts[y(i,0)]++;
    }
    double sum=0.0;
    for(const auto& [label,count]:counts){
        double p=static_cast<double>(count)/numRows;
        sum+=p*p;
    }
    return 1-sum;
}

double DecisionTree::majorityClass(const Matrix& y)const{
    size_t rows= y.getRows();
    if(rows==0) return 0.0;
    std::map<double, size_t> counts;
    for(size_t i=0;i<rows;i++){
        counts[y(i,0)]++;
    }
    size_t maxCount=0;
    double bestLabel =0.0;
    for(const auto& [label,count]:counts){
        if(count>maxCount){
            maxCount=count;
            bestLabel=label;
        }
    }
    return bestLabel;
}
std::optional<DecisionTree::Split> DecisionTree::findBestSplit(const Matrix& X, const Matrix& y) const {
    size_t numRows = X.getRows();
    size_t numCols = X.getCols();

    if (numRows < minSamplesSplit) return std::nullopt;

    double parentGini = this->computeGini(y);
    double bestGain = -1.0;
    bool foundSplit = false;
    Split bestSplit{0, 0.0, 0.0};
    std::map<double, size_t> initialRightCounts;
    for (size_t i = 0; i < numRows; ++i) {
        initialRightCounts[y(i, 0)]++;
    }

    for (size_t c = 0; c < numCols; ++c) {
        std::vector<std::pair<double, double>> featureAndLabel(numRows);
        for (size_t r = 0; r < numRows; ++r) {
            featureAndLabel[r] = { X(r, c), y(r, 0) };
        }
        std::sort(featureAndLabel.begin(), featureAndLabel.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
            }
        );
        std::map<double, size_t> leftCounts;
        std::map<double, size_t> rightCounts=initialRightCounts;
        size_t leftSize=0,rightSize=numRows;
        for(size_t i=0;i<numRows-1;i++){
            double currentVal=featureAndLabel[i].first;
            double nextVal=featureAndLabel[i+1].first;
            double label=featureAndLabel[i].second;
            leftCounts[label]++;
            leftSize++;
            rightCounts[label]--;
            rightSize--;
            if(currentVal==nextVal)continue;
            if(leftSize<minSamplesLeaf||rightSize<minSamplesLeaf)continue;
            double leftSum=0.0;
            for(const auto&[lbl,count]:leftCounts){
                if(count==0)continue;
                double p=static_cast<double>(count)/leftSize;
                leftSum+=p*p;
            }
            double leftGini=1-leftSum;
            double rightSum=0.0;
            for(const auto&[lbl,count]:rightCounts){
                if(count==0)continue;
                double p=static_cast<double>(count)/rightSize;
                rightSum+=p*p;
            }
            double rightGini=1-rightSum;
            double weightedGini=(static_cast<double>(leftSize)/numRows)*leftGini+(static_cast<double>(rightSize)/numRows)*rightGini;
            double impuritydecrease=parentGini-weightedGini;
            if(impuritydecrease>bestGain){
                bestGain=impuritydecrease;
                bestSplit.featureIndex=c;
                bestSplit.threshold=(currentVal+nextVal)/2.0;
                bestSplit.impurityDecrease=impuritydecrease;
                foundSplit=true;
            }
        }
    }
    if (!foundSplit || bestGain <= 0.0) {
        return std::nullopt;
    }
    return bestSplit;
}
