#include "Decision_Trees.hpp"

//constructor
DecisionTree::DecisionTree()
    : type(TreeType::Classification), maxDepth(10), minSamplesSplit(2), minSamplesLeaf(1)
{

}
DecisionTree::DecisionTree(TreeType type, size_t maxDepth, size_t minSamplesSplit, size_t minSamplesLeaf)
    : type(type), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), minSamplesLeaf(minSamplesLeaf) 
{

}
//training
void DecisionTree::fit(const Matrix& X, const Matrix& y) {
    this->root = this->buildTree(X, y, 0);
}
//prediction
Matrix DecisionTree::predict(const Matrix& X) const{
    Matrix result(X.getRows(),1);
    for(size_t i=0;i<X.getRows();++i){
        result(i,0)=predictSingle(X,i,root.get());
    }
    return result;
}

//build tree
std::unique_ptr<DecisionTree::Node> DecisionTree::buildTree(const Matrix& X, const Matrix& y, size_t depth) const{
    auto node=std::make_unique<Node>();
    size_t numRows=X.getRows();
    size_t numCols=X.getCols();
    if(depth>=maxDepth||numRows<minSamplesSplit||(this->computeImpurity(y)==0)){
        node->isLeaf=true;
        node->value=this->computeLeafValue(y);
        return node;
    }
    auto bestSplit=this->findBestSplit(X,y);
    if(bestSplit==std::nullopt){
        node->isLeaf=true;
        node->value=this->computeLeafValue(y);
        return node;
    }
    const auto& split=bestSplit.value();
    node->isLeaf=false;
    node->splitFeatureIndex=split.featureIndex;
    node->splitThreshold=split.threshold;
    std::vector<size_t> leftIndices;
    std::vector<size_t> rightIndices;
    leftIndices.reserve(numRows);
    rightIndices.reserve(numRows);
    for(size_t i=0;i<numRows;++i){
        if(X(i,node->splitFeatureIndex)<node->splitThreshold){
            leftIndices.push_back(i);
        }
        else{
            rightIndices.push_back(i);
        }
    }
    if (leftIndices.empty() || rightIndices.empty()) {
        node->isLeaf = true;
        node->value=this->computeLeafValue(y);
        return node;
    }
    Matrix X_left(leftIndices.size(),numCols);
    Matrix Y_left(leftIndices.size(),1);
    for(size_t i=0;i<leftIndices.size();++i){
        size_t r=leftIndices[i];
        for(size_t c=0;c<numCols;++c){
            X_left(i,c)=X(r,c);
        }
        Y_left(i,0)=y(r,0);
    }
    Matrix X_right(rightIndices.size(),numCols);
    Matrix Y_right(rightIndices.size(),1);
    for(size_t i=0;i<rightIndices.size();++i){
        size_t r=rightIndices[i];
        for(size_t c=0;c<numCols;++c){
            X_right(i,c)=X(r,c);
        }
        Y_right(i,0)=y(r,0);
    }
    node->left=buildTree(X_left,Y_left,depth+1);
    node->right=buildTree(X_right,Y_right,depth+1);
    return node;
}

// tree splitting and leaf operaations
std::optional<DecisionTree::Split> DecisionTree::findBestSplit(const Matrix& X, const Matrix& y) const {
    size_t numRows = X.getRows();
    size_t numCols = X.getCols();

    if (numRows < minSamplesSplit) return std::nullopt;

    double parentImpurity = this->computeImpurity(y);
    double bestGain = -1.0;
    bool foundSplit = false;
    Split bestSplit{0, 0.0, 0.0};
    size_t numClasses = 0;
    std::vector<size_t> initialRightCounts;
    
    if (type == TreeType::Classification) {
        for (size_t i = 0; i < numRows; ++i) {
            size_t c = static_cast<size_t>(y(i, 0));
            if (c >= numClasses) numClasses = c + 1;
        }
        initialRightCounts.assign(numClasses, 0);
        for (size_t i = 0; i < numRows; ++i) {
            initialRightCounts[static_cast<size_t>(y(i, 0))]++;
        }
    }
    double initialTotalSum = 0.0;
    double initialTotalSumSq = 0.0;
    if (type == TreeType::Regression) {
        for (size_t r = 0; r < numRows; ++r) {
            const double val = y(r, 0);
            initialTotalSum += val;
            initialTotalSumSq += val * val;
        }
    }
    std::vector<std::pair<double, double>> featureAndLabel(numRows);
    std::vector<size_t> leftCounts(numClasses, 0);
    std::vector<size_t> rightCounts(numClasses, 0);
    for (size_t c = 0; c < numCols; ++c) {
        for (size_t r = 0; r < numRows; ++r) {
            featureAndLabel[r] = { X(r, c), y(r, 0) };
        }
        std::sort(featureAndLabel.begin(), featureAndLabel.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
            }
        );
        if (type == TreeType::Classification) {
            std::fill(leftCounts.begin(), leftCounts.end(), 0);
            rightCounts = initialRightCounts;
        }
        double leftSum = 0.0, leftSumSq = 0.0;
        double rightSum = initialTotalSum, rightSumSq = initialTotalSumSq;
        size_t leftSize=0,rightSize=numRows;
        for(size_t i=0;i<numRows-1;i++){
            double currentVal=featureAndLabel[i].first;
            double nextVal=featureAndLabel[i+1].first;
            double label=featureAndLabel[i].second;
            leftSize++;
            rightSize--;
            if(type==TreeType::Classification){
                const size_t classIdx = static_cast<size_t>(label);
                leftCounts[classIdx]++;
                rightCounts[classIdx]--;
            }
            else{
                leftSum+=label;
                leftSumSq+=label*label;
                rightSum-=label;
                rightSumSq-=label*label;
            }
            if(currentVal==nextVal)continue;
            if(leftSize<minSamplesLeaf||rightSize<minSamplesLeaf)continue;
            double leftImpurity = 0.0;
            double rightImpurity = 0.0;
            if(type==TreeType::Classification){
                double lSum=0.0;
                for (size_t k = 0; k < numClasses; ++k) {
                    if (leftCounts[k] == 0) continue;
                    const double p = static_cast<double>(leftCounts[k]) / leftSize;
                    lSum += p * p;
                }
                leftImpurity=1-lSum;
                double rSum=0.0;
                for (size_t k = 0; k < numClasses; ++k) {
                    if (rightCounts[k] == 0) continue;
                    const double p = static_cast<double>(rightCounts[k]) / rightSize;
                    rSum += p * p;
                }
                rightImpurity=1-rSum;
            }
            else{
                const double leftMean = leftSum / leftSize;
                leftImpurity = (leftSumSq / leftSize) - (leftMean * leftMean);
                if (leftImpurity < 0.0) leftImpurity = 0.0;

                const double rightMean = rightSum / rightSize;
                rightImpurity = (rightSumSq / rightSize) - (rightMean * rightMean);
                if (rightImpurity < 0.0) rightImpurity = 0.0;
            }
            double weightedImpurity=(static_cast<double>(leftSize)/numRows)*leftImpurity+(static_cast<double>(rightSize)/numRows)*rightImpurity;
            double impuritydecrease=parentImpurity-weightedImpurity;
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
double DecisionTree::meanValue(const Matrix& y) const{
    return y.mean();
}
double DecisionTree::computeVariance(const Matrix&y)const{
    double mean=y.mean();
    size_t rows=y.getRows();
    if (rows == 0) return 0.0;
    double sum=0;
    for(size_t i=0;i<rows;++i){
        double var= (y(i,0)-mean)*(y(i,0)-mean);
        sum+=var;
    }
    return sum/static_cast<double>(rows);
}
double DecisionTree::computeImpurity(const Matrix& y) const {
    return (type == TreeType::Classification) ? computeGini(y) : computeVariance(y);
}

double DecisionTree::computeLeafValue(const Matrix& y) const {
    return (type == TreeType::Classification) ? majorityClass(y) : meanValue(y);
}
double DecisionTree::predictSingle(const Matrix& X, size_t rowIndex, const Node* node) const{
    if(node->isLeaf) return node->value;
    if(X(rowIndex,node->splitFeatureIndex)<node->splitThreshold){
        return predictSingle(X,rowIndex,node->left.get());
    }
    else{
        return predictSingle(X,rowIndex,node->right.get());
    }
}