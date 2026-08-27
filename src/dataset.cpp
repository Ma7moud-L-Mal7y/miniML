#include "Dataset.hpp"

//constructors

Dataset::Dataset(const Matrix& x,const Matrix& y)
    : featureMatrix(x), labelMatrix(y)
{

}
bool Dataset::isNumericString(const std::string& s) const {
    if (s.empty()) return false;
    try {
        size_t pos;
        std::stod(s, &pos);
        return pos == s.size();
    } 
    catch (...) {
        return false;
    }
}

Dataset::Dataset(const std::string& filepath, size_t labelNums)
    : featureMatrix(0,0), labelMatrix(0,0)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + filepath);
    }
    std::vector<std::string> allNames;
    std::string line;
    std::getline(file, line);
    std::stringstream headerstream(line);
    std::string name;
    while (std::getline(headerstream, name, ',')) {
        allNames.push_back(name);
    }
    size_t allCols = allNames.size();
    size_t featureColsOriginal = allCols - labelNums;
    std::vector<std::vector<std::string>> allRows;
    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::string piece;
        std::vector<std::string> row;
        while (std::getline(linestream, piece, ',')) {
            row.push_back(piece);
        }
        allRows.push_back(row);
    }
    size_t rowNums = allRows.size();
    std::vector<bool> isNumericCol(featureColsOriginal, true);
    for (size_t c = 0; c < featureColsOriginal; c++) {
        for (size_t r = 0; r < rowNums; r++) {
            if (!isNumericString(allRows[r][c])) {
                isNumericCol[c] = false;
                break;
            }
        }
    }
    std::vector<std::vector<std::string>> categories(featureColsOriginal);
    for (size_t c = 0; c < featureColsOriginal; c++) {
        if (isNumericCol[c]) continue;
        for (size_t r = 0; r < rowNums; r++) {
            const std::string& val = allRows[r][c];
            bool seen = false;
            for (const auto& existing : categories[c]) {
                if (existing == val) { seen = true; break; }
            }
            if (!seen) categories[c].push_back(val);
        }
    }
    std::vector<size_t> startIndex(featureColsOriginal);
    size_t expandedFeatureCount = 0;
    for (size_t c = 0; c < featureColsOriginal; c++) {
        startIndex[c] = expandedFeatureCount;
        if (isNumericCol[c]) {
            featureNames.push_back(allNames[c]);
            expandedFeatureCount++;
        } else {
            for (const auto& category : categories[c]) {
                featureNames.push_back(allNames[c] + "_" + category);
            }
            expandedFeatureCount += categories[c].size();
        }
    }
    for (size_t c = featureColsOriginal; c < allCols; c++) {
        labelNames.push_back(allNames[c]);
    }
    featureMatrix = Matrix(rowNums, expandedFeatureCount);
    labelMatrix = Matrix(rowNums, labelNums);
    for (size_t r = 0; r < rowNums; r++) {
        for (size_t c = 0; c < featureColsOriginal; c++) {
            if (isNumericCol[c]) {
                featureMatrix(r, startIndex[c]) = std::stod(allRows[r][c]);
            } else {
                const std::string& val = allRows[r][c];
                for (size_t k = 0; k < categories[c].size(); k++) {
                    if (categories[c][k] == val) {
                        featureMatrix(r, startIndex[c] + k) = 1.0;
                        break;
                    }
                }
            }
        }
        for (size_t c = featureColsOriginal; c < allCols; c++) {
            labelMatrix(r, c - featureColsOriginal) = std::stod(allRows[r][c]);
        }
    }
}
//get dimensions

size_t Dataset::getSampleNums()const{
    return featureMatrix.getRows();
}
size_t Dataset::getFeatureNums()const{
    return featureMatrix.getCols();
}
size_t Dataset::getLabelNums()const{
    return labelMatrix.getCols();
}

//showing dataset

void Dataset::show(size_t start, size_t end)const{
    size_t sampleNums=featureMatrix.getRows();
    if(start>sampleNums||start>end||end>sampleNums||start==end){
        throw std::invalid_argument("invalid starting or ending position");
    }
    size_t featureNums=featureMatrix.getCols();
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << std::setw(18) << featureNames[j];
    }
    std::cout << ' ';
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << std::setw(18) << labelNames[j]; 
    }
    std::cout << "\n";
    for(size_t i=start;i<end;i++){
        for(size_t j=0;j<featureNums;j++){
            std::cout << std::setw(18) << featureMatrix(i,j);
        }
        std ::cout<< ' ';
        for(size_t j=0;j<labelNums;j++){
            std::cout << std::setw(18) << labelMatrix(i,j);
        }
        std::cout<< "\n";
    }
}

void Dataset::show()const{
    size_t sampleNums=featureMatrix.getRows();  
    (*this).show(0,sampleNums);
}

void Dataset:: showFeatures()const{
    size_t featureNums=featureMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << featureNames[j] << ' ';
    }
    std::cout<< '\n';
    featureMatrix.show();
}
void Dataset::showLabels()const{
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << labelNames[j] << ' ';
    }
    std::cout<< '\n';
    labelMatrix.show();
}
void Dataset::showFeatureNames()const{
    size_t featureNums=featureMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << featureNames[j] << ' ';
    }
}
void Dataset::showLabelNames()const{
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << labelNames[j] << ' ';
    }
}
void Dataset::head()const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<5){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(0,5);
}
void Dataset::head(size_t n)const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<n){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(0,n);
}

void Dataset::tail()const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<5){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(sampleNums-5,sampleNums);
}
void Dataset::tail(size_t n)const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<n){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(sampleNums-n,sampleNums);
}



//get matrices or vectors

Matrix Dataset::getX()const{
    return featureMatrix;
}
Matrix Dataset::getY()const{
    return labelMatrix;
}
std::vector<std::string> Dataset:: getFeatureNames()const{
    return featureNames;
}
std::vector<std::string> Dataset:: getLabelNames()const{
    return labelNames;
}

// data set operations
trainTest Dataset:: trainTestSplit(size_t startTrain, size_t endTrain,size_t startTest, size_t endTest)const{
    if (startTrain < endTest && startTest < endTrain) {
        throw std::range_error("train sample and test sample overlap");
    }
    Matrix trainX = featureMatrix.rowSlice(startTrain,endTrain);
    Matrix trainY = labelMatrix.rowSlice(startTrain,endTrain);
    Matrix testX = featureMatrix.rowSlice(startTest,endTest);
    Matrix testY = labelMatrix.rowSlice(startTest,endTest);
    Dataset trainset(trainX,trainY);
    Dataset testset(testX,testY);
    return trainTest{trainset,testset};
}
trainTest Dataset:: trainTestSplit(size_t splitPoint)const{
    size_t rows= this->getSampleNums();
    if (splitPoint == 0 || splitPoint >= rows)
        throw std::invalid_argument("splitPoint must be between 1 and rows-1");
    return this->trainTestSplit(0, splitPoint,splitPoint,rows);
}
void Dataset::shuffle(){
    size_t rows = this->getSampleNums();
    if (rows <= 1) return;
    std::random_device rd;
    std::mt19937 engine(rd());
    for (size_t i = 0; i < rows - 1; i++) {
        std::uniform_int_distribution<size_t> dist(i, rows - 1);
        size_t randomNumber = dist(engine);
        featureMatrix.swapRows(i, randomNumber);
        labelMatrix.swapRows(i, randomNumber);
    }
}
void Dataset::normalize(){
    Matrix means=featureMatrix.mean(0);
    Matrix stds=featureMatrix.std(0);
    for(size_t i=0;i<featureMatrix.getRows();i++){
        for(size_t j=0;j<featureMatrix.getCols();j++){
            if(stds(0,j) > epsilon){
                featureMatrix(i,j) = (featureMatrix(i,j) - means(0,j)) / stds(0,j);
            }
        }
    }
}