#include "Dataset.hpp"

//constructors
Dataset::Dataset(const std::string& filepath,size_t labelNums)
    :labelCount(labelNums)
{   
    std:: ifstream file(filepath);
    if(!file.is_open()){
        throw std::runtime_error("could not open file: " + filepath);
    }
    std::vector<std::string> allNames;
    std::string line;
    std::getline(file,line);
    std::stringstream headerstream(line);
    std::string name;
    while(std::getline(headerstream,name,',')) allNames.push_back(name);
    std::vector<std::vector<std::string>> allRows;
    while(std::getline(file,line)){
        std::stringstream linestream(line);
        std::string piece;
        std::vector<std::string> row;
        while(std::getline(linestream,piece,',')) row.push_back(piece);
        allRows.push_back(row);
    }
    size_t allCols=allNames.size();
    if (labelNums > allCols) throw std::invalid_argument("label count exceeds total columns");
    columns.resize(allCols);
    for(size_t c=0;c<allCols;c++){
        columns[c].name= allNames[c];
        bool numeric=true;
        for(const auto& row:allRows){
            if(!isNumericString(row[c])){numeric=false;break;}
        }
        columns[c].isNumeric=numeric;
        for(const auto& row:allRows){
            if(numeric) columns[c].numericValues.push_back(std::stod(row[c]));
            else columns[c].textValues.push_back(row[c]);
        }
    }
    for(size_t c=0;c<allCols-labelNums;c++){
        if(columns[c].isNumeric) selectedFeatureNames.push_back(columns[c].name);
    }
}
Dataset::Dataset(std::vector<Column> cols, size_t labelNums, 
                 std::vector<std::string> selectedFeatures)
    : columns(std::move(cols)), 
      labelCount(labelNums), 
      selectedFeatureNames(std::move(selectedFeatures)) 
{

}
static Column sliceColumn(const Column& col, size_t start, size_t end) {
    Column result;
    result.name = col.name;
    result.isNumeric = col.isNumeric;
    if (col.isNumeric) {
        result.numericValues.assign(
            col.numericValues.begin() + start,
            col.numericValues.begin() + end
        );
    } else {
        result.textValues.assign(
            col.textValues.begin() + start,
            col.textValues.begin() + end
        );
    }
    return result;
}
//get dimensions
size_t Dataset::getSampleNums() const {
    if (columns.empty()) return 0;
    return columns[0].isNumeric 
           ? columns[0].numericValues.size() 
           : columns[0].textValues.size();
}
size_t Dataset::getFeatureNums()const{
    return selectedFeatureNames.size();
}
size_t Dataset::getLabelNums()const{
    return labelCount;
}

//get matrices

Matrix Dataset::getX() const {
    if (columns.empty()) {
        return Matrix(0, 0);
    }
    size_t rows = columns[0].isNumeric ? columns[0].numericValues.size() : columns[0].textValues.size();
    Matrix result(rows, selectedFeatureNames.size());
    for (size_t i = 0; i < selectedFeatureNames.size(); i++) {
        const Column& col = findCol(selectedFeatureNames[i]);
        if (!col.isNumeric) {
            throw std::runtime_error("Feature column '" + col.name + "' is not numeric");
        }
        for (size_t j = 0; j < rows; j++) {
            result(j, i) = col.numericValues[j]; 
        }
    }
    return result;
}

Matrix Dataset::getY() const {
    if (columns.empty()) return Matrix(0, 0);
    size_t rows = columns[0].isNumeric ? columns[0].numericValues.size() : columns[0].textValues.size();
    size_t featureCols = columns.size() - labelCount;
    Matrix result(rows, labelCount);
    for (size_t i = 0; i < labelCount; i++) {
        const Column& col = columns[featureCols + i];
        if (!col.isNumeric) {
            throw std::runtime_error("Label column '" + col.name + "' is not numeric");
        }
        for (size_t j = 0; j < rows; j++) {
            result(j, i) = col.numericValues[j];
        }
    }
    return result;
}
std::vector<std::string> Dataset::getSelectedFeatureNames()const{
    return selectedFeatureNames;
}
std::vector<std::string> Dataset::getAllFeatureNames()const{
    size_t featureCount= columns.size()-labelCount;
    std::vector<std::string> names;
    for(size_t c=0;c<featureCount;c++){
        if(columns[c].isNumeric){
            names.push_back(columns[c].name);
        }
    }
    return names;
}
std::vector<std::string> Dataset::getLabelNames()const{
    size_t featureCount= columns.size()-labelCount;
    std::vector<std::string> names;
    for(size_t c=featureCount;c<columns.size();c++){
        names.push_back(columns[c].name);
    }
    return names;
}
std::vector<std::string> Dataset::getStringColumn(const std::string& colName)const{
    const Column& col = findCol(colName);
    if(col.isNumeric){
        throw std::invalid_argument("column '" + colName + "' is numeric, not a string column");
    }
    return col.textValues;
}
std::vector<std::string> Dataset::getStringColumnNames() const {
    std::vector<std::string> names;
    for (const auto& col : columns) {
        if (!col.isNumeric) {
            names.push_back(col.name);
        }
    }
    return names;
}
//showing dataset
void Dataset::show(size_t start, size_t end) const {
    size_t rows = getSampleNums();
    if (start >= rows || start > end || end > rows || start == end) {
        throw std::invalid_argument("invalid starting or ending position");
    }
    for (size_t c=0;c<columns.size();c++) std::cout << std::setw(15) << columns[c].name;
    std::cout<< '\n';
    for (size_t i = start; i < end; i++) {
        for (size_t c=0;c<columns.size();c++) {
            if(columns[c].isNumeric){
                std::cout << std::setw(15) << columns[c].numericValues[i];
            }
            else{
                std::cout << std::setw(15) << columns[c].textValues[i];
            }
        }
        std::cout<<'\n';
    }
}
void Dataset::showSelected(size_t start, size_t end) const {
    size_t rows = getSampleNums();
    if (start >= rows || start > end || end > rows || start == end) {
        throw std::invalid_argument("invalid starting or ending position");
    }
    size_t firstLabel = columns.size() - labelCount;
    for (const auto& name : selectedFeatureNames) {
        std::cout << std::setw(18) << name;
    }
    for (size_t c = firstLabel; c < columns.size(); ++c) {
        std::cout << std::setw(18) << columns[c].name;
    }
    std::cout << '\n';
    for (size_t i = start; i < end; i++) {
        for (const auto& name : selectedFeatureNames) {
            const Column& col = findCol(name);
            std::cout << std::setw(18) << col.numericValues[i];
        }
        for (size_t c = firstLabel; c < columns.size(); ++c) {
            std::cout << std::setw(18) << columns[c].numericValues[i];
        }
        std::cout << '\n';
    }
}
void Dataset::show() const {
    show(0, getSampleNums());
}

void Dataset::head() const {
    show(0, std::min<size_t>(5, getSampleNums()));
}

void Dataset::head(size_t n) const {
    show(0, std::min(n, getSampleNums()));
}

void Dataset::tail() const {
    size_t rows = getSampleNums();
    size_t n = std::min<size_t>(5, rows);
    show(rows - n, rows);
}

void Dataset::tail(size_t n) const {
    size_t rows = getSampleNums();
    n = std::min(n, rows);
    show(rows - n, rows);
}
//data set operations
void Dataset::selectFeatures(const std::vector<std::string>& featureNames) {
    std::vector<std::string> newSelection;
    newSelection.reserve(featureNames.size());

    for (const auto& name : featureNames) {
        for (const auto& already : newSelection) {
            if (already == name) {
                throw std::invalid_argument("Duplicate feature name in selection: " + name);
            }
        }
        const Column& col = findCol(name);
        size_t colIndex = 0;
        for (; colIndex < columns.size(); ++colIndex) {
            if (columns[colIndex].name == name) break;
        }
        if (colIndex >= columns.size() - labelCount) {
            throw std::invalid_argument("'" + name + "' is a label column and cannot be selected as a feature");
        }
        if (!col.isNumeric) {
            throw std::invalid_argument("Feature '" + name + "' is not numeric");
        }

        newSelection.push_back(name);
    }
    selectedFeatureNames = std::move(newSelection);
    hasNormStats=false;
    normMeans.clear();
    normStds.clear();
}
void Dataset::selectFeatures(std::initializer_list<std::string> featureNames) {
    selectFeatures(std::vector<std::string>(featureNames));
}
void Dataset::resetFeatures() {
    selectedFeatureNames.clear();
    size_t featureCount = columns.size() - labelCount;
    for (size_t c = 0; c < featureCount; ++c)
        if (columns[c].isNumeric)
            selectedFeatureNames.push_back(columns[c].name);
    hasNormStats = false;
    normMeans.clear();
    normStds.clear();
}
trainTest Dataset::trainTestSplit(size_t startTrain, size_t endTrain, size_t startTest, size_t endTest) const {
    size_t rows = getSampleNums();
    if (endTrain > rows || endTest > rows)
        throw std::range_error("split indices out of bounds");
    if (startTrain >= endTrain || startTest >= endTest)
        throw std::invalid_argument("empty train or test set");
    if (startTrain < endTest && startTest < endTrain)
        throw std::range_error("train and test sets overlap");
    std::vector<Column> trainCols;
    std::vector<Column> testCols;
    trainCols.reserve(columns.size());
    testCols.reserve(columns.size());
    for (const auto& col : columns) {
        trainCols.push_back(sliceColumn(col, startTrain, endTrain));
        testCols.push_back(sliceColumn(col, startTest, endTest));
    }
    return trainTest{
        Dataset(std::move(trainCols), labelCount, selectedFeatureNames),
        Dataset(std::move(testCols), labelCount, selectedFeatureNames)
    };
}

trainTest Dataset::trainTestSplit(size_t splitPoint) const {
    size_t rows = getSampleNums();
    if (splitPoint == 0 || splitPoint >= rows)
        throw std::invalid_argument("splitPoint must be between 1 and rows-1");
    return trainTestSplit(0, splitPoint, splitPoint, rows);
}


Matrix Dataset::normalize(){
    Matrix X = getX();
    size_t rows = X.getRows();
    size_t cols = X.getCols();
    if (rows == 0 || cols == 0) return X;
    Matrix means = X.mean(0);
    Matrix stds  = X.std(0);
    normMeans.resize(cols);
    normStds.resize(cols);
    for(size_t c=0;c<cols;c++){
        normMeans[c]=means(0,c);
        normStds[c]=stds(0,c);
    }
    hasNormStats=true;
    for (size_t c = 0; c < cols; ++c) {
        double mean = normMeans[c];
        double stddev = normStds[c];
        if (stddev < 1e-8) {
            for (size_t r = 0; r < rows; ++r) {
                X(r, c) = 0.0;
            }
        } else {
            for (size_t r = 0; r < rows; ++r) {
                X(r, c) = (X(r, c) - mean) / stddev;
            }
        }
    }
    return X;
}
Matrix Dataset::normalize(const Matrix& x)const{
    if(!hasNormStats)throw std::runtime_error("please normalize the data first before testing");
    if(x.getCols()!=normMeans.size())throw std::range_error("matrix dimensions dont match");
    Matrix result=x;
    size_t rows=result.getRows();
    size_t cols= result.getCols();
    for(size_t c=0;c<cols;c++){
        double mean=normMeans[c];
        double stddev=normStds[c];
        for(size_t r=0;r<rows;r++){
            result(r, c) = (stddev < 1e-8) ? 0.0 : (result(r, c) - mean) / stddev;
        }
    }
    return result;
}
void Dataset::shuffle() {
    size_t rows = getSampleNums();
    if (rows <= 1) return; 
    std::random_device rd;
    std::mt19937 engine(rd());
    for (size_t i = 0; i < rows - 1; ++i) {
        std::uniform_int_distribution<size_t> dist(i, rows - 1);
        size_t j = dist(engine);
        if (i != j) {
            swapRows(i, j);
        }
    }
}
//helper functions
void Dataset::swapRows(size_t r1, size_t r2) {
    for (auto& col : columns) {
        if (col.isNumeric) {
            std::swap(col.numericValues[r1], col.numericValues[r2]);
        } else {
            std::swap(col.textValues[r1], col.textValues[r2]);
        }
    }
}
bool Dataset::isNumericString(const std::string& s) const{
    if (s.empty()) return false;
    try {
        size_t pos;
        std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}
const Column& Dataset::findCol(const std::string& name) const {
    for (const auto& col : columns) {
        if (col.name == name) {
            return col;
        }
    }
    throw std::runtime_error("Column not found: " + name);
}