#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <random>
#include "Matrix.hpp"
#include <map>

struct Column {
    std::string name;
    bool isNumeric;
    std::vector<double> numericValues;
    std::vector<std::string> textValues;
    mutable bool hasEncoding = false;
    mutable std::map<std::string, double> encodeMap;
    mutable std::map<double, std::string> decodeMap;
};
struct trainTest;

class Dataset {
public:
    //constructor
    Dataset(const std::string& filepath, size_t labelNums);
    //dimensions
    size_t getSampleNums() const;
    size_t getFeatureNums() const;
    size_t getLabelNums() const;
    //showing dataset
    void show(size_t start, size_t end) const;
    void showSelected(size_t start, size_t end) const;
    void show() const;
    void head() const;
    void head(size_t n) const;
    void tail() const;
    void tail(size_t n) const;
    //get matrices
    Matrix getX() const;
    Matrix getY() const;
    Matrix getXEncoded() const;
    Matrix getYEncoded() const; 
    std::vector<std::string> getSelectedFeatureNames() const;
    std::vector<std::string> getAllFeatureNames()const;
    std::vector<std::string> getLabelNames() const;
    std::vector<std::string> getStringColumn(const std::string& colName)const;
    std::vector<std::string> getStringColumnNames()const;
    //dataset operations
    trainTest trainTestSplit(size_t startTrain, size_t endTrain, size_t startTest, size_t endTest) const;
    trainTest trainTestSplit(size_t splitPoint) const;
    void shuffle();
    Matrix normalize();
    Matrix normalize(const Matrix& x)const;
    void selectFeatures(const std::vector<std::string>& featureNames);
    void selectFeatures(std::initializer_list<std::string> featureNames);
    void resetFeatures();
    void selectLabel(const std::string& labelName);
    void resetLabels();
    std::string decodeLabel(double value,const std::string colName) const;  

private:
    std::vector<Column> columns;
    size_t labelCount;
    std::vector<std::string> selectedFeatureNames;
    bool isNumericString(const std::string& s) const;
    const Column& findCol(const std::string& name) const;
    Dataset(std::vector<Column> cols, size_t labelNums, std::vector<std::string> selectedFeatures);
    void swapRows(size_t r1, size_t r2);
    std::vector<double> normMeans;
    std::vector<double> normStds;
    bool hasNormStats = false;
    std::string selectedLabelName;
    void encodeColumn(const Column& col) const;
};

struct trainTest {
    Dataset trainSample;
    Dataset testSample;
};