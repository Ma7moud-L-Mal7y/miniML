#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <random>
#include "Matrix.hpp"

struct Column {
    std::string name;
    bool isNumeric;
    std::vector<double> numericValues;
    std::vector<std::string> textValues;
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
};

struct trainTest {
    Dataset trainSample;
    Dataset testSample;
};