#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include "Matrix.hpp"
#include <vector>
#include <iomanip>
#include <random>
struct trainTest;

class Dataset{
    public:
    // constructors
    Dataset(const std::string& filepath, size_t labelNums=1);
    Dataset(const Matrix& x,const Matrix& y);

    //get dimensions
    size_t getSampleNums() const;
    size_t getFeatureNums() const;
    size_t getLabelNums() const;

    // show dataset 
    void show(size_t start, size_t end) const;
    void show() const;
    void showFeatures() const;
    void showLabels()const;
    void showFeatureNames() const;
    void showLabelNames()const;
    void head() const;
    void head(size_t n) const;
    void tail() const;
    void tail(size_t n) const;

    //get matrices or vectors
    Matrix getX()const;
    Matrix getY()const;
    std::vector<std::string> getFeatureNames()const;
    std::vector<std::string> getLabelNames()const;
    
    //dataset operations
    trainTest trainTestSplit(size_t startTrain, size_t endTrain,size_t startTest, size_t endTest)const;
    trainTest trainTestSplit(size_t splitPoint)const;
    void shuffle();

    private:
        Matrix featureMatrix, labelMatrix;
        std ::vector<std::string> featureNames;
        std ::vector<std::string> labelNames;
};

struct trainTest{
    Dataset trainSample;
    Dataset testSample;
};