#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include "Matrix.hpp"
#include <vector>
#include <iomanip>

class Dataset{
    public:
    // constructors
    Dataset(const std::string& filepath, size_t labelNums=1);
    Dataset(const Matrix& x,const Matrix& y);

    //get dimensions
    size_t getSamples() const;
    size_t getFeatures() const;
    size_t getLabels() const;

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

    //get matrices
    Matrix getX()const;
    Matrix getY()const;
    

    private:
        Matrix featureMatrix, labelMatrix;
        std ::vector<std::string> featureNames;
        std ::vector<std::string> labelNames;
};
