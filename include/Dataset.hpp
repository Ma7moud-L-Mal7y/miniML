#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include "matrix.hpp"
#include <vector>

class Dataset{
    public:
    // constructors
    Dataset(const std::string& filepath);
    Dataset(const Matrix& x,const Matrix& y);

    private:
        Matrix featureMatrix, labelMatrix;
        std ::vector<std::string> featureNames;
        std ::vector<std::string> labelNames;
};
