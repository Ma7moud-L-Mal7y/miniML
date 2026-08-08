#include <iostream>
#include "linearAlgebra.hpp"


// constructors and destructor
Matrix::Matrix(size_t r, size_t c)
    : rows(r), cols(c), data(new double[r*c])
{
    for(size_t i = 0; i < r*c; ++i){
        data[i] = 0.0;
    }
};

Matrix::Matrix(const Matrix& m2)
    : rows(m2.rows), cols(m2.cols), data(new double[rows*cols])
{
    
    for(size_t i = 0; i < rows * cols; i++){
        data[i] = m2.data[i];
    }
}

Matrix::~Matrix(){
    delete[] data;
}

// Matrix operations
Matrix Matrix::operator+(const Matrix&m2) const{
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    Matrix result(rows, cols);
    for(size_t i = 0; i < rows * cols; i++){
        result.data[i] = m2.data[i] + data[i];
    }

    return result;
}