#include <iostream>
#include <algorithm>
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

Matrix Matrix::operator-(const Matrix&m2) const{
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    Matrix result(rows, cols);
    for(size_t i = 0; i < rows * cols; i++){
        result.data[i] = data[i] - m2.data[i];
    }

    return result;
}

Matrix Matrix::operator*(const Matrix&m2) const{
    if(cols != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    Matrix result(rows, m2.cols);
    for(size_t j = 0; j < rows; j++){
        for(size_t i = 0; i < m2.cols; i++){
            double SOP = 0;
            for(size_t k = 0; k < cols; k++){
                SOP += data[k + cols * j] * m2.data[i + m2.cols * k];
            }
            result.data[j * m2.cols + i] = SOP;
        }
    }

    return result;
}

Matrix Matrix::operator*(const double k) const{
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows * cols; i++){
        result.data[i] = data[i] * k;
    }
    return result;
}

Matrix& Matrix::operator=(const Matrix& m2){
    if(this == &m2) return *this;

    rows = m2.rows;
    cols = m2.cols;
    delete[] data;
    data = new double[rows * cols];
    for(int i = 0; i < rows * cols; i++){
        data[i] = m2.data[i];
    }
    return *this;
}

double& Matrix::operator()(size_t r, size_t c){
    if(r >= rows || c >= cols){
        throw std::range_error("index out of range");
    }
    return data[r * cols + c];
}

Matrix Matrix::transpose(){
    Matrix trans(cols, rows);
    for(size_t j = 0; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            trans.data[j + i * rows] = data[i + j * cols];
        }
    }
    return trans;
}

size_t Matrix::getRows(){
    return rows;
}

size_t Matrix::getCols(){
    return cols;
}

// showing matrix
void Matrix::show(){
    for(int j = 0; j < rows; j++){
        for(int i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::head(){
    if(rows < 5){
        (*this).show();
        return;
    }

    for(int j = 0; j < 5; j++){
        for(int i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::head(size_t n){
    if(rows < n){
        (*this).show();
        return;
    }

    for(int j = 0; j < n; j++){
        for(int i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::tail(){
    if(rows < 5){
        (*this).show();
        return;
    }

    for(int j = rows - 5; j < rows; j++){
        for(int i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::tail(size_t n){
    if(rows < n){
        (*this).show();
        return;
    }

    for(int j = rows - n; j < rows; j++){
        for(int i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}