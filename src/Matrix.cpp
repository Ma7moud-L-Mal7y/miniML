#include <iostream>
#include <algorithm>
#include "Matrix.hpp"


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

Matrix Matrix::Identity(size_t n){
    Matrix result(n, n);
    for(size_t i = 0; i < n; i++){
        result.data[i + i*n] = 1;
    }
    return result;
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
    for(size_t i = 0; i < rows * cols; i++){
        data[i] = m2.data[i];
    }
    return *this;
}

double& Matrix::operator()(size_t r, size_t c) const{
    if(r >= rows || c >= cols){
        throw std::range_error("index out of range");
    }
    return data[r * cols + c];
}

bool Matrix::operator==(const Matrix& m2) const{
    if(rows != m2.rows || cols != m2.cols)
        return false;
    for(size_t i = 0; i < rows * cols; i++){
        if(std::abs(data[i] - m2.data[i]) > epsilon)
            return false;
    }
    return true;
}

Matrix Matrix::transpose() const{
    Matrix trans(cols, rows);
    for(size_t j = 0; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            trans.data[j + i * rows] = data[i + j * cols];
        }
    }
    return trans;
}

double Matrix::determinant() const{

}

luDecomposition Matrix::luDecompose(){
    if(rows != cols){
        throw std::range_error("matrix is not square");
    }

    luDecomposition result;
    Matrix U(*this);
    Matrix L = Matrix::Identity(rows);

}

size_t Matrix::getRows() const{
    return rows;
}

size_t Matrix::getCols() const{
    return cols;
}

// elementary row operations
void Matrix::swapRows(size_t r1, size_t r2){
    for(size_t i = 0; i < cols; i++){
        double tmp = data[r1 * cols + i];
        data[r1 * cols + i] = data[r2 * cols + i];
        data[r2 * cols + i] = tmp;
    }
}

void Matrix::scaleRow(size_t r, double k){
    for(size_t i = 0; i < cols; i++){
        data[r*cols + i] *= k;
    }
}

void Matrix::addScaledRow(size_t dst, size_t scr, double k){
    for(size_t i = 0; i < cols; i++){
        data[dst*cols + i] += k * data[scr * cols + i];
    }
}


// showing matrix
void Matrix::show() const{
    for(size_t j = 0; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::head() const{
    if(rows < 5){
        (*this).show();
        return;
    }

    for(size_t j = 0; j < 5; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::head(size_t n) const{
    if(rows < n){
        (*this).show();
        return;
    }

    for(size_t j = 0; j < n; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::tail() const{
    if(rows < 5){
        (*this).show();
        return;
    }

    for(size_t j = rows - 5; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::tail(size_t n) const{
    if(rows < n){
        (*this).show();
        return;
    }

    for(size_t j = rows - n; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

// free functions
Matrix operator*(double k, const Matrix& m){
    return m * k;
}