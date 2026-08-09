#pragma once
#include <cstddef>


class Matrix{
public:
    // constructor and destructor
    Matrix(size_t r, size_t c);
    Matrix(const Matrix& m);
    ~Matrix();

    // Matrix operations
    Matrix operator+(const Matrix& m2) const;
    Matrix operator-(const Matrix& m2) const;
    Matrix operator*(const Matrix& m2) const;
    Matrix operator*(const double k) const;
    Matrix& operator=(const Matrix& m2);
    double& operator()(size_t r, size_t c);

    Matrix transpose();

    size_t getRows();
    size_t getCols();

    // showing matrix
    void show();
    void head();
    void head(size_t n);
    void tail();
    void tail(size_t n);

private:
    size_t rows, cols;
    double *data;
};