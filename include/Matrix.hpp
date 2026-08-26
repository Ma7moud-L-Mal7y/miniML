#pragma once

#include <iostream>
#include <algorithm>
#include <initializer_list>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cfloat>
#include <iomanip>


constexpr double epsilon = 0.0000001;

// helper structs
struct luDecomposition;

class Matrix{
public:
    // constructor and destructor
    Matrix(size_t r, size_t c);
    Matrix(const Matrix& m);
    Matrix(size_t r, size_t c, std::initializer_list<double> list);
    ~Matrix();

    static Matrix Identity(size_t n);

    // Matrix operations
    Matrix operator+(const Matrix& m2) const;
    Matrix operator-(const Matrix& m2) const;
    Matrix operator*(const Matrix& m2) const;
    
    Matrix& operator+=(const Matrix& m2);
    Matrix& operator-=(const Matrix& m2);
    Matrix& operator=(const Matrix& m2);
    double operator()(size_t r, size_t c) const;
    double& operator()(size_t r, size_t c);
    bool operator==(const Matrix& m2) const;

    luDecomposition luDecompose() const;
    Matrix transpose() const;
    double determinant() const;
    size_t rank() const;
    Matrix solve(const Matrix& b) const;
    Matrix solve(const Matrix& b, luDecomposition lu) const;
    Matrix inverse() const;

    Matrix rowSlice(size_t r1, size_t r2) const;
    Matrix colSlice(size_t c1, size_t c2) const;
    Matrix appendRows(const Matrix& m2) const;
    Matrix appendCols(const Matrix& m2) const;

    // element-wise operations
    Matrix hadamard(const Matrix& m2) const;
    Matrix apply(double (*func)(double)) const;
    Matrix operator+(const double k) const;
    Matrix operator-(const double k) const;
    Matrix operator*(const double k) const;
    Matrix& operator+=(const double k);
    Matrix& operator-=(const double k);
    Matrix& operator*=(const double k);

    // redutions and aggregations
    double sum() const;
    Matrix sum(int axis) const;
    double mean() const;
    Matrix mean(int axis) const;
    double std()const;
    Matrix std(int axis)const;

    // get dimensions
    size_t getRows() const;
    size_t getCols() const;

    // elementary row operations
    void swapRows(size_t r1, size_t r2);
    void scaleRow(size_t r, double k);
    void addScaledRow(size_t r1, size_t r2, double k);

    // checks
    bool isSquare() const;
    bool isSymmetric() const;
    bool isSingular() const;

    // showing matrix
    void show() const;
    void head() const;
    void head(size_t n) const;
    void tail() const;
    void tail(size_t n) const;

private:
    size_t rows, cols;
    double *data;
};

// struct definition
struct luDecomposition{
    Matrix L;
    Matrix U;
    std::vector<size_t> P;
    size_t swaps;
};

// free functions
Matrix operator*(double k, const Matrix& m);
Matrix operator+(double k, const Matrix& m);
Matrix operator-(double k, const Matrix& m);

void showLU(const luDecomposition& lu);