#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>


#define epsilon 0.00001

// helper structs
typedef struct{
    Matrix L;
    Matrix U;
    std::vector<size_t> P;
} luDecomposition;

class Matrix{
public:
    // constructor and destructor
    Matrix(size_t r, size_t c);
    Matrix(const Matrix& m);
    ~Matrix();

    static Matrix Identity(size_t n);

    // Matrix operations
    Matrix operator+(const Matrix& m2) const;
    Matrix operator-(const Matrix& m2) const;
    Matrix operator*(const Matrix& m2) const;
    Matrix operator*(const double k) const;
    Matrix& operator=(const Matrix& m2);
    double& operator()(size_t r, size_t c) const;
    bool operator==(const Matrix& m2) const;

    Matrix transpose() const;
    double determinant() const;
    luDecomposition luDecompose();

    size_t getRows() const;
    size_t getCols() const;

    // elementary row operations
    void swapRows(size_t r1, size_t r2);
    void scaleRow(size_t r, double k);
    void addScaledRow(size_t r1, size_t r2, double k);

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

// free functions
Matrix operator*(double k, Matrix& m);

