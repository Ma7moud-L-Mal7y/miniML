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

Matrix::Matrix(size_t r, size_t c, std::initializer_list<double> list)
    : rows(r), cols(c), data(new double[r*c])
{   
    size_t n = list.size();
    if(n > r*c)
        throw std::invalid_argument("Matrix can't hold init list");

    size_t i = 0;
    for(const auto& element : list){
        data[i++] = element;
    }
    for(; i < r*c; i++){
        data[i] = 0;
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

Matrix& Matrix::operator+=(const Matrix& m2){
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    for(int i = 0; i < rows * cols; i++){
        data[i] += m2.data[i];
    }

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& m2){
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    for(int i = 0; i < rows * cols; i++){
        data[i] -= m2.data[i];
    }

    return *this;
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

double Matrix::operator()(size_t r, size_t c) const{
    if(r >= rows || c >= cols){
        throw std::range_error("index out of range");
    }
    return data[r * cols + c];
}

double& Matrix::operator()(size_t r, size_t c){
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

luDecomposition Matrix::luDecompose() const{
    if(!isSquare()){
        throw std::range_error("matrix is not square");
    }

    Matrix U(*this);
    Matrix L = Matrix::Identity(rows);
    std::vector<size_t> P(rows);
    for(size_t i = 0; i < rows; i++){
        P[i] = i;
    }
    size_t swaps = 0;

    for(size_t col = 0; col < cols; col++){
        double max_element = 0.0;
        size_t max_idx = 0;
        for(size_t row = col; row < rows; row++){
            if(std::abs(U(row, col)) > max_element){
                max_element = std::abs(U(row, col));
                max_idx = row;
            }
        }
        std::swap(P[max_idx], P[col]);
        U.swapRows(col, max_idx);
        swaps += (col != max_idx);

        if(max_idx != col){
            for(size_t k = 0; k < col; k++){
                double tmp = L(col, k);
                L(col, k) = L(max_idx, k);
                L(max_idx, k) = tmp;
            }
        }


        for(size_t row = col + 1; row < rows; row++){
            double m = U(row, col) / U(col, col);
            L(row, col) = m;
            U.addScaledRow(row, col, -m);
        }
    }

    return luDecomposition{L, U, P, swaps};
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
    luDecomposition lu = luDecompose();
    double result = ((lu.swaps % 2 == 0) ? 1 : -1);
    for(size_t i = 0; i < rows; i++){
        if(lu.U(i, i) < epsilon && lu.U(i, i) > -epsilon) return 0;
        result *= lu.U(i, i);
    }

    return result;
}

size_t Matrix::rank() const{
    size_t r = 0;
    luDecomposition lu = luDecompose();
    for(size_t i = 0; i < rows; i++){
        r += (std::abs(lu.U(i, i)) > epsilon);
    }

    return r;
}

Matrix Matrix::solve(const Matrix& b) const{
    Matrix x(rows, 1);

    // solve logic

    return x;
}

Matrix Matrix::inverse() const{
    // inverse logic
}

// element-wise operations
Matrix Matrix::hadamard(const Matrix& m2) const{
    if(rows != m2.rows || cols != m2.cols)
        throw std::range_error("dimension are not compatible");

    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            result.data[j + i * cols] = (*this)(i, j) * m2(i, j);
        }
    }

    return result;
}

Matrix Matrix::apply(double (*func)(double)) const{
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            result.data[j + i * cols] = func((*this)(i, j));
        }
    }

    return result;
}

Matrix Matrix::operator+(const double k) const{
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows * cols; i++){
        result.data[i] = data[i] + k;
    }
    return result;
}

Matrix Matrix::operator-(const double k) const{
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows * cols; i++){
        result.data[i] = data[i] - k;
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

Matrix& Matrix::operator+=(const double k){
    for(int i = 0; i < rows * cols; i++){
        data[i] += k;
    }

    return *this;
}

Matrix& Matrix::operator-=(const double k){
    for(int i = 0; i < rows * cols; i++){
        data[i] -= k;
    }

    return *this;
}

Matrix& Matrix::operator*=(const double k){
    for(int i = 0; i < rows * cols; i++){
        data[i] *= k;
    }

    return *this;
}

// reductions and aggregations
double Matrix::sum() const{
    double result = 0;
    for(size_t i = 0; i < rows*cols; i++){
        result += data[i];
    }
    return result;
}

double Matrix::mean() const{
    return sum()/(rows*cols);
}

Matrix Matrix::sum(int axis) const{
    if(axis == 0){
        Matrix result(rows, 1);
        for(size_t i = 0; i < rows; i++){
            double sum = 0;
            for(size_t j = 0; j < cols; j++){
                sum += (*this)(i, j);
            }
            result.data[i] = sum;
        }

        return result;
    }
    else if(axis == 1){
        Matrix result(1, cols);
        for(size_t i = 0; i < cols; i++){
            double sum = 0;
            for(size_t j = 0; j < rows; j++){
                sum += (*this)(j, i);
            }
            result.data[i] = sum;
        }

        return result;
    }
    else
        throw std::invalid_argument("axis must be 0 or 1");
}

Matrix Matrix::mean(int axis) const{
    if(axis == 0){
        Matrix result(rows, 1);
        for(size_t i = 0; i < rows; i++){
            double sum = 0;
            for(size_t j = 0; j < cols; j++){
                sum += (*this)(i, j);
            }
            result.data[i] = sum / cols;
        }

        return result;
    }
    else if(axis == 1){
        Matrix result(1, cols);
        for(size_t i = 0; i < cols; i++){
            double sum = 0;
            for(size_t j = 0; j < rows; j++){
                sum += (*this)(j, i);
            }
            result.data[i] = sum/ rows;
        }

        return result;
    }
    else
        throw std::invalid_argument("axis must be 0 or 1");
}

// get dimensions
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

// checks
bool Matrix::isSquare() const{
    return(rows == cols);
}
bool Matrix::isSymmetric() const{
    if(!isSquare()) return false;

    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            if(std::abs((*this)(i, j) - (*this)(j, i)) > epsilon)
                return false;
        }
    }
    return true;
}
bool Matrix::isSingular() const{
    return (std::abs((determinant())) < epsilon);
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
Matrix operator*(const double k, const Matrix& m){
    return m * k;
}

Matrix operator+(const double k, const Matrix& m){
    return m + k;
}

Matrix operator-(const double k, const Matrix& m){
    return m - k;
}