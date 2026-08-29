#include "Matrix.hpp"

// constructors and destructor
Matrix::Matrix(size_t r, size_t c)
    : rows(r), cols(c), data(new double[r*c])
{

}

Matrix::Matrix(size_t r, size_t c, double k)
    : rows(r), cols(c), data(new double[r*c])
{
    for(size_t i = 0; i < r*c; ++i){
        data[i] = k;
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
        for(size_t j = 0; j < n; j++){
            result(i, j) = (i == j);
        }
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

    // C_i,j = sum{ A_i,k * B_k,j }
    Matrix result(rows, m2.cols, 0);
    for(size_t i = 0; i < rows; i++){
        for(size_t k = 0; k < cols; k++){
            double a = (*this)(i, k);
            for(size_t j = 0; j < m2.cols; j++){
                result(i, j) += a * m2(k, j);
            }
        }
    }

    return result;
}

Matrix& Matrix::operator+=(const Matrix& m2){
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    for(size_t i = 0; i < rows * cols; i++){
        data[i] += m2.data[i];
    }

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& m2){
    if(cols != m2.cols || rows != m2.rows){
        throw std::range_error("dimensions are not compatible");
    }

    for(size_t i = 0; i < rows * cols; i++){
        data[i] -= m2.data[i];
    }

    return *this;
}

Matrix& Matrix::operator=(const Matrix& m2){
    if(this == &m2) return *this;
    
    double *newData = new double[m2.rows * m2.cols];
    for(size_t i = 0; i < m2.rows*m2.cols; i++){
        newData[i] = m2.data[i];
    }

    delete[] data;
    data = newData;
    rows = m2.rows;
    cols = m2.cols;

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
        size_t max_idx = col;
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
    luDecomposition lu = luDecompose();
    return solve(b, lu);
}

Matrix Matrix::solve(const Matrix& b, luDecomposition lu) const{
    if(!isSquare())
        throw std::range_error("Matrix is not square");

    if(b.rows != rows || b.cols != 1)
        throw std::range_error("b must be a column vector matching A's dimensions");

    size_t n = rows;

    // Ly = Pb <1>
    // Ux = y  <2>
    Matrix  y(n, 1);
    Matrix  x(n, 1);
    Matrix Pb(n, 1);

    for(size_t i = 0; i < n; i++){
        Pb(i, 0) = b(lu.P[i], 0);
    }
    
    y(0,0) = Pb(0,0);
    for(size_t i = 1; i < n; i++){
        double sum = 0;
        for(size_t j = 0; j < i; j++){
            sum += y(j, 0) * lu.L(i, j);
        }
        y(i, 0) = Pb(i, 0) - sum;
    }
    
    x(n-1, 0) = y(n-1, 0) / lu.U(n-1, n-1);
    for(size_t i = n-1; i-- > 0;){
        double sum = 0;
        for(size_t j = n-1; j != i; j--){
            sum += lu.U(i, j) * x(j, 0);
        }

        x(i, 0) = (y(i, 0) - sum) / lu.U(i,i);
    }

    return x;
}

Matrix Matrix::inverse() const{
    if(!isSquare())
        throw std::range_error("Matrix is not square");

    luDecomposition lu = luDecompose();
    Matrix result(rows, cols);
    for(size_t j = 0; j < cols; j++){
        Matrix e(rows, 1);
        e(j, 0) = 1;
        Matrix col = solve(e, lu);
        for(size_t i = 0; i < rows; i++){
            result(i, j) = col(i, 0);
        }
    }

    return result;

}

Matrix Matrix::rowSlice(size_t r1, size_t r2) const{
    if(r1 > r2)
        std::swap(r1, r2);

    if(r2 > rows)
        throw std::range_error("out of bound rows");

    Matrix result(r2 - r1, cols);
    for(size_t i = r1; i < r2; i++){
        for(size_t j = 0; j < cols; j++){
            result(i - r1, j) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::colSlice(size_t c1, size_t c2) const{
    if(c1 > c2)
        std::swap(c1, c2);

    if(c2 > cols)
        throw std::range_error("out of bound cols");

    Matrix result(rows, c2 - c1 );
    for(size_t j = c1; j <c2; j++){
        for(size_t i = 0; i < rows; i++){
            result(i, j - c1) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::appendRows(const Matrix& m2) const{
    if(cols != m2.cols)
        throw std::range_error("can't append matrices with different column lengths");

    Matrix result(rows + m2.rows, cols);
    for(size_t j = 0; j < cols; j++){
        for(size_t i = 0; i < rows; i++){
            result(i, j) = (*this)(i,j);
        }
        for(size_t i = 0; i < m2.rows; i++){
            result(i+rows, j) = m2(i, j);
        }
    }

    return result;
}

Matrix Matrix::appendCols(const Matrix& m2) const{
    if(rows != m2.rows)
        throw std::range_error("can't append matrices with different row lengths");

    Matrix result(rows, cols + m2.cols);
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            result(i, j) = (*this)(i,j);
        }
        for(size_t j = 0; j < m2.cols; j++){
            result(i, j + cols) = m2(i, j);
        }
    }

    return result;
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
    for(size_t i = 0; i < rows * cols; i++){
        data[i] += k;
    }

    return *this;
}

Matrix& Matrix::operator-=(const double k){
    for(size_t i = 0; i < rows * cols; i++){
        data[i] -= k;
    }

    return *this;
}

Matrix& Matrix::operator*=(const double k){
    for(size_t i = 0; i < rows * cols; i++){
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
    if(axis == 1){
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
    else if(axis == 0){
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
    return sum(axis) * (1.0/((axis) ? cols : rows));
}
double Matrix::std()const{
    double mean=this->mean();
    double variance=0;
    for(size_t i=0;i<rows*cols;i++){
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance/=(rows*cols);
    return std::sqrt(variance);
}
Matrix Matrix::std(int axis)const{
    if(axis==0){
        Matrix result(1,cols);
        Matrix means=this->mean(0);
        for(size_t i=0;i<cols;i++){
            double colVar=0;
            for(size_t j=0;j<rows;j++){
                double diff=(*this)(j,i)-means(0,i);
                colVar+=diff*diff;
            }
            colVar/=rows;
            result(0,i)=std::sqrt(colVar);
        }
        return result;
    }
    else if(axis==1){
        Matrix result(rows,1);
        Matrix means=this->mean(1);
        for(size_t i=0;i<rows;i++){
            double rowVar=0;
            for(size_t j=0;j<cols;j++){
                double diff=(*this)(i,j)-means(i,0);
                rowVar+=diff*diff;
            }
            rowVar/=cols;
            result(i,0)=std::sqrt(rowVar);
        }
        return result;
    }
    else {
    throw std::invalid_argument("axis must be 0 or 1");
    }   
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
        for(size_t j = 0; j < i; j++){
            if(std::abs((*this)(i, j) - (*this)(j, i)) > epsilon)
                return false;
        }
    }
    return true;
}
bool Matrix::isSingular() const{
    return (rank() < rows);
}

// showing matrix
void Matrix::show() const{
    std::cout << std::fixed << std::setprecision(4);
    for(size_t j = 0; j < rows; j++){
        for(size_t i = 0; i < cols; i++){
            std::cout << data[i + j * cols] << " ";
        }
        std::cout << '\n';
    }
}

void Matrix::head() const{
    std::cout << std::fixed << std::setprecision(4);
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
    std::cout << std::fixed << std::setprecision(4);
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
    std::cout << std::fixed << std::setprecision(4);
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
    std::cout << std::fixed << std::setprecision(4);
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

Matrix operator+(double k, const Matrix& m){
    return m + k;
}

Matrix operator-(double k, const Matrix& m){
    return (m*-1) + k;
}

void showLU(const luDecomposition& lu){
    std::cout << std::fixed << std::setprecision(4);
    size_t rows = lu.L.getRows(), cols = lu.L.getCols();
    for(size_t i = 0; i < rows; i++){
        std::cout << "| ";
        for(size_t j = 0; j < cols; j++){
            std::cout << lu.L(i,j) << " ";
        }
        std::cout << "| ";
        for(size_t j = 0; j < cols; j++){
            std::cout << lu.U(i,j) << " ";
        }
        std::cout << "|\n";
    }
}