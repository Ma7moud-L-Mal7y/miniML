#include "Matrix.hpp"


double quad(double x){
    return pow(x, 2);
}


int main(){
    Matrix A(3,3);

    double x = A.determinant();
    std::cout << x << '\n';
    

    return 0;
}