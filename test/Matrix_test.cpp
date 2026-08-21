#include "Matrix.hpp"

int main(){
    Matrix A(3, 2, {1,2,3,4,5,6});
    Matrix B(2, 4, {3,2,3,2,3,2,3,2});
    Matrix C(4, 2, {1,2,4,-8,3,-5,6,7});
    A = A + 0.25;
    B = B + 3.5673;
    Matrix D(A * B * C);

    A.show();
    printf("\n");
    B.show();
    printf("\n");
    B.transpose().show();
    printf("\n");
    C.show();
    printf("\n");
    C.show();
    printf("\n");
    D.show();
    printf("\n");
    

    return 0;
}