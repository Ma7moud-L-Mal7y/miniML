#include "Matrix.hpp"
#include "Dataset.hpp"


double quad(double x){
    return pow(x, 2);
}


int main(){
    Matrix A(3,3);

    double x = A.determinant();
    std::cout << x << '\n';
    
    Dataset s("C:/Users/boody/Downloads/small_dataset.csv");
    s.show();
    std::cout<<'\n';
    s.shuffle();
    s.show();
    return 0;
}