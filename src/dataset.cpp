#include "Dataset.hpp"

//constructors

Dataset::Dataset(const Matrix& x,const Matrix& y)
    : featureMatrix(x), labelMatrix(y)
{

}
Dataset::Dataset(const std::string& filepath){
    
}