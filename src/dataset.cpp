#include "Dataset.hpp"

//constructors

Dataset::Dataset(const Matrix& x,const Matrix& y)
    : featureMatrix(x), labelMatrix(y)
{

}
Dataset::Dataset(const std::string& filepath,size_t labelNums)
    : featureMatrix(0,0), labelMatrix(0,0)
{
    std :: ifstream file(filepath);
    if(!file.is_open()){
        throw std:: runtime_error("could not open file: "+filepath);
    }
    std ::vector<std::string> allNames;
    std:: string line;
    std::getline(file,line);
    std::stringstream headerstream(line);
    std::string name;
    while(std::getline(headerstream,name,',')){
        allNames.push_back(name);
    }
    size_t allCols=allNames.size();
    size_t featureNums=allCols-labelNums;
    for(size_t i=0;i<featureNums;i++){
        featureNames.push_back(allNames[i]);
    }
    for(size_t i=featureNums;i<allCols;i++){
        labelNames.push_back(allNames[i]);
    }
    size_t rowNums=0;
    while(std::getline(file,line)){
        rowNums++;
    }
    featureMatrix=Matrix(rowNums,featureNums);
    labelMatrix=Matrix(rowNums,labelNums);
    file.close();
    file.open(filepath);
    std ::getline(file,line);
    size_t r=0;
    while(std::getline(file,line)){
        std::stringstream linestream(line);
        std:: string piece;
        size_t c=0;
        while(std::getline(linestream,piece,',')){
            double value=std::stod(piece);
            if(c<featureNums){
                featureMatrix(r,c)=value;
            }
            else{
                labelMatrix(r,c-featureNums)=value;
            }
            c++;
        }
        r++;
    }
}

//get dimensions

size_t Dataset::getSamples()const{
    return featureMatrix.getRows();
}
size_t Dataset::getFeatures()const{
    return featureMatrix.getCols();
}
size_t Dataset::getLabels()const{
    return labelMatrix.getCols();
}

//showing dataset

void Dataset::show(size_t start, size_t end)const{
    size_t sampleNums=featureMatrix.getRows();
    if(start>sampleNums||start>end||end>sampleNums||start==end){
        throw std::invalid_argument("invalid starting or ending position");
    }
    size_t featureNums=featureMatrix.getCols();
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << std::setw(10) << featureNames[j];
    }
    std::cout << ' ';
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << std::setw(10) << labelNames[j]; 
    }
    std::cout << "\n";
    for(size_t i=start;i<end;i++){
        for(size_t j=0;j<featureNums;j++){
            std::cout << std::setw(10) << featureMatrix(i,j);
        }
        std ::cout<< ' ';
        for(size_t j=0;j<labelNums;j++){
            std::cout << std::setw(10) << labelMatrix(i,j);
        }
        std::cout<< "\n";
    }
}

void Dataset::show()const{
    size_t sampleNums=featureMatrix.getRows();  
    (*this).show(0,sampleNums);
}

void Dataset:: showFeatures()const{
    size_t featureNums=featureMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << featureNames[j] << ' ';
    }
    std::cout<< '\n';
    featureMatrix.show();
}
void Dataset::showLabels()const{
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << labelNames[j] << ' ';
    }
    std::cout<< '\n';
    labelMatrix.show();
}
void Dataset::showFeatureNames()const{
    size_t featureNums=featureMatrix.getCols();
    for (size_t j = 0; j < featureNums; j++) {
        std::cout << featureNames[j] << ' ';
    }
}
void Dataset::showLabelNames()const{
    size_t labelNums= labelMatrix.getCols();
    for (size_t j = 0; j < labelNums; j++) {
        std::cout << labelNames[j] << ' ';
    }
}
void Dataset::head()const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<5){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(0,5);
}
void Dataset::head(size_t n)const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<n){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(0,n);
}

void Dataset::tail()const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<5){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(sampleNums-5,sampleNums);
}
void Dataset::tail(size_t n)const{
    size_t sampleNums=featureMatrix.getRows();  
    if(sampleNums<n){
        (*this).show(0,sampleNums);
        return;
    }
    (*this).show(sampleNums-n,sampleNums);
}



//get matrices

Matrix Dataset::getX()const{
    return featureMatrix;
}
Matrix Dataset::getY()const{
    return labelMatrix;
}