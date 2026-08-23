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
        throw std:: runtime_error("could not open file"+filepath);
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
    for(int i=0;i<featureNums;i++){
        featureNames.push_back(allNames[i]);
    }
    for(int i=featureNums;i<allCols;i++){
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
        size_t c;
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