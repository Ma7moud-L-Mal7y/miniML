#include "KMeans.hpp"

KMean::KMean(size_t k_, size_t maxIter_, double tol_)
    : k_(k_), maxIter_(maxIter_), tol_(tol_), centroids_(std::nullopt)
{

}

void KMean::fit(const Matrix& trainX){
    // dimensions
    size_t n = trainX.getRows();
    size_t d = trainX.getCols();

    // random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(0, n-1);

    // choose the first centroid randomly 
    size_t randIdx = distrib(gen);
    centroids_ = trainX.rowSlice(randIdx, randIdx + 1);

    // k-means++ loop
    std::vector<double> minDistSqr(n, std::numeric_limits<double>::infinity());
    for(size_t k = 1; k < k_; k++){
        // distance between centroid and each n sample point
        Matrix distSqr = euclidDist(trainX, centroids_.value());

        // update minDistSqr
        double totalDistSqr = 0.0;
        for(size_t i = 0; i < n; i++){
            minDistSqr[i] = std::min(distSqr(i, distSqr.getCols() - 1), minDistSqr[i]);
            totalDistSqr += minDistSqr[i];
        }

        // generate the next target centroid randomly between 0 and totalDistSqr
        std::uniform_real_distribution<double> distribReal(0.0, totalDistSqr);
        double target = distribReal(gen);

        // Sample next centroid using D(x)^2 weighted probabilities
        size_t nextIdx = n-1;
        double cumulative = 0.0;
        for(size_t i = 0; i < n; i++){
            cumulative += minDistSqr[i];
            if(cumulative >= target){
                nextIdx = i;
                break;
            }
        }

        // append the new centroid to centroids_
        Matrix newCentroid = trainX.rowSlice(nextIdx, nextIdx + 1);
        centroids_ = centroids_.value().appendRows(newCentroid);
    }

    // Lloyd algorithm loop
    for(size_t iter = 0; iter < maxIter_; iter++){
        // clear labels for the new iteration
        labels_.clear();
        inertia_ = 0.0;
    
        // assign each point to the nearest centroid
        Matrix dist = euclidDist(trainX, centroids_.value());
        for(size_t i = 0; i < n; i++){
            double minDist = std::numeric_limits<double>::infinity();
            size_t minIdx = UINT64_MAX;
            for(size_t j = 0; j < k_; j++){
                if(dist(i, j) < minDist){
                    minDist = dist(i, j);
                    minIdx = j;
                }
            }
            inertia_ += minDist;
            labels_.push_back(minIdx);
        }

        // taking mean of cluster
        std::vector<size_t> pointCount(k_, 0);
        Matrix pointSum(k_, d, 0.0);
        for(size_t i = 0; i < n; i++){
            pointCount[labels_[i]]++;
            for(size_t j = 0; j < d; j++){
                pointSum(labels_[i], j) += trainX(i, j);
            }
        }
        
        // update centroids = mean of cluster && checking tolerance
        Matrix oldCentroids(centroids_.value());
        for(size_t i = 0; i < k_; i++){
            if(pointCount[i] == 0)
                continue;

            for(size_t j = 0; j < d; j++){
                centroids_.value()(i, j) = pointSum(i, j) / static_cast<double>(pointCount[i]);
            }
        }

        double shift = (centroids_.value() - oldCentroids).apply(std::fabs).sum();
        if(shift < tol_){
            converged = true;
            break;
        }
    }
}

std::vector<size_t> KMean::predict(const Matrix& testX){
    if(!centroids_.has_value())
        throw std::runtime_error("No centroids are initialized");

    Matrix D = euclidDist(testX, centroids_.value());
    size_t n = D.getRows();
    std::vector<size_t> result;
    for(size_t i = 0; i < n; i++){
        double minElement = std::numeric_limits<double>::infinity();
        size_t minIdx = UINT64_MAX;
        for(size_t j = 0; j < centroids_.value().getRows(); j++){
            if(D(i, j) < minElement){
                minElement = D(i, j);
                minIdx = j;
            }
        }
        result[i] = minIdx;
    }

    return result;
}

// evaluation
double KMean::getInertia() const{
    return inertia_;
}

double KMean::getSilhouette() const{
    return 0.0;
}

double KMean::getDaviesBouldin() const{
    return 0.0;
}

// check
bool KMean::isConverged() const{
    return converged;
}

// helper functions
Matrix euclidDist(const Matrix& X, const Matrix& C){
    size_t n = X.getRows();
    size_t k = C.getRows();
    size_t d = X.getCols();

    if(d != C.getCols())
        throw std::length_error("dimensions are not compatible");

    Matrix result(n, k);
    // result(i,j) = SUM{X_i2} - 2 X*C_t + SUM{C_j2}
    Matrix X_i  = X.hadamard(X).sum(1);
    Matrix C_j  = C.hadamard(C).sum(1);
    Matrix midTerm = (X * C.transpose());
    midTerm *= -2;

    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < k; j++){
            result(i, j) = X_i(i, 0) + midTerm(i, j) + C_j(j, 0);
        }
    }

    return result;
}