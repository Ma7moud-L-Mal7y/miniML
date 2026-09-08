#include <optional>
#include <random>
#include <algorithm>

#include "Matrix.hpp"


class KMean{
public:
    // constructors
    KMean(size_t k_ = 5, size_t maxIter_ = 1000, double tol_ = 1e-8);

    // train
    void fit(const Matrix& trainX);

    // test
    std::vector<size_t> predict(const Matrix& testX);

    // evaluate 
    double getInertia() const;
    double getSilhouette(const Matrix& X) const;

    // check
    bool isConverged() const;

private:
    size_t k_;
    size_t maxIter_;
    double tol_;

    std::optional<Matrix> centroids_;
    std::vector<size_t> labels_;
    double inertia_ = 0.0;
    bool converged = false;
};

// helper functions
Matrix euclidDist(const Matrix& X, const Matrix& point);
double silhouetteScore(const Matrix& X, const std::vector<size_t>& labels);