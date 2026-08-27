#include <iostream>
#include <iomanip>
#include "Dataset.hpp"
#include "Linear_Regression.hpp"

int main() {
    try {
        std::cout << "============================================" << std::endl;
        std::cout << "   minml: E-Commerce Sales Linear Regression" << std::endl;
        std::cout << "============================================" << std::endl;

        // 1. Load the dataset
        // Relative path from the build folder or project root
        std::string csv_path = "examples/E-Commerce Sales Analytics.csv";
        
        // Pass labelNums = 1 to indicate the last column ('revenue') is the target y
        Dataset fullDataset(csv_path, 1);

        std::cout << "[+] Dataset successfully loaded!" << std::endl;
        std::cout << "    - Samples (Rows): " << fullDataset.getSampleNums() << std::endl;
        std::cout << "    - Features (X)  : " << fullDataset.getFeatureNums() << std::endl;
        std::cout << "    - Labels (y)    : " << fullDataset.getLabelNums() << std::endl;
        std::cout << std::endl;

        // 2. Shuffle dataset to break order bias
        fullDataset.shuffle();

        // 3. Perform 80/20 Train/Test Split
        size_t total_samples = fullDataset.getSampleNums();
        size_t split_index = static_cast<size_t>(total_samples * 0.8);

        trainTest split = fullDataset.trainTestSplit(split_index);
        
        Dataset trainSet = split.trainSample;
        Dataset testSet  = split.testSample;

        std::cout << "[+] Train/Test Split Completed:" << std::endl;
        std::cout << "    - Training Samples: " << trainSet.getSampleNums() << std::endl;
        std::cout << "    - Testing Samples : " << testSet.getSampleNums() << std::endl;
        std::cout << std::endl;

        // 4. Extract Feature and Label Matrices
        Matrix X_train = trainSet.getX();
        Matrix y_train = trainSet.getY();

        Matrix X_test  = testSet.getX();
        Matrix y_test  = testSet.getY();

        // 5. Initialize and Train Linear Regression
        LinearRegression model;
        std::cout << "[+] Training Linear Regression Model..." << std::endl;
        Matrix beta = model.fit(X_train, y_train);

        std::cout << "    Learned Weights (Beta):" << std::endl;
        beta.show();
        std::cout << std::endl;

        // 6. Predict on Held-Out Test Set
        Matrix y_pred = model.predict(X_test);

        // 7. Evaluate Performance Metrics
        double mse  = metrics::computeMSE(y_test, y_pred);
        double rmse = metrics::computeRMSE(y_test, y_pred);
        double mae  = metrics::computeMAE(y_test, y_pred);
        double r2   = metrics::computeR2(y_test, y_pred);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "============================================" << std::endl;
        std::cout << "            Model Evaluation Metrics        " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "  - MAE  (Mean Absolute Error) : " << mae  << std::endl;
        std::cout << "  - MSE  (Mean Squared Error)  : " << mse  << std::endl;
        std::cout << "  - RMSE (Root Mean Sq Error)  : " << rmse << std::endl;
        std::cout << "  - R²   (Coefficient of Det.) : " << r2   << std::endl;
        std::cout << "============================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}