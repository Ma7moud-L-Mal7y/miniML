#include <iostream>
#include <iomanip>
#include "Dataset.hpp"
#include "Logistic_Regression.hpp"

int main() {
    try {
        std::cout << "============================================" << std::endl;
        std::cout << "   minml: Customer Churn Logistic Regression" << std::endl;
        std::cout << "============================================" << std::endl;

        // 1. Load the dataset
        std::string csv_path = "examples/customer_churn.csv";

        // Pass labelNums = 1 to indicate the last column ('churned') is the target y
        Dataset fullDataset(csv_path, 1);

        std::cout << "[+] Dataset successfully loaded!" << std::endl;
        std::cout << "    - Samples (Rows): " << fullDataset.getSampleNums() << std::endl;
        std::cout << "    - Features (X)  : " << fullDataset.getFeatureNums() << std::endl;
        std::cout << "    - Labels (y)    : " << fullDataset.getLabelNums() << std::endl;
        std::cout << std::endl;

        // 2. Normalize features BEFORE splitting -- fit stats on train only, later
        //    (we normalize the whole dataset for simplicity here; see note below
        //     for the train/test-safe version now that Dataset supports it)
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

        // 4. Normalize: fit on TRAIN, apply the same stats to TEST
        Matrix X_train = trainSet.normalize();          // computes + stores mean/std, returns normalized X
        Matrix y_train = trainSet.getY();

        Matrix X_test_raw = testSet.getX();
        Matrix X_test = trainSet.normalize(X_test_raw);  // apply TRAIN's stats to test data
        Matrix y_test = testSet.getY();

        // 5. Initialize and Train Logistic Regression
        LogisticRegression model;
        std::cout << "[+] Training Logistic Regression Model..." << std::endl;
        Matrix beta = model.fit(X_train, y_train);

        std::cout << "    Learned Weights (Beta):" << std::endl;
        beta.show();
        std::cout << std::endl;

        // 6. Predict on Held-Out Test Set
        Matrix y_prob  = model.predict(X_test);        // raw probabilities
        Matrix y_pred  = model.predictClass(X_test);    // thresholded 0/1

        // 7. Evaluate Performance Metrics
        ConfusionMatrix cm = classification_metrics::confusionMatrix(y_test, y_pred);
        double accuracy  = classification_metrics::computeAccuracy(cm);
        double precision = classification_metrics::computePrecision(cm);
        double recall    = classification_metrics::computeRecall(cm);
        double f1        = classification_metrics::computeF1Score(cm);
        double logloss   = classification_metrics::computeLogLoss(y_test, y_prob);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "============================================" << std::endl;
        std::cout << "            Model Evaluation Metrics        " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "  - Accuracy   : " << accuracy  << std::endl;
        std::cout << "  - Precision  : " << precision << std::endl;
        std::cout << "  - Recall     : " << recall    << std::endl;
        std::cout << "  - F1 Score   : " << f1        << std::endl;
        std::cout << "  - Log Loss   : " << logloss   << std::endl;
        std::cout << "============================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}