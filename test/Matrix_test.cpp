// test/LogisticRegression_test.cpp
#include "Dataset.hpp"
#include "Logistic_Regression.hpp"
#include <iostream>

int main() {
    // 1. Load the massive dataset: 14 feature columns (10 numeric + 4 categorical),
    //    2 label columns (risk_score, is_high_risk)
    Dataset data("massive_learnable_dataset.csv", 2);

    std::cout << "=== Loaded Dataset (first 5 rows) ===\n";
    data.head();

    std::cout << "\nFeature columns (after one-hot expansion): " << data.getFeatureNums() << "\n";
    std::cout << "Label columns: " << data.getLabelNums() << "\n";

    // 2. Normalize features BEFORE splitting/shuffling
    data.normalize();

    // 3. Shuffle so the split isn't biased by original row order
    data.shuffle();

    // 4. Split into train/test (80% train, 20% test)
    size_t totalRows = data.getSampleNums();
    size_t splitPoint = static_cast<size_t>(0.8 * totalRows);

    trainTest split = data.trainTestSplit(splitPoint);

    std::cout << "\nTrain rows: " << split.trainSample.getSampleNums();
    std::cout << " | Test rows: " << split.testSample.getSampleNums() << "\n";

    // 5. Extract X and Y, then slice Y down to just is_high_risk (column index 1)
    //    since LogisticRegression expects a single-column label matrix
Matrix trainX = split.trainSample.getX();
Matrix fullTrainY = split.trainSample.getY();
Matrix trainY = fullTrainY.colSlice(1, 2);

std::cout << "trainX: " << trainX.getRows() << " x " << trainX.getCols() << "\n";
std::cout << "trainY: " << trainY.getRows() << " x " << trainY.getCols() << "\n";

Matrix testX = split.testSample.getX();
Matrix fullTestY = split.testSample.getY();
Matrix testY = fullTestY.colSlice(1, 2);

std::cout << "testX: " << testX.getRows() << " x " << testX.getCols() << "\n";
std::cout << "testY: " << testY.getRows() << " x " << testY.getCols() << "\n";

LogisticRegression model(trainX, trainY, 1000, 0.01);
model.fit();

    std::cout << "\n=== Learned Beta ===\n";
    model.showBeta();

    // 7. Evaluate on test set
    Matrix testPredictions = model.predictClass(testX);

    size_t correct = 0;
    size_t total = testY.getRows();
    for (size_t i = 0; i < total; i++) {
        if (testPredictions(i, 0) == testY(i, 0)) correct++;
    }
    double testAccuracy = static_cast<double>(correct) / total;
    std::cout << "\nTest set accuracy: " << testAccuracy * 100 << "%\n";

    // 8. Also report train accuracy for overfitting comparison
    Matrix trainPredictions = model.predictClass(trainX);
    size_t trainCorrect = 0;
    size_t trainTotal = trainY.getRows();
    for (size_t i = 0; i < trainTotal; i++) {
        if (trainPredictions(i, 0) == trainY(i, 0)) trainCorrect++;
    }
    double trainAccuracy = static_cast<double>(trainCorrect) / trainTotal;
    std::cout << "Train set accuracy: " << trainAccuracy * 100 << "%\n";

    return 0;
}