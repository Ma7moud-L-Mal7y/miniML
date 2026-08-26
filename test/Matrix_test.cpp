// test/LogisticRegression_test.cpp
#include "Dataset.hpp"
#include "Logistic_Regression.hpp"
#include <iostream>

int main() {
    // 1. Load the small dataset (age, height, weight -> is_athlete)
    Dataset data("small_dataset.csv", 1);

    std::cout << "=== Loaded Dataset ===\n";
    data.show();
    data.normalize();
    std::cout<<"\n\n";
    data.show();

    // 2. Get raw X and y
    Matrix X = data.getX();
    Matrix Y = data.getY();

    // 3. Train logistic regression
    LogisticRegression model(X, Y, 100000, 0.01);
    Matrix learnedBeta = model.fit();

    std::cout << "\n=== Learned Beta ===\n";
    model.showBeta();

    // 4. Predict probabilities on the SAME data (sanity check, not real evaluation)
    Matrix probs = model.predict(X);
    std::cout << "\n=== Predicted Probabilities ===\n";
    probs.show();

    // 5. Predict classes (0/1)
    Matrix classes = model.predictClass(X);
    std::cout << "\n=== Predicted Classes ===\n";
    classes.show();

    // 6. Compare predicted classes to actual labels, manually
    std::cout << "\n=== Actual vs Predicted ===\n";
    size_t correct = 0;
    size_t total = Y.getRows();
    for (size_t i = 0; i < total; i++) {
        double actual = Y(i, 0);
        double predicted = classes(i, 0);
        std::cout << "Row " << i << " | actual: " << actual
                   << " | predicted: " << predicted << "\n";
        if (actual == predicted) correct++;
    }

    double accuracy = static_cast<double>(correct) / total;
    std::cout << "\nManual accuracy: " << accuracy * 100 << "%\n";

    return 0;
}