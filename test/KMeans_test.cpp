#include <gtest/gtest.h>
#include "KMeans.hpp"
#include "Dataset.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <fstream>

TEST(KMeanUnitTest, ConvergesToKnownCentroidsOnSeparatedClusters) {
    std::ifstream probe("test/KMeanSyntheticClusters.csv");
    if (!probe.good()) {
        GTEST_SKIP() << "Dataset file not found at " << "test/KMeanSyntheticClusters.csv";
    }
    probe.close();

    Dataset ds("test/KMeanSyntheticClusters.csv", 0);
    ASSERT_EQ(ds.getSampleNums(), 8u);
    Matrix X = ds.getX();

    KMean model(2, 100, 1e-8);
    model.fit(X);

    EXPECT_TRUE(model.isConverged());

    // Inertia should match the hand-computed value closely.
    EXPECT_NEAR(model.getInertia(), 4.0, 1e-6);

    // Labels: rows 0-3 must share a label, rows 4-7 must share a
    // (different) label — exact integer id is not guaranteed, so we
    // check relationships, not fixed values.
    std::vector<size_t> predictions = model.predict(X);
    ASSERT_EQ(predictions.size(), 8u);

    for (size_t i = 1; i < 4; i++)
        EXPECT_EQ(predictions[i], predictions[0]);
    for (size_t i = 5; i < 8; i++)
        EXPECT_EQ(predictions[i], predictions[4]);
    EXPECT_NE(predictions[0], predictions[4]);

    // A brand-new point clearly nearer group A should predict group A's label.
    Matrix nearGroupA(1, 2);
    nearGroupA(0,0) = 1.5; nearGroupA(0,1) = 1.5;
    std::vector<size_t> predA = model.predict(nearGroupA);
    EXPECT_EQ(predA[0], predictions[0]);
}