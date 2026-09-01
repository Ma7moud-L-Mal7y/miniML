#include <gtest/gtest.h>
#include "Linear_Regression.hpp"
#include "Dataset.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <fstream>

// =====================================================================
// Integration: Dataset -> train/test split -> fit -> predict -> metrics
// Run against multiple CSVs to confirm the pipeline generalizes beyond
// a single dataset's shape/column count/target semantics.
//
// ASSUMPTION (same as before): Dataset(path, 1) treats the LAST column
// as the single numeric label column, and handles any non-numeric
// feature columns internally (e.g. FuelConsumptionCo2's MAKE/MODEL/
// VEHICLECLASS/TRANSMISSION/FUELTYPE text columns). Adjust here if
// Dataset's actual contract differs.
// =====================================================================

TEST(LinearRegressionIntegrationTest, FitsAndPredictsOnFuelConsumptionCsv) {
    std::ifstream probe("examples/FuelConsumptionCo2.csv");
        if (!probe.good()) {
            GTEST_SKIP() << "Dataset file not found at " << "examples/FuelConsumptionCo2.csv";
        }
        probe.close();

        Dataset ds("examples/FuelConsumptionCo2.csv", 1);
        ASSERT_GT(ds.getSampleNums(), 0u);
        ds.selectFeatures({"ENGINESIZE", "CYLINDERS"});

        size_t splitPoint = static_cast<size_t>(ds.getSampleNums() * 0.8);
        ASSERT_GT(splitPoint, 0u);
        ASSERT_LT(splitPoint, ds.getSampleNums());

        trainTest split = ds.trainTestSplit(splitPoint);

        Matrix trainX = addOnesCol(split.trainSample.getX());
        Matrix trainY = split.trainSample.getY();
        Matrix testX = addOnesCol(split.testSample.getX());
        Matrix testY = split.testSample.getY();

        LinearRegression model(trainX, trainY);
        EXPECT_NO_THROW(model.fit());

        Matrix predictions = model.predict(testX);
        ASSERT_EQ(predictions.getRows(), testY.getRows());

        double r2 = metrics::computeR2(testY, predictions);
        double rmse = metrics::computeRMSE(testY, predictions);

        EXPECT_FALSE(std::isnan(r2));
        EXPECT_FALSE(std::isnan(rmse));
        EXPECT_GE(rmse, 0.0);
        EXPECT_LE(r2, 1.0 + 1e-9);
}

TEST(LinearRegressionIntegrationTest, FitsAndPredictsOnECommerceSalesCsv) {
    std::ifstream probe("examples/E-Commerce Sales Analytics.csv");
        if (!probe.good()) {
            GTEST_SKIP() << "Dataset file not found at " << "examples/E-Commerce Sales Analytics.csv";
        }
        probe.close();

        Dataset ds("examples/E-Commerce Sales Analytics.csv", 1);
        ASSERT_GT(ds.getSampleNums(), 0u);

        size_t splitPoint = static_cast<size_t>(ds.getSampleNums() * 0.8);
        ASSERT_GT(splitPoint, 0u);
        ASSERT_LT(splitPoint, ds.getSampleNums());

        trainTest split = ds.trainTestSplit(splitPoint);

        Matrix trainX = addOnesCol(split.trainSample.getX());
        Matrix trainY = split.trainSample.getY();
        Matrix testX = addOnesCol(split.testSample.getX());
        Matrix testY = split.testSample.getY();

        LinearRegression model(trainX, trainY);
        EXPECT_NO_THROW(model.fit());

        Matrix predictions = model.predict(testX);
        ASSERT_EQ(predictions.getRows(), testY.getRows());

        double r2 = metrics::computeR2(testY, predictions);
        double rmse = metrics::computeRMSE(testY, predictions);

        EXPECT_FALSE(std::isnan(r2));
        EXPECT_FALSE(std::isnan(rmse));
        EXPECT_GE(rmse, 0.0);
        EXPECT_LE(r2, 1.0 + 1e-9);
}