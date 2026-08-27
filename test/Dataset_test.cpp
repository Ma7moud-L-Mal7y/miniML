#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "Dataset.hpp"

class DatasetTest : public ::testing::Test {
protected:
    std::string csvPath = "examples/E-Commerce Sales Analytics.csv";

    void SetUp() override {
        // Create a temporary Kaggle-style CSV for testing
        std::ofstream out(csvPath);
        out << "sqft,bedrooms,price\n";
        out << "1000,2,200000\n";
        out << "1500,3,300000\n";
        out << "2000,4,400000\n";
        out << "2500,4,500000\n";
        out << "3000,5,600000\n";
        out.close();
    }

    void TearDown() override {
        // Clean up test artifact after running
        std::remove(csvPath.c_str());
    }
};

TEST_F(DatasetTest, ParseCSVAndDimensions) {
    // Parse CSV where the last 1 column is the label target (price)
    Dataset ds(csvPath, 1);

    EXPECT_EQ(ds.getSampleNums(), 5);
    EXPECT_EQ(ds.getFeatureNums(), 2);
    EXPECT_EQ(ds.getLabelNums(), 1);

    auto fNames = ds.getFeatureNames();
    auto lNames = ds.getLabelNames();
    
    EXPECT_EQ(fNames[0], "sqft");
    EXPECT_EQ(fNames[1], "bedrooms");
    EXPECT_EQ(lNames[0], "price");

    // Check first sample feature and label values
    EXPECT_DOUBLE_EQ(ds.getX()(0, 0), 1000.0);
    EXPECT_DOUBLE_EQ(ds.getY()(0, 0), 200000.0);
}

TEST_F(DatasetTest, TrainTestSplit) {
    Dataset ds(csvPath, 1);

    // Split at row index 3 (3 train samples: rows 0,1,2; 2 test samples: rows 3,4)
    trainTest split = ds.trainTestSplit(3);

    EXPECT_EQ(split.trainSample.getSampleNums(), 3);
    EXPECT_EQ(split.testSample.getSampleNums(), 2);

    // Verify train dataset values
    EXPECT_DOUBLE_EQ(split.trainSample.getX()(2, 0), 2000.0);
    EXPECT_DOUBLE_EQ(split.trainSample.getY()(2, 0), 400000.0);

    // Verify test dataset values
    EXPECT_DOUBLE_EQ(split.testSample.getX()(0, 0), 2500.0);
    EXPECT_DOUBLE_EQ(split.testSample.getY()(0, 0), 500000.0);
}

TEST_F(DatasetTest, ShufflePreservesPairing) {
    Dataset ds(csvPath, 1);
    ds.shuffle();

    EXPECT_EQ(ds.getSampleNums(), 5);

    // Ensure features (X) and labels (Y) stayed aligned after shuffling
    Matrix X = ds.getX();
    Matrix Y = ds.getY();
    for (size_t i = 0; i < ds.getSampleNums(); ++i) {
        double sqft = X(i, 0);
        double price = Y(i, 0);
        // Verify target relation: price == sqft * 200
        EXPECT_DOUBLE_EQ(price, sqft * 200.0);
    }
}