#include <gtest/gtest.h>
#include "Dataset.hpp"
#include "Matrix.hpp"
#include <fstream>
#include <cstdio>
#include <cmath>
#include <set>

// =====================================================================
// NOTE: Dataset.hpp declares getFeatureNums(), getLabelNums(),
// getFeatureNames(), getLabelNames(), show()/show(start,end),
// showFeatures(), showLabels(), showFeatureNames(), showLabelNames(),
// head()/head(n), tail()/tail(n) -- none of these are defined in the
// .cpp provided. Tests for them are intentionally omitted here because
// calling them would fail at LINK time and take down the whole test
// binary. Add them back once implementations exist.
// =====================================================================

namespace {
    void writeCsv(const std::string& path, const std::vector<std::string>& lines) {
        std::ofstream file(path);
        for (const auto& line : lines) file << line << "\n";
        file.close();
    }
}

// ---------------------------------------------------------------------
// Fixture: mixed numeric + text CSV
//   columns: id, category, f1, f2, label   (labelNums = 1)
//   category is intentionally non-numeric (text), the rest are numeric.
//   label = id * 1000, chosen so shuffle-pairing is easy to verify.
// ---------------------------------------------------------------------
class DatasetTest : public ::testing::Test {
protected:
    std::string path = "temp_test_dataset.csv";

    void SetUp() override {
        writeCsv(path, {
            "id,category,f1,f2,label",
            "1,A,10,100,1000",
            "2,B,20,200,2000",
            "3,A,30,300,3000",
            "4,C,40,400,4000",
            "5,B,50,500,5000",
            "6,A,60,600,6000"
        });
    }

    void TearDown() override {
        std::remove(path.c_str());
    }
};

// ------------------------------ construction / getSampleNums -------------

TEST_F(DatasetTest, ConstructFromFile_CorrectSampleCount) {
    Dataset ds(path, 1);
    EXPECT_EQ(ds.getSampleNums(), 6u);
}

TEST_F(DatasetTest, ConstructFromFile_MissingFileThrows) {
    EXPECT_THROW(Dataset("no_such_file.csv", 1), std::runtime_error);
}

// ------------------------------ getX / getY (default selection) ----------

// category is non-numeric so it should be silently excluded from the
// auto-selected features; only id, f1, f2 (in that column order) should
// end up in getX() by default.
TEST_F(DatasetTest, GetX_DefaultSelection_ExcludesNonNumericColumn) {
    Dataset ds(path, 1);
    Matrix X = ds.getX();
    EXPECT_EQ(X.getRows(), 6u);
    EXPECT_EQ(X.getCols(), 3u); // id, f1, f2 -- category dropped

    // row 0: id=1, f1=10, f2=100
    EXPECT_DOUBLE_EQ(X(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(X(0, 1), 10.0);
    EXPECT_DOUBLE_EQ(X(0, 2), 100.0);

    // row 5: id=6, f1=60, f2=600
    EXPECT_DOUBLE_EQ(X(5, 0), 6.0);
    EXPECT_DOUBLE_EQ(X(5, 1), 60.0);
    EXPECT_DOUBLE_EQ(X(5, 2), 600.0);
}

TEST_F(DatasetTest, GetY_CorrectValues) {
    Dataset ds(path, 1);
    Matrix Y = ds.getY();
    EXPECT_EQ(Y.getRows(), 6u);
    EXPECT_EQ(Y.getCols(), 1u);
    EXPECT_DOUBLE_EQ(Y(0, 0), 1000.0);
    EXPECT_DOUBLE_EQ(Y(5, 0), 6000.0);
}

TEST_F(DatasetTest, GetY_NonNumericLabelColumnThrows) {
    // category as the label column (last column) -> getY() should throw
    // since label extraction requires numeric data.
    writeCsv(path, {
        "f1,f2,category",
        "1,2,A",
        "3,4,B"
    });
    Dataset ds(path, 1); // label column = "category"
    EXPECT_THROW(ds.getY(), std::runtime_error);
}

// ------------------------------ selectFeatures ----------------------------

TEST_F(DatasetTest, SelectFeatures_ValidSubset_ChangesGetX) {
    Dataset ds(path, 1);
    ds.selectFeatures({"f1", "f2"});
    Matrix X = ds.getX();
    EXPECT_EQ(X.getCols(), 2u);
    EXPECT_DOUBLE_EQ(X(0, 0), 10.0);  // f1
    EXPECT_DOUBLE_EQ(X(0, 1), 100.0); // f2
}

TEST_F(DatasetTest, SelectFeatures_VectorAndInitializerListAgree) {
    Dataset ds1(path, 1);
    Dataset ds2(path, 1);
    ds1.selectFeatures(std::vector<std::string>{"id", "f1"});
    ds2.selectFeatures({"id", "f1"});
    Matrix X1 = ds1.getX();
    Matrix X2 = ds2.getX();
    EXPECT_EQ(X1.getCols(), X2.getCols());
    for (size_t r = 0; r < X1.getRows(); r++)
        for (size_t c = 0; c < X1.getCols(); c++)
            EXPECT_DOUBLE_EQ(X1(r, c), X2(r, c));
}

TEST_F(DatasetTest, SelectFeatures_DuplicateNameThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.selectFeatures({"f1", "f1"}), std::invalid_argument);
}

TEST_F(DatasetTest, SelectFeatures_UnknownColumnThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.selectFeatures({"nonexistent"}), std::runtime_error);
}

TEST_F(DatasetTest, SelectFeatures_LabelColumnRejected) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.selectFeatures({"label"}), std::invalid_argument);
}

TEST_F(DatasetTest, SelectFeatures_NonNumericColumnRejected) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.selectFeatures({"category"}), std::invalid_argument);
}

// ------------------------------ trainTestSplit ----------------------------

TEST_F(DatasetTest, TrainTestSplit_SinglePoint_CorrectSizes) {
    Dataset ds(path, 1);
    trainTest split = ds.trainTestSplit(4);
    EXPECT_EQ(split.trainSample.getSampleNums(), 4u);
    EXPECT_EQ(split.testSample.getSampleNums(), 2u);

    Matrix trainX = split.trainSample.getX();
    Matrix testX = split.testSample.getX();
    EXPECT_DOUBLE_EQ(trainX(0, 0), 1.0); // id of first train row
    EXPECT_DOUBLE_EQ(testX(0, 0), 5.0);  // id of first test row
}

TEST_F(DatasetTest, TrainTestSplit_SinglePoint_ZeroThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.trainTestSplit(0), std::invalid_argument);
}

TEST_F(DatasetTest, TrainTestSplit_SinglePoint_OutOfRangeThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.trainTestSplit(6), std::invalid_argument);
    EXPECT_THROW(ds.trainTestSplit(100), std::invalid_argument);
}

TEST_F(DatasetTest, TrainTestSplit_FourArg_NonOverlapping_Works) {
    Dataset ds(path, 1);
    trainTest split = ds.trainTestSplit(0, 3, 3, 6);
    EXPECT_EQ(split.trainSample.getSampleNums(), 3u);
    EXPECT_EQ(split.testSample.getSampleNums(), 3u);
}

TEST_F(DatasetTest, TrainTestSplit_FourArg_Overlapping_Throws) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.trainTestSplit(0, 4, 2, 6), std::range_error);
}

TEST_F(DatasetTest, TrainTestSplit_FourArg_OutOfBoundsThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.trainTestSplit(0, 3, 3, 100), std::range_error);
}

TEST_F(DatasetTest, TrainTestSplit_FourArg_EmptyRangeThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.trainTestSplit(3, 3, 3, 6), std::invalid_argument); // empty train
}

TEST_F(DatasetTest, TrainTestSplit_PreservesSelectedFeatures) {
    Dataset ds(path, 1);
    ds.selectFeatures({"f1"});
    trainTest split = ds.trainTestSplit(3);
    EXPECT_EQ(split.trainSample.getX().getCols(), 1u);
    EXPECT_EQ(split.testSample.getX().getCols(), 1u);
}

// ------------------------------ shuffle -----------------------------------

TEST_F(DatasetTest, Shuffle_PreservesRowCountAndFeatureLabelPairing) {
    Dataset ds(path, 1);
    ds.shuffle();
    EXPECT_EQ(ds.getSampleNums(), 6u);

    Matrix X = ds.getX(); // columns: id, f1, f2
    Matrix Y = ds.getY();

    std::set<double> seenIds;
    for (size_t i = 0; i < X.getRows(); i++) {
        // label was constructed as id * 1000 -- verify rows moved together
        EXPECT_DOUBLE_EQ(Y(i, 0), X(i, 0) * 1000.0);
        seenIds.insert(X(i, 0));
    }
    std::set<double> expectedIds = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(seenIds, expectedIds);
}

// ------------------------------ normalize ----------------------------------

TEST_F(DatasetTest, Normalize_ProducesZeroMeanUnitStd) {
    Dataset ds(path, 1);
    ds.normalize();
    Matrix X = ds.getX(); // id, f1, f2 -- all numeric feature columns

    for (size_t c = 0; c < X.getCols(); c++) {
        double sum = 0.0;
        for (size_t r = 0; r < X.getRows(); r++) sum += X(r, c);
        double mean = sum / X.getRows();
        EXPECT_NEAR(mean, 0.0, 1e-9);

        double sqSum = 0.0;
        for (size_t r = 0; r < X.getRows(); r++) sqSum += (X(r, c) - mean) * (X(r, c) - mean);
        double stddev = std::sqrt(sqSum / X.getRows()); // population std, matches implementation
        EXPECT_NEAR(stddev, 1.0, 1e-6);
    }
}

TEST(DatasetNormalizeEdgeCase, ConstantColumnBecomesZeroNotNaN) {
    std::string path = "dataset_test_constant.csv";
    writeCsv(path, {
        "f1,f2,label",
        "5,1,10",
        "5,2,20",
        "5,3,30"
    });
    Dataset ds(path, 1);
    ds.normalize();
    Matrix X = ds.getX(); // f1 (constant), f2
    for (size_t r = 0; r < X.getRows(); r++) {
        EXPECT_DOUBLE_EQ(X(r, 0), 0.0); // constant column -> zeroed, not NaN
        EXPECT_FALSE(std::isnan(X(r, 1)));
    }
    std::remove(path.c_str());
}

// ------------------------- real dataset sanity check ------------------------

// With column-level numeric detection, loading the real e-commerce CSV
// should now succeed (Date/Category/Region/PaymentMethod just get
// excluded from the auto-selected numeric features), and TotalSales
// (the last column) is numeric, so getY() should not throw either.
TEST(DatasetRealFileTest, RealCsvLoadsAndYieldsNumericLabel) {
    const std::string realPath = "examples/E-Commerce Sales Analytics.csv";
    std::ifstream probe(realPath);
    if (!probe.good()) {
        GTEST_SKIP() << "Real dataset file not found at " << realPath;
    }
    probe.close();

    Dataset ds(realPath, 1);
    EXPECT_GT(ds.getSampleNums(), 0u);
    EXPECT_NO_THROW(ds.getY());
    EXPECT_GT(ds.getX().getCols(), 0u); // at least the numeric columns auto-selected
}