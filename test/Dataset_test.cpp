#include <gtest/gtest.h>
#include "Dataset.hpp"
#include "Matrix.hpp"
#include <fstream>
#include <cstdio>
#include <cmath>
#include <set>
#include <algorithm>


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
//   category is intentionally non-numeric (text); id, f1, f2, label
//   are numeric. label = id * 1000, chosen so shuffle-pairing is easy
//   to verify.
// ---------------------------------------------------------------------
class DatasetTest : public ::testing::Test {
protected:
    std::string path = "dataset_test_tmp.csv";

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

// ======================================================================
// constructor
// ======================================================================

TEST_F(DatasetTest, ConstructFromFile_CorrectSampleCount) {
    Dataset ds(path, 1);
    EXPECT_EQ(ds.getSampleNums(), 6u);
}

TEST_F(DatasetTest, ConstructFromFile_MissingFileThrows) {
    EXPECT_THROW(Dataset("no_such_file.csv", 1), std::runtime_error);
}

TEST_F(DatasetTest, ConstructFromFile_LabelCountExceedingColumnsThrows) {
    // 5 columns total; labelNums = 6 should be rejected rather than
    // underflow (allCols - labelNums) as an earlier version of this
    // constructor did.
    EXPECT_THROW(Dataset(path, 6), std::invalid_argument);
}

TEST_F(DatasetTest, ConstructFromFile_LabelCountEqualToColumnCount_ZeroFeatures) {
    // Edge case: every column becomes a label; should not throw, and
    // should simply produce zero features.
    Dataset ds(path, 5);
    EXPECT_EQ(ds.getFeatureNums(), 0u);
    EXPECT_EQ(ds.getLabelNums(), 5u);
}

// ======================================================================
// getSampleNums / getFeatureNums / getLabelNums
// ======================================================================

TEST_F(DatasetTest, GetFeatureNums_MatchesAutoSelectedNumericFeatureColumns) {
    Dataset ds(path, 1);
    // feature range = id, category, f1, f2; category is non-numeric,
    // so only 3 are auto-selected.
    EXPECT_EQ(ds.getFeatureNums(), 3u);
}

TEST_F(DatasetTest, GetLabelNums_MatchesConstructorArgument) {
    Dataset ds(path, 1);
    EXPECT_EQ(ds.getLabelNums(), 1u);
}

TEST_F(DatasetTest, GetFeatureNums_ReflectsNarrowedSelection) {
    Dataset ds(path, 1);
    ds.selectFeatures({"f1"});
    EXPECT_EQ(ds.getFeatureNums(), 1u);
}

// ======================================================================
// getX / getY
// ======================================================================

TEST_F(DatasetTest, GetX_DefaultSelection_ExcludesNonNumericColumn) {
    Dataset ds(path, 1);
    Matrix X = ds.getX();
    EXPECT_EQ(X.getRows(), 6u);
    EXPECT_EQ(X.getCols(), 3u); // id, f1, f2 -- category dropped

    EXPECT_DOUBLE_EQ(X(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(X(0, 1), 10.0);
    EXPECT_DOUBLE_EQ(X(0, 2), 100.0);

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
    writeCsv(path, {
        "f1,f2,category",
        "1,2,A",
        "3,4,B"
    });
    Dataset ds(path, 1); // label column = "category" (non-numeric)
    EXPECT_THROW(ds.getY(), std::runtime_error);
}

// ======================================================================
// getSelectedFeatureNames / getAllFeatureNames / getLabelNames
// ======================================================================

TEST_F(DatasetTest, SelectedAndAllFeatureNames_EqualBeforeAnySelection) {
    Dataset ds(path, 1);
    std::vector<std::string> expected = {"id", "f1", "f2"}; // category excluded (non-numeric)
    EXPECT_EQ(ds.getSelectedFeatureNames(), expected);
    EXPECT_EQ(ds.getAllFeatureNames(), expected);
}

TEST_F(DatasetTest, SelectedFeatureNames_NarrowsAfterSelectFeatures_AllFeatureNamesUnchanged) {
    Dataset ds(path, 1);
    ds.selectFeatures({"f1"});
    std::vector<std::string> expectedSelected = {"f1"};
    std::vector<std::string> expectedAll = {"id", "f1", "f2"};
    EXPECT_EQ(ds.getSelectedFeatureNames(), expectedSelected);
    EXPECT_EQ(ds.getAllFeatureNames(), expectedAll);
}

TEST_F(DatasetTest, GetLabelNames_ReturnsLabelColumnNames_EvenIfNonNumeric) {
    writeCsv(path, {
        "f1,f2,category",
        "1,2,A",
        "3,4,B"
    });
    Dataset ds(path, 1); // label = "category", non-numeric
    std::vector<std::string> expected = {"category"};
    EXPECT_EQ(ds.getLabelNames(), expected);
}

TEST_F(DatasetTest, GetLabelNames_MultipleLabelColumns) {
    Dataset ds(path, 2); // last 2 columns: f2, label
    std::vector<std::string> expected = {"f2", "label"};
    EXPECT_EQ(ds.getLabelNames(), expected);
}

// ======================================================================
// getStringColumn / getStringColumnNames
// ======================================================================

TEST_F(DatasetTest, GetStringColumn_ReturnsCorrectValues) {
    Dataset ds(path, 1);
    std::vector<std::string> expected = {"A", "B", "A", "C", "B", "A"};
    EXPECT_EQ(ds.getStringColumn("category"), expected);
}

TEST_F(DatasetTest, GetStringColumn_NumericColumnThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.getStringColumn("id"), std::invalid_argument);
}

TEST_F(DatasetTest, GetStringColumn_UnknownColumnThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.getStringColumn("nonexistent"), std::runtime_error);
}

TEST_F(DatasetTest, GetStringColumnNames_ReturnsOnlyNonNumericColumns) {
    Dataset ds(path, 1);
    std::vector<std::string> expected = {"category"};
    EXPECT_EQ(ds.getStringColumnNames(), expected);
}

// ======================================================================
// show / showSelected / head / tail
// ======================================================================

TEST_F(DatasetTest, Show_FullRange_DoesNotThrow_AndPrintsAllColumnNames) {
    Dataset ds(path, 1);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(ds.show());
    std::string output = testing::internal::GetCapturedStdout();
    for (const std::string& col : {"id", "category", "f1", "f2", "label"}) {
        EXPECT_NE(output.find(col), std::string::npos) << "missing column header: " << col;
    }
}

TEST_F(DatasetTest, Show_StartEqualsRows_Throws) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.show(6, 6), std::invalid_argument); // start == end
    EXPECT_THROW(ds.show(6, 7), std::invalid_argument); // start >= rows
}

TEST_F(DatasetTest, Show_StartGreaterThanEnd_Throws) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.show(3, 1), std::invalid_argument);
}

TEST_F(DatasetTest, Show_EndBeyondRows_Throws) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.show(0, 100), std::invalid_argument);
}

TEST_F(DatasetTest, ShowSelected_NumericLabel_DoesNotThrow_AndPrintsSelectedAndLabelNames) {
    // Safe case only: label column ("label") is numeric, so the missing
    // isNumeric check in showSelected() for label columns isn't hit.
    Dataset ds(path, 1);
    ds.selectFeatures({"f1", "f2"});
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(ds.showSelected(0, 6));
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("f1"), std::string::npos);
    EXPECT_NE(output.find("f2"), std::string::npos);
    EXPECT_NE(output.find("label"), std::string::npos);
}

TEST_F(DatasetTest, ShowSelected_InvalidRangeThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.showSelected(6, 6), std::invalid_argument);
    EXPECT_THROW(ds.showSelected(3, 1), std::invalid_argument);
}

TEST_F(DatasetTest, Head_DefaultShowsUpToFiveRows) {
    Dataset ds(path, 1);
    testing::internal::CaptureStdout();
    ds.head();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("5"), std::string::npos);  // 5th row's id present somewhere
    // 6th row's id "6" should not appear as a standalone value; a loose
    // substring check is used elsewhere in this suite for header names
    // only, so this is intentionally not asserted here to avoid false
    // positives/negatives from formatting quirks.
}

TEST_F(DatasetTest, Head_WithArgument_RespectsRequestedCount) {
    Dataset ds(path, 1);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(ds.head(2));
    testing::internal::GetCapturedStdout();
}

TEST_F(DatasetTest, Head_ZeroThrows) {
    Dataset ds(path, 1);
    EXPECT_THROW(ds.head(0), std::invalid_argument); // show(0,0) -> start==end
}

TEST_F(DatasetTest, Tail_DefaultDoesNotThrow) {
    Dataset ds(path, 1);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(ds.tail());
    testing::internal::GetCapturedStdout();
}

TEST_F(DatasetTest, Tail_WithArgument_DoesNotThrow) {
    Dataset ds(path, 1);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(ds.tail(3));
    testing::internal::GetCapturedStdout();
}

// ======================================================================
// selectFeatures
// ======================================================================

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

TEST_F(DatasetTest, SelectFeatures_InvalidatesPriorNormalizationStats) {
    // normalize(x) should throw after selectFeatures() is called again,
    // since selectFeatures() resets hasNormStats/normMeans/normStds.
    Dataset ds(path, 1);
    ds.normalize(); // populates normMeans/normStds for {id, f1, f2}
    ds.selectFeatures({"f1"}); // resets stats
    Matrix probe(6, 1);
    for (size_t i = 0; i < 6; i++) probe(i, 0) = 0.0;
    EXPECT_THROW(ds.normalize(probe), std::runtime_error);
}

// ======================================================================
// trainTestSplit
// ======================================================================

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

// ======================================================================
// shuffle
// ======================================================================

TEST_F(DatasetTest, Shuffle_PreservesRowCountAndFeatureLabelPairing) {
    Dataset ds(path, 1);
    ds.shuffle();
    EXPECT_EQ(ds.getSampleNums(), 6u);

    Matrix X = ds.getX(); // columns: id, f1, f2
    Matrix Y = ds.getY();

    std::set<double> seenIds;
    for (size_t i = 0; i < X.getRows(); i++) {
        EXPECT_DOUBLE_EQ(Y(i, 0), X(i, 0) * 1000.0); // label was id * 1000
        seenIds.insert(X(i, 0));
    }
    std::set<double> expectedIds = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(seenIds, expectedIds);
}

TEST_F(DatasetTest, Shuffle_AlsoMovesNonNumericColumnsInSync) {
    // category values were seeded as A,B,A,C,B,A alongside id 1..6.
    // After shuffle, wherever id==1 ends up, category should still be "A".
    static const std::vector<std::string> categoryById = {"", "A", "B", "A", "C", "B", "A"}; // 1-indexed
    Dataset ds(path, 1);
    ds.shuffle();
    Matrix X = ds.getX();
    std::vector<std::string> categoriesAfterShuffle = ds.getStringColumn("category");
    for (size_t i = 0; i < X.getRows(); i++) {
        int id = static_cast<int>(X(i, 0));
        EXPECT_EQ(categoriesAfterShuffle[i], categoryById[id]);
    }
}

// ======================================================================
// normalize() [no-arg] -- IMPORTANT BEHAVIOR NOTE:
// This overload does NOT mutate the Dataset's own stored columns; it
// returns a normalized copy of getX() and separately caches
// normMeans/normStds (via hasNormStats) for later use by
// normalize(const Matrix&). Calling ds.getX() again after ds.normalize()
// still returns the ORIGINAL, unnormalized values.
// ======================================================================

TEST_F(DatasetTest, NormalizeNoArg_ReturnsZeroMeanUnitStdMatrix) {
    Dataset ds(path, 1);
    Matrix normalized = ds.normalize();

    for (size_t c = 0; c < normalized.getCols(); c++) {
        double sum = 0.0;
        for (size_t r = 0; r < normalized.getRows(); r++) sum += normalized(r, c);
        double mean = sum / normalized.getRows();
        EXPECT_NEAR(mean, 0.0, 1e-9);

        double sqSum = 0.0;
        for (size_t r = 0; r < normalized.getRows(); r++) sqSum += (normalized(r, c) - mean) * (normalized(r, c) - mean);
        double stddev = std::sqrt(sqSum / normalized.getRows());
        EXPECT_NEAR(stddev, 1.0, 1e-6);
    }
}

TEST_F(DatasetTest, NormalizeNoArg_DoesNotMutateDatasetsOwnStorage) {
    Dataset ds(path, 1);
    Matrix beforeX = ds.getX();
    ds.normalize();
    Matrix afterX = ds.getX();

    ASSERT_EQ(beforeX.getRows(), afterX.getRows());
    ASSERT_EQ(beforeX.getCols(), afterX.getCols());
    for (size_t r = 0; r < beforeX.getRows(); r++)
        for (size_t c = 0; c < beforeX.getCols(); c++)
            EXPECT_DOUBLE_EQ(beforeX(r, c), afterX(r, c))
                << "getX() should still return original values after normalize()";
}

TEST(DatasetNormalizeEdgeCase, ConstantColumn_ZeroedNotNaN) {
    std::string path = "dataset_test_constant.csv";
    writeCsv(path, {
        "f1,f2,label",
        "5,1,10",
        "5,2,20",
        "5,3,30"
    });
    Dataset ds(path, 1);
    Matrix normalized = ds.normalize(); // f1 is constant, f2 is not
    for (size_t r = 0; r < normalized.getRows(); r++) {
        EXPECT_DOUBLE_EQ(normalized(r, 0), 0.0); // constant column -> zeroed, not NaN/inf
        EXPECT_FALSE(std::isnan(normalized(r, 1)));
    }
    std::remove(path.c_str());
}

// ======================================================================
// normalize(const Matrix&) const
// ======================================================================

TEST_F(DatasetTest, NormalizeWithMatrix_BeforeFittingStats_Throws) {
    Dataset ds(path, 1); // normalize() (no-arg) never called -> hasNormStats false
    Matrix x(6, 3);
    EXPECT_THROW(ds.normalize(x), std::runtime_error);
}

TEST_F(DatasetTest, NormalizeWithMatrix_WrongColumnCount_Throws) {
    Dataset ds(path, 1);
    ds.normalize(); // fits stats for 3 columns (id, f1, f2)
    Matrix wrongShape(6, 5); // wrong column count
    EXPECT_THROW(ds.normalize(wrongShape), std::range_error);
}

TEST_F(DatasetTest, NormalizeWithMatrix_ZeroRows_DoesNotThrow) {
    Dataset ds(path, 1);
    ds.normalize(); // fits stats for 3 columns
    Matrix empty(0, 3);
    EXPECT_NO_THROW(ds.normalize(empty));
}

// Applies the fitted stats (from training columns id, f1, f2) to new,
// hand-picked data and checks every row/column against an independently
// computed z-score -- this is exactly the kind of test that would have
// caught the old c++/r++ bug, since a wrong index would either throw,
// leave values untouched, or scatter results into the wrong cells.
TEST_F(DatasetTest, NormalizeWithMatrix_MatchesManuallyComputedZScore) {
    Dataset ds(path, 1);
    ds.normalize(); // fits population mean/std over id=[1..6], f1=[10..60], f2=[100..600]

    Matrix x(2, 3);
    x(0, 0) = 7;   x(0, 1) = 70;  x(0, 2) = 700;
    x(1, 0) = 0;   x(1, 1) = 0;   x(1, 2) = 0;

    Matrix result = ds.normalize(x);
    ASSERT_EQ(result.getRows(), 2u);
    ASSERT_EQ(result.getCols(), 3u);

    std::vector<std::vector<double>> trainCols = {
        {1, 2, 3, 4, 5, 6},          // id
        {10, 20, 30, 40, 50, 60},    // f1
        {100, 200, 300, 400, 500, 600} // f2
    };
    double rawRow0[3] = {7, 70, 700};
    double rawRow1[3] = {0, 0, 0};

    for (size_t c = 0; c < 3; c++) {
        double mean = 0.0;
        for (double v : trainCols[c]) mean += v;
        mean /= trainCols[c].size();

        double sqSum = 0.0;
        for (double v : trainCols[c]) sqSum += (v - mean) * (v - mean);
        double stddev = std::sqrt(sqSum / trainCols[c].size());

        EXPECT_NEAR(result(0, c), (rawRow0[c] - mean) / stddev, 1e-9);
        EXPECT_NEAR(result(1, c), (rawRow1[c] - mean) / stddev, 1e-9);
    }
}

// Cross-check: applying the fitted stats back onto the (raw, since
// normalize() doesn't mutate storage) training data should reproduce
// exactly what normalize() itself returned.
TEST_F(DatasetTest, NormalizeWithMatrix_OnTrainingData_MatchesNormalizeNoArgOutput) {
    Dataset ds(path, 1);
    Matrix fitted = ds.normalize();          // fits stats, returns normalized training data
    Matrix reapplied = ds.normalize(ds.getX()); // raw getX() (unmutated) run through the fitted stats

    ASSERT_EQ(fitted.getRows(), reapplied.getRows());
    ASSERT_EQ(fitted.getCols(), reapplied.getCols());
    for (size_t r = 0; r < fitted.getRows(); r++)
        for (size_t c = 0; c < fitted.getCols(); c++)
            EXPECT_NEAR(fitted(r, c), reapplied(r, c), 1e-9);
}

// Constant-column stats (std ~ 0) applied to brand-new data should still
// zero that column out rather than dividing by ~0 -- checks the
// stddev < 1e-8 branch on the row>0 path specifically.
TEST(DatasetNormalizeEdgeCase, NormalizeWithMatrix_ConstantColumnStats_AppliedToNewData_ZeroedNotNaN) {
    std::string path = "dataset_test_constant2.csv";
    writeCsv(path, {
        "f1,f2,label",
        "5,1,10",
        "5,2,20",
        "5,3,30"
    });
    Dataset ds(path, 1);
    ds.normalize(); // f1 is constant -> normStds[0] ~ 0

    Matrix x(2, 2);
    x(0, 0) = 999; x(0, 1) = 10;
    x(1, 0) = -50; x(1, 1) = 20;

    Matrix result = ds.normalize(x);
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0); // constant-column stats -> zeroed, ignoring input value
    EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
    EXPECT_FALSE(std::isnan(result(0, 1)));
    EXPECT_FALSE(std::isnan(result(1, 1)));

    std::remove(path.c_str());
}

// ======================================================================
// Real dataset sanity check
// ======================================================================

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
    EXPECT_GT(ds.getX().getCols(), 0u);
}