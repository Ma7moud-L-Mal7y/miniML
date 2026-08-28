#include <gtest/gtest.h>
#include "Linear_Regression.hpp"
#include "Dataset.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <fstream>

namespace {
    Matrix columnVector(const std::vector<double>& vals) {
        Matrix m(vals.size(), 1);
        for (size_t i = 0; i < vals.size(); i++) m(i, 0) = vals[i];
        return m;
    }

    Matrix matrixFromRows(const std::vector<std::vector<double>>& rows) {
        size_t r = rows.size(), c = rows.empty() ? 0 : rows[0].size();
        Matrix m(r, c);
        for (size_t i = 0; i < r; i++)
            for (size_t j = 0; j < c; j++)
                m(i, j) = rows[i][j];
        return m;
    }
}

// =====================================================================
// addOnesCol
// =====================================================================

TEST(AddOnesColTest, AppendsColumnOfOnesAtEnd) {
    Matrix A = matrixFromRows({{1, 2}, {3, 4}});
    Matrix result = addOnesCol(A);

    EXPECT_EQ(result.getRows(), 2u);
    EXPECT_EQ(result.getCols(), 3u);

    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 1.0); // appended ones col

    EXPECT_DOUBLE_EQ(result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 2), 1.0);
}

// =====================================================================
// LinearRegression -- fit / predict, using a noiseless linear relation
// y = 2*x1 + 3*x2 + 5  (intercept 5, appended as LAST beta element
// since addOnesCol appends ones at the end, not the front)
// =====================================================================

class LinearRegressionPerfectFitTest : public ::testing::Test {
protected:
    Matrix Xones = addOnesCol(matrixFromRows({
        {1, 1},
        {2, 1},
        {1, 2},
        {3, 2},
        {2, 3}
    }));
    Matrix Y = columnVector({10, 12, 13, 17, 18}); // 2*x1 + 3*x2 + 5
};

TEST_F(LinearRegressionPerfectFitTest, Constructor_FitNoArgs_RecoversTrueCoefficients) {
    LinearRegression model(Xones, Y);
    Matrix beta = model.fit();

    ASSERT_EQ(beta.getRows(), 3u);
    EXPECT_NEAR(beta(0, 0), 2.0, 1e-6); // x1 coefficient
    EXPECT_NEAR(beta(1, 0), 3.0, 1e-6); // x2 coefficient
    EXPECT_NEAR(beta(2, 0), 5.0, 1e-6); // intercept (last, from addOnesCol)
}

TEST_F(LinearRegressionPerfectFitTest, Predict_AfterFit_MatchesTrainingLabelsClosely) {
    LinearRegression model(Xones, Y);
    model.fit();
    Matrix predictions = model.predict(Xones);

    ASSERT_EQ(predictions.getRows(), Y.getRows());
    for (size_t i = 0; i < Y.getRows(); i++) {
        EXPECT_NEAR(predictions(i, 0), Y(i, 0), 1e-6);
    }
}

TEST_F(LinearRegressionPerfectFitTest, Fit_TwoArgOverload_ReturnsSameBetaAsNoArgOverload) {
    LinearRegression model; // default constructed, no X/y yet
    Matrix beta = model.fit(Xones, Y);

    EXPECT_NEAR(beta(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(beta(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(beta(2, 0), 5.0, 1e-6);

    Matrix predictions = model.predict(Xones);
    for (size_t i = 0; i < Y.getRows(); i++) {
        EXPECT_NEAR(predictions(i, 0), Y(i, 0), 1e-6);
    }
}

// =====================================================================
// Guard clauses: using the model before it has what it needs
// =====================================================================

TEST(LinearRegressionGuardsTest, FitNoArgs_WithoutXY_Throws) {
    LinearRegression model; // default ctor: X, y both nullopt
    EXPECT_THROW(model.fit(), std::runtime_error);
}

TEST(LinearRegressionGuardsTest, Predict_BeforeFit_Throws) {
    LinearRegression model; // beta is nullopt
    Matrix dummy = matrixFromRows({{1, 1}});
    EXPECT_THROW(model.predict(dummy), std::runtime_error);
}

TEST(LinearRegressionGuardsTest, ShowBeta_BeforeFit_Throws) {
    LinearRegression model;
    EXPECT_THROW(model.showBeta(), std::runtime_error);
}

TEST(LinearRegressionGuardsTest, ShowBeta_AfterFit_DoesNotThrow) {
    Matrix Xones = addOnesCol(matrixFromRows({{1}, {2}, {3}}));
    Matrix Y = columnVector({3, 5, 7}); // y = 2x + 1
    LinearRegression model(Xones, Y);
    model.fit();

    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(model.showBeta());
    testing::internal::GetCapturedStdout(); // just drain it, content format unknown
}

// =====================================================================
// metrics:: -- known-value checks
// y = [1,2,3], y_hat = [0,0,0]
// SSE = 1+4+9 = 14 | SST (mean=2) = 1+0+1 = 2
// MSE = 14/3 | MAE = (1+2+3)/3 = 2 | RMSE = sqrt(14/3)
// =====================================================================

class MetricsKnownValuesTest : public ::testing::Test {
protected:
    Matrix y = columnVector({1, 2, 3});
    Matrix yHatZero = columnVector({0, 0, 0});
    Matrix yHatPerfect = columnVector({1, 2, 3});
};

TEST_F(MetricsKnownValuesTest, SSE_MatchesHandComputedValue) {
    EXPECT_NEAR(metrics::computeSSE(y, yHatZero), 14.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, SST_MatchesHandComputedValue) {
    EXPECT_NEAR(metrics::computeSST(y), 2.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, MSE_MatchesHandComputedValue) {
    EXPECT_NEAR(metrics::computeMSE(y, yHatZero), 14.0 / 3.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, MAE_MatchesHandComputedValue) {
    EXPECT_NEAR(metrics::computeMAE(y, yHatZero), 2.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, RMSE_IsSqrtOfMSE) {
    double expected = std::sqrt(14.0 / 3.0);
    EXPECT_NEAR(metrics::computeRMSE(y, yHatZero), expected, 1e-9);
}

TEST_F(MetricsKnownValuesTest, MaxError_MatchesHandComputedValue) {
    Matrix yHat = columnVector({0, 10, 3}); // errors: 1, -5, 0
    EXPECT_NEAR(metrics::computeMaxError(y, yHat), 8.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, R2_IsOne_ForPerfectPredictions) {
    EXPECT_NEAR(metrics::computeR2(y, yHatPerfect), 1.0, 1e-9);
}

TEST_F(MetricsKnownValuesTest, R2_IsBetweenZeroAndOne_ForMeanBaseline) {
    // predicting the mean for everything gives SSE == SST, so R2 should be 0
    Matrix yHatMean = columnVector({2, 2, 2});
    EXPECT_NEAR(metrics::computeR2(y, yHatMean), 0.0, 1e-9);
}

// Known quirk: when y has ~zero variance, SST < epsilon short-circuits
// computeR2 to return 1.0 regardless of how bad y_hat is. This documents
// current behavior rather than asserting it's the "right" answer --
// worth being aware of if you ever evaluate on a near-constant target.
TEST(MetricsQuirkTest, R2_ReturnsOne_WhenYIsConstant_EvenWithBadPredictions) {
    Matrix yConst = columnVector({5, 5, 5});
    Matrix yHatBad = columnVector({100, -3, 7});
    EXPECT_DOUBLE_EQ(metrics::computeR2(yConst, yHatBad), 1.0);
}

TEST(MetricsAdjustedR2Test, MatchesItsOwnFormula_ForSaneFeatureCount) {
    Matrix y = columnVector({1, 2, 3, 4});
    Matrix yHat = columnVector({1.1, 1.9, 3.2, 3.8});
    size_t features = 1;

    double r2 = metrics::computeR2(y, yHat);
    size_t n = y.getRows();
    double expectedAdjustedR2 = 1.0 - (1.0 - r2) * (static_cast<double>(n - 1) / static_cast<double>(n - features - 1));

    EXPECT_NEAR(metrics::computeAdjustedR2(y, yHat, features), expectedAdjustedR2, 1e-9);
}

// =====================================================================
// Integration: Dataset -> train/test split -> fit -> predict -> metrics
// ASSUMPTION: last CSV column ("CO2EMISSIONS" in the standard
// FuelConsumptionCo2 dataset) is numeric and is the single label column.
// Adjust labelNums/column assumptions here if your file differs.
// =====================================================================

TEST(LinearRegressionIntegrationTest, FitsAndPredictsOnFuelConsumptionCsv) {
    const std::string path = "examples/FuelConsumptionCo2.csv";
    std::ifstream probe(path);
    if (!probe.good()) {
        GTEST_SKIP() << "Dataset file not found at " << path;
    }
    probe.close();

    Dataset ds(path, 1);
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

    Matrix predictions = model.predict(testX); // throws (and fails the test) if predict() throws
    ASSERT_EQ(predictions.getRows(), testY.getRows());

    double r2 = metrics::computeR2(testY, predictions);
    double rmse = metrics::computeRMSE(testY, predictions);

    EXPECT_FALSE(std::isnan(r2));
    EXPECT_FALSE(std::isnan(rmse));
    EXPECT_LE(r2, 1.0 + 1e-9); // R2 can be negative for a bad model, but never > 1
}