#include <gtest/gtest.h>
#include "Matrix.hpp"


// Utility helper to compare double values within threshold
constexpr double EPSILON = 1e-5;

// ============================================================================
// Constructors, Accessors, and Equality
// ============================================================================

TEST(MatrixTest, ConstructorDimensions) {
    Matrix m(3, 4);
    EXPECT_EQ(m.getRows(), 3);
    EXPECT_EQ(m.getCols(), 4);
}

TEST(MatrixTest, InitializerListConstructor) {
    Matrix m(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getCols(), 3);
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 5.0);
}

TEST(MatrixTest, CopyConstructor) {
    Matrix original(2, 2, {1.5, 2.5, 3.5, 4.5});
    Matrix copy(original);

    EXPECT_EQ(copy.getRows(), 2);
    EXPECT_EQ(copy.getCols(), 2);
    EXPECT_DOUBLE_EQ(copy(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(copy(1, 1), 4.5);

    // Verify deep copy modification independence
    copy(0, 0) = 99.0;
    EXPECT_DOUBLE_EQ(original(0, 0), 1.5);
}

TEST(MatrixTest, CopyAssignmentOperator) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(1, 1, {0.0});
    b = a;

    EXPECT_EQ(b.getRows(), 2);
    EXPECT_EQ(b.getCols(), 2);
    EXPECT_DOUBLE_EQ(b(1, 0), 3.0);
}

TEST(MatrixTest, CallOperatorBoundsCheck) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    const Matrix const_m = m;

    EXPECT_THROW(m(2, 0), std::range_error);
    EXPECT_THROW(const_m(0, 2), std::range_error);
}

TEST(MatrixTest, EqualityOperator) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix c(2, 2, {1.0, 2.0, 3.0, 4.001});

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(MatrixTest, IdentityFactory) {
    Matrix eye = Matrix::Identity(3);
    EXPECT_EQ(eye.getRows(), 3);
    EXPECT_EQ(eye.getCols(), 3);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(eye(r, c), (r == c) ? 1.0 : 0.0);
        }
    }
}

// ============================================================================
// Elementary Row Operations
// ============================================================================

TEST(MatrixTest, SwapRows) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    m.swapRows(0, 1);
    EXPECT_DOUBLE_EQ(m(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
}

TEST(MatrixTest, ScaleRow) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    m.scaleRow(0, 2.5);
    EXPECT_DOUBLE_EQ(m(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(m(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
}

TEST(MatrixTest, AddScaledRow) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    m.addScaledRow(1, 0, 2.0); // row1 = row1 + 2 * row0
    EXPECT_DOUBLE_EQ(m(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
}

// ============================================================================
// Matrix Arithmetic & Operators
// ============================================================================

TEST(MatrixTest, MatrixAddition) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    Matrix expected(2, 2, {6.0, 8.0, 10.0, 12.0});

    EXPECT_TRUE((a + b) == expected);

    Matrix mismatched(2, 3);
    EXPECT_THROW(a + mismatched, std::range_error);
}

TEST(MatrixTest, MatrixSubtraction) {
    Matrix a(2, 2, {5.0, 7.0, 9.0, 11.0});
    Matrix b(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix expected(2, 2, {4.0, 5.0, 6.0, 7.0});

    EXPECT_TRUE((a - b) == expected);
}

TEST(MatrixTest, MatrixMultiplication) {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix b(3, 2, {7.0, 8.0, 9.0, 1.0, 2.0, 3.0});
    Matrix expected(2, 2, {31.0, 19.0, 85.0, 55.0});

    EXPECT_TRUE((a * b) == expected);

    Matrix invalid_b(2, 2);
    EXPECT_THROW(a * invalid_b, std::range_error);
}

TEST(MatrixTest, InPlaceOperators) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {2.0, 2.0, 2.0, 2.0});

    a += b;
    EXPECT_DOUBLE_EQ(a(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(a(1, 1), 6.0);

    a -= b;
    EXPECT_DOUBLE_EQ(a(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(a(1, 1), 4.0);
}

// ============================================================================
// Scalar Operations
// ============================================================================

TEST(MatrixTest, ScalarOperations) {
    Matrix m(2, 2, {2.0, 4.0, 6.0, 8.0});

    EXPECT_TRUE((m + 2.0) == Matrix(2, 2, {4.0, 6.0, 8.0, 10.0}));
    EXPECT_TRUE((m - 2.0) == Matrix(2, 2, {0.0, 2.0, 4.0, 6.0}));
    EXPECT_TRUE((m * 2.0) == Matrix(2, 2, {4.0, 8.0, 12.0, 16.0}));

    // Test free functions
    EXPECT_TRUE((2.0 * m) == Matrix(2, 2, {4.0, 8.0, 12.0, 16.0}));
    EXPECT_TRUE((2.0 + m) == Matrix(2, 2, {4.0, 6.0, 8.0, 10.0}));
    EXPECT_TRUE((10.0 - m) == Matrix(2, 2, {8.0, 6.0, 4.0, 2.0}));
}

TEST(MatrixTest, InPlaceScalarOperators) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    m += 2.0;
    EXPECT_DOUBLE_EQ(m(0, 0), 3.0);
    m -= 1.0;
    EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
    m *= 3.0;
    EXPECT_DOUBLE_EQ(m(0, 0), 6.0);
}

// ============================================================================
// Transformations & Element-wise Operations
// ============================================================================

TEST(MatrixTest, Transpose) {
    Matrix m(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix t = m.transpose();

    EXPECT_EQ(t.getRows(), 3);
    EXPECT_EQ(t.getCols(), 2);
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(t(2, 1), 6.0);
}

TEST(MatrixTest, HadamardProduct) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    Matrix expected(2, 2, {5.0, 12.0, 21.0, 32.0});

    EXPECT_TRUE(a.hadamard(b) == expected);
}

TEST(MatrixTest, ApplyFunction) {
    Matrix m(2, 2, {0.0, 1.0, 4.0, 9.0});
    Matrix sqrts = m.apply(std::sqrt);

    EXPECT_DOUBLE_EQ(sqrts(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(sqrts(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(sqrts(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(sqrts(1, 1), 3.0);
}

// ============================================================================
// Slicing and Appending
// ============================================================================

TEST(MatrixTest, RowAndColSlicing) {
    Matrix m(3, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    });

    Matrix rSlice = m.rowSlice(0, 2); // Rows 0 and 1
    EXPECT_EQ(rSlice.getRows(), 2);
    EXPECT_EQ(rSlice.getCols(), 3);
    EXPECT_DOUBLE_EQ(rSlice(1, 2), 6.0);

    Matrix cSlice = m.colSlice(1, 3); // Cols 1 and 2
    EXPECT_EQ(cSlice.getRows(), 3);
    EXPECT_EQ(cSlice.getCols(), 2);
    EXPECT_DOUBLE_EQ(cSlice(0, 0), 2.0);
}

TEST(MatrixTest, AppendRowsAndCols) {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(1, 2, {5.0, 6.0});
    Matrix appRows = a.appendRows(b);

    EXPECT_EQ(appRows.getRows(), 3);
    EXPECT_EQ(appRows.getCols(), 2);
    EXPECT_DOUBLE_EQ(appRows(2, 0), 5.0);

    Matrix c(2, 1, {7.0, 8.0});
    Matrix appCols = a.appendCols(c);

    EXPECT_EQ(appCols.getRows(), 2);
    EXPECT_EQ(appCols.getCols(), 3);
    EXPECT_DOUBLE_EQ(appCols(0, 2), 7.0);
}

// ============================================================================
// Reductions and Aggregations
// ============================================================================

TEST(MatrixTest, GlobalSumAndMean) {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(m.sum(), 10.0);
    EXPECT_DOUBLE_EQ(m.mean(), 2.5);
}

TEST(MatrixTest, AxisSumAndMean) {
    Matrix m(3, 2, {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    });

    // Axis 0: Collapse down rows (result 1 x cols)
    Matrix sumAxis0 = m.sum(0);
    EXPECT_EQ(sumAxis0.getRows(), 1);
    EXPECT_EQ(sumAxis0.getCols(), 2);
    EXPECT_DOUBLE_EQ(sumAxis0(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(sumAxis0(0, 1), 12.0);

    // Axis 1: Collapse across columns (result rows x 1)
    Matrix meanAxis1 = m.mean(1);
    EXPECT_EQ(meanAxis1.getRows(), 3);
    EXPECT_EQ(meanAxis1.getCols(), 1);
    EXPECT_DOUBLE_EQ(meanAxis1(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(meanAxis1(2, 0), 5.5);
}

// ============================================================================
// Matrix Checks & Properties
// ============================================================================

TEST(MatrixTest, ChecksAndProperties) {
    Matrix sq(2, 2, {1.0, 2.0, 2.0, 1.0});
    Matrix rect(2, 3);

    EXPECT_TRUE(sq.isSquare());
    EXPECT_FALSE(rect.isSquare());

    EXPECT_TRUE(sq.isSymmetric());
    Matrix nonSym(2, 2, {1.0, 3.0, 2.0, 1.0});
    EXPECT_FALSE(nonSym.isSymmetric());
}

// ============================================================================
// Linear Algebra Solvers, LU Decomposition, Rank & Determinant
// ============================================================================

TEST(MatrixTest, LUDecompositionAndDeterminant) {
    Matrix m(3, 3, {
        2.0, -1.0, 0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0
    });

    luDecomposition lu = m.luDecompose();
    
    // Check L is lower triangular with 1s on diagonal
    for (size_t r = 0; r < 3; ++r) {
        EXPECT_DOUBLE_EQ(lu.L(r, r), 1.0);
    }

    // det(m) should be 4.0
    EXPECT_NEAR(m.determinant(), 4.0, EPSILON);
    EXPECT_FALSE(m.isSingular());
}

TEST(MatrixTest, SingularMatrixCheck) {
    Matrix singular(2, 2, {1.0, 2.0, 2.0, 4.0});
    EXPECT_TRUE(singular.isSingular());
    EXPECT_NEAR(singular.determinant(), 0.0, EPSILON);
}

TEST(MatrixTest, Rank) {
    Matrix fullRank(2, 2, {1.0, 0.0, 0.0, 1.0});
    Matrix rankDeficient(2, 2, {1.0, 2.0, 2.0, 4.0});

    EXPECT_EQ(fullRank.rank(), 2);
    EXPECT_EQ(rankDeficient.rank(), 1);
}

TEST(MatrixTest, LinearSolveAndInverse) {
    // System Ax = b
    Matrix A(2, 2, {2.0, 1.0, 1.0, 3.0});
    Matrix b(2, 1, {5.0, 5.0});

    Matrix x = A.solve(b);
    EXPECT_NEAR(x(0, 0), 2.0, EPSILON);
    EXPECT_NEAR(x(1, 0), 1.0, EPSILON);

    // Test matrix inverse property (A * A^-1 = I)
    Matrix invA = A.inverse();
    Matrix identity = A * invA;
    EXPECT_TRUE(identity == Matrix::Identity(2));
}