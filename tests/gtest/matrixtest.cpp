#include <gtest/gtest.h>
#include "pssolver.hpp"

using namespace pssolver;

// VectorTest fixture
// Necessary for templates
template <class T>
class MatrixTest : public ::testing::Test
{
protected:
    MatrixTest() : Matrix_(new Matrix<T>(10, 10, 20)) {}
    virtual ~MatrixTest() { delete Matrix_; }
    Matrix<T>* const Matrix_;
};

typedef ::testing::Types<double, float> MyTypes;

TYPED_TEST_CASE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, MatrixConstructors)
{

    Matrix<TypeParam> A;
    EXPECT_EQ(0, A.GetNRows());
    EXPECT_EQ(0, A.GetNCols());
    EXPECT_EQ(0, A.GetNnz());

    EXPECT_TRUE(A.GetFormat() == CSR);
    
    Matrix<TypeParam> B(10, 15, 20);
    EXPECT_EQ(10, B.GetNRows());
    EXPECT_EQ(15, B.GetNCols());
    EXPECT_EQ(20, B.GetNnz());

    Matrix<TypeParam> C;
    C.AllocateCOO(7, 14, 21);
    EXPECT_EQ(7, C.GetNRows());
    EXPECT_EQ(14, C.GetNCols());
    EXPECT_EQ(21, C.GetNnz());

    C.ReadFile("matrix.mtx");


}


int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}