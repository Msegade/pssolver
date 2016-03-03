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
    C.Allocate(7, 14, 21, COO);
    EXPECT_EQ(7, C.GetNRows());
    EXPECT_EQ(14, C.GetNCols());
    EXPECT_EQ(21, C.GetNnz());
    C.Allocate(7, 14, 21, CSR);

    C.ReadFile("../tests/matrices/matrix.mtx");
    EXPECT_TRUE(C.GetFormat() == CSR);
    EXPECT_EQ(5, C.GetNRows());
    EXPECT_EQ(5, C.GetNCols());
    EXPECT_EQ(13, C.GetNnz());
    EXPECT_EQ(2, C(4,4));
    EXPECT_EQ(-1, C(4,3));
    Matrix<TypeParam> D;
    D = C;
    EXPECT_TRUE(C.GetFormat() == CSR);
    EXPECT_EQ(5, C.GetNRows());
    EXPECT_EQ(5, C.GetNCols());
    EXPECT_EQ(13, C.GetNnz());
    EXPECT_EQ(2, C(4,4));
    EXPECT_EQ(-1, C(4,3));


}

TYPED_TEST(MatrixTest, MatrixDataManipulation)
{

    Matrix<TypeParam> A;
    A.ReadFile("../tests/matrices/matrix.mtx");

    A.MoveToDevice();
    EXPECT_TRUE(A.IsDevice());

    EXPECT_TRUE(A.GetFormat() == CSR);
    EXPECT_EQ(5, A.GetNRows());
    EXPECT_EQ(5, A.GetNCols());
    EXPECT_EQ(13, A.GetNnz());
    EXPECT_EQ(2, A(4,4));
    EXPECT_EQ(-1, A(4,3));

    A.Allocate(100, 100, 121, CSR);
    EXPECT_EQ(100, A.GetNRows());
    EXPECT_EQ(100, A.GetNCols());
    EXPECT_EQ(121, A.GetNnz());

    A.MoveToHost();
    EXPECT_TRUE(A.IsHost());
    EXPECT_EQ(100, A.GetNRows());
    EXPECT_EQ(100, A.GetNCols());
    EXPECT_EQ(121, A.GetNnz());

    Matrix<TypeParam> B;
    B = A;
    EXPECT_TRUE(A.GetFormat() == CSR);
    EXPECT_TRUE(A.IsHost());
    EXPECT_EQ(100, A.GetNRows());
    EXPECT_EQ(100, A.GetNCols());
    EXPECT_EQ(121, A.GetNnz());

}

TYPED_TEST(MatrixTest, MatrixOperationsHost)
{
    Matrix<TypeParam> A;
    A.ReadFile("../tests/matrices/matrix.mtx");

    Vector<TypeParam> b(A.GetNRows(), 1.0);
    Vector<TypeParam> c(A.GetNRows());

    c = A*b;
    EXPECT_EQ(1.0, c[0]);
    for (int i = 1; i < c.GetSize()-1; i++)
    {
        EXPECT_EQ(0.0, c[i]);
    }
    EXPECT_EQ(1.0, c[4]);

    c.SetVal(0.0);
    MatVec(A, b, c);
    EXPECT_EQ(1.0, c[0]);
    for (int i = 1; i < c.GetSize()-1; i++)
    {
        EXPECT_EQ(0.0, c[i]);
    }
    EXPECT_EQ(1.0, c[4]);

}

TYPED_TEST(MatrixTest, MatrixOperationsDevice)
{
    Matrix<TypeParam> A;
    A.ReadFile("../tests/matrices/matrix.mtx");
    A.MoveToDevice();

    Vector<TypeParam> b(A.GetNRows(), 1.0);
    Vector<TypeParam> c(A.GetNRows());
    b.MoveToDevice();
    c.MoveToDevice();

    c = A*b;
    EXPECT_EQ(1.0, c[0]);
    for (int i = 1; i < c.GetSize()-1; i++)
    {
        EXPECT_EQ(0.0, c[i]);
    }
    EXPECT_EQ(1.0, c[4]);

    c.SetVal(0.0);
    MatVec(A, b, c);
    EXPECT_EQ(1.0, c[0]);
    for (int i = 1; i < c.GetSize()-1; i++)
    {
        EXPECT_EQ(0.0, c[i]);
    }
    EXPECT_EQ(1.0, c[4]);

}


int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
