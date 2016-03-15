#include <gtest/gtest.h>
#include "pssolver.hpp"
#include <iostream>
#include <unistd.h>

using namespace pssolver;

// VectorTest fixture
// Necessary for templates
template <class T>
class VectorTest : public ::testing::Test
{
protected:
    VectorTest() : Vector_(new Vector<T>(100, 3)) {}
    virtual ~VectorTest() { delete Vector_; }
    Vector<T>* const Vector_;
};

typedef ::testing::Types<double, float, int> MyTypes;

TYPED_TEST_CASE(VectorTest, MyTypes);

TYPED_TEST(VectorTest, VectorConstructors)
{
    Vector<TypeParam> a;
    EXPECT_EQ(0, a.GetSize());

    Vector<TypeParam> b(5);
    EXPECT_EQ(5, b.GetSize());

    Vector<TypeParam> c(10, 3);
    EXPECT_EQ(10, c.GetSize());
    for (int i = 0; i<10; i++)
    {
        EXPECT_EQ(3, c[i]);
    }

    Vector<TypeParam> d(c);
    EXPECT_EQ(10, d.GetSize());
    for (int i = 0; i<10; i++)
    {
        EXPECT_EQ(3, d[i]);
    }


}

TYPED_TEST(VectorTest, VectorDataManipulation)
{

    Vector<TypeParam> a;
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    std::cout << "*********************" << std::endl;
    std::cout << cwd << std::endl;
    a.ReadFile("../tests/gtest/vector.txt");
    TypeParam value = 1.0;
    for (int i = 0; i<a.GetSize(); i++)
    {
        EXPECT_NEAR(value, a[i], 0.00001);
        value = value + 1.1;
    }
    EXPECT_TRUE(a.IsHost());


    Vector<TypeParam> b;
    b.Allocate(100);
    EXPECT_EQ(100, b.GetSize());
    
    b.SetVal(7);
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(7, b[i]);
    }

    Vector<TypeParam> c;
    c = b;
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(7, c[i]);
    }

    // Mov to the device
    b.MoveToDevice();
    EXPECT_TRUE(b.IsDevice());

    b.Allocate(5);
    EXPECT_EQ(5, b.GetSize());

    b.SetVal(9);
    for (int i = 0; i<5; i++)
    {
        EXPECT_EQ(9, b[i]);
    }
    c.MoveToDevice();
    // assigment
    c = b;
    EXPECT_TRUE(b.IsDevice());
    for (int i = 0; i<5; i++)
    {
        EXPECT_EQ(9, c[i]);
    }

    

}

TYPED_TEST(VectorTest, VectorOperationsHost)
{
    // *(this->Vector_) is the one created in the fixture
    Vector<TypeParam> b;
    //Assigment
    b = *(this->Vector_);
    EXPECT_EQ(100, b.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(3, b[i]);
    }
    // Increment
    b += *(this->Vector_);
    EXPECT_EQ(100, b.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(6, b[i]);
    }
    // Sum 
    Vector<TypeParam> c;
    c = b + *(this->Vector_);
    EXPECT_EQ(100, c.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(9, c[i]);
    }
    // Substraction
    c = c - b;
    EXPECT_EQ(100, c.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(3, c[i]);
    }

    // Norm
    c.Allocate(5);
    c.SetVal(7);
    EXPECT_NEAR(15.652476, c.Norm(), 0.0001);

    // Dot Product
    b.Allocate(5);
    b.SetVal(4);
    EXPECT_EQ(140, b*c);

    // ScalarMul
    b *= 3;
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(12, b[i]);
    }

    Vector<TypeParam> d;
    // Scalar Add
    TypeParam val = 2.0;
    ScalarAdd(b, val, c, d);
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(26, d[i]);
    }

    // Scalar Mul
    ScalarMul(b, val, d); 
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(24, d[i]);
    }


}

TYPED_TEST(VectorTest, VectorOperationsDevice)
{
    // *(this->Vector_) is the one created in the fixture
    Vector<TypeParam> b(10,5);
    Vector<TypeParam> c(100,7);
    b.MoveToDevice();
    c.MoveToDevice();
    //Assigment
    b = c;
    EXPECT_EQ(100, b.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(7, b[i]);
    }
    // Increment
    b += c;
    EXPECT_EQ(100, b.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(14, b[i]);
    }
    // Sum
    Vector<TypeParam> d;
    d.MoveToDevice();
    d = b + c;
    EXPECT_EQ(100, b.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(21, d[i]);
    }

    // Substraction
    c = c - b;
    EXPECT_EQ(100, c.GetSize());
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(-7, c[i]);
    }
    // Norm
    c.Allocate(5);
    c.SetVal(7);
    EXPECT_NEAR(15.652476, c.Norm(), 0.0001);

    // Dot Product
    b.Allocate(5);
    b.SetVal(4);
    EXPECT_EQ(140, b*c);

    // ScalarMul
    b *= 3;
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(12, b[i]);
    }

    // Scalar Add No extra allocs
    TypeParam val = 2.0;
    ScalarAdd(b, val, c, d);
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(26, d[i]);
    }

    // Scalar Mul No extra allocs
    ScalarMul(b, val, d); 
    for (int i = 0; i<4; i++)
    {
        EXPECT_EQ(24, d[i]);
    }



}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
