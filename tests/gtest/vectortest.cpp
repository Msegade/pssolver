#include <gtest/gtest.h>
#include "pssolver.hpp"

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

}

TYPED_TEST(VectorTest, VectorDataManipulation)
{
    Vector<TypeParam> b;
    b.Allocate(100);
    EXPECT_EQ(100, b.GetSize());
    
    b.SetVal(7);
    for (int i = 0; i<100; i++)
    {
        EXPECT_EQ(7, b[i]);
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

}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
