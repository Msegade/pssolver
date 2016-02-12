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

}

TYPED_TEST(VectorTest, VectorOperations)
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

}


int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
