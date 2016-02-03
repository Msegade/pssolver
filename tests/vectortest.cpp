#include <catch.hpp>

#include "pssolver.hpp"
#include <unistd.h>
#include <iostream>


using namespace pssolver;
using namespace std;

#define TEST_TYPE(NAME, DESCRIPTION, T)           \
TEST_CASE( NAME,DESCRIPTION)                      \
{                                                 \
    Vector<T> v(5);                               \
    REQUIRE( v.GetSize() == 5);                   \
                                                  \
    v.Allocate(10);                               \
    REQUIRE( v.GetSize() == 10);                  \
                                                  \
    v.SetVal(7);                                  \
    REQUIRE( v[3] == 7);                          \
                                                  \
    Vector<T> v2(10, 3);                          \
    REQUIRE( v2[3] == 3);                         \
    REQUIRE( v2.GetSize() == 10);                 \
    v2 = v;                                       \
                                                  \
    REQUIRE( v2[3] == 7);                         \
                                                  \
    v2 += v;                                      \
    REQUIRE( v2[3] == 14);                        \
                                                  \
}                                               

TEST_TYPE("Vector of Integers","Vector<int>", int);
TEST_TYPE("Vector of floats","Vector<float>", float);
TEST_TYPE("Vector of doubles","Vector<double>", double);


//int main()
//{
    // Vector<float> b(); Doesn't work because fo vexing parse issue;
    //Vector<float> b(100,3.0);
    //Vector<float> c(100,5.1);
    //c.Allocate(200);
    //b+=c;
    //cout << b[3] << endl;



    //Vector<float> c;
    //Vector<float> d;

    //d = c = b;

    //std::cout << b.GetSize() << std::endl;

    //b.Allocate(1000);

    //std::cout << "==== GetSize =====" << std::endl;
    //std::cout << b.GetSize() << std::endl;

    //std::cout << "==== info =====" << std::endl;
    //std::cout << "**********************"<< std::endl;

    //b.info();

    //std::cout << "**********************"<< std::endl;

    //b.SetVal(5.0);

    //std::cout << "==== Operator [] =====" << std::endl;
    //std::cout << b[3] << std::endl;

    //b[3] = 7.3;
    //std::cout << "==== Operator[] as lvalue =====" << std::endl;
    //std::cout << b[3] << std::endl;

    //c = b;
    //std::cout << "==== Assingment=====" << std::endl;
    //std::cout << c[3] << std::endl;


    


//    return 0;
//}
