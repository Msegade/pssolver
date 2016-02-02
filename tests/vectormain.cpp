#include "pssolver.hpp"
#include <unistd.h>
#include <iostream>

using namespace pssolver;
using namespace std;

int main()
{

    // Vector<float> b(); Doesn't work because fo vexing parse issue;
    Vector<float> b(100,3.0);
    Vector<float> c(100,5.1);
    c.Allocate(200);
    b+=c;
    cout << b[3] << endl;



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


    


    return 0;
}
