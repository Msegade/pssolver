#include "pssolver.hpp"
#include <iostream>
#include <fstream>

using namespace pssolver;

int main(void)
{
    Vector<double> b(5, 12);
    Vector<double> c(5, 7);
    Vector<double> d(31);

    ScalarAdd(b, 2.0, c, d);

    std::cout << d;

}
