#include "pssolver.hpp"
#include <iostream>
#include <fstream>

using namespace pssolver;

int main(void)
{
    Matrix<double> C;
    try 
    {
        C.ReadFile("../tests/matrices/matrix.mtx");
    }
    catch (std::exception  f)
    {
        std::cerr << "Error opening/reading/closing" << std::endl;
        return 1;
    }

    C.MoveToDevice();

    std::cout << C(4,4) << std::endl;


}
