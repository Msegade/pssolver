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

    C.Allocate(100, 100, 121, CSR);
    std::cout << C.GetNRows() << std::endl;


}
