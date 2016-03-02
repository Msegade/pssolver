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

    std::cout << "NRows = " << C.GetNRows() << std::endl;
    std::cout << "NCols = " << C.GetNCols() << std::endl;
    std::cout << "Nnz = " << C.GetNnz() << std::endl;
    std::cout << C << std::endl;

}
