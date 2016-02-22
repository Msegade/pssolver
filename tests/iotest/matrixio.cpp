#include "pssolver.hpp"
#include <iostream>

using namespace pssolver;

int main(int argc, char* argv[] )
{
    if (argc != 2)
    {
        std::cerr << "Usage matrixmain [matrixfile]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    std::string filename = std::string(argv[1]);

    Matrix<double> A;
    A.ReadFile(filename);

    std::cout << A << std::endl;
}
