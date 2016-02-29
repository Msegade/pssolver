#include "pssolver.hpp"
#include <iostream>

using namespace pssolver;

int main(int argc, char* argv[] )
{
    if (argc != 3)
    {
        std::cerr << "Usage vectorio [vectorfile] [matrixfile]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    std::string vectorfile = std::string(argv[2]);
    std::string matrixfile = std::string(argv[1]);

    Vector<double> b;
    b.ReadFile(vectorfile);
    //std::cout << b;

    Matrix<double> A;
    A.ReadFile(matrixfile);
    //std::cout << A;

    //b.MoveToDevice();
    //A.MoveToDevice();

    LinearSystem<Matrix<double>, Vector<double>, double> LS(A,b);
    std::cout << LS.SolveCG(1000, 1e-12);
}
