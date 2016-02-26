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
    
    std::string vectorfile = std::string(argv[1]);
    std::string matrixfile = std::string(argv[2]);

    Vector<double> b;
    b.ReadFile(vectorfile);
    //std::cout << b;

    std::cout << "Matrix read" << std::endl;
    Matrix<double> A;
    A.ReadFile(matrixfile);
    //std::cout << A;

    b.MoveToDevice();
    A.MoveToDevice();

    std::cout << "Conjugated Gradients" << std::endl;
    LinearSystem<Matrix<double>, Vector<double>> LS(A,b);
    std::cout << LS.SolveCG(1000, 1e-12);
}
