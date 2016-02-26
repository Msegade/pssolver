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

    if( A.GetFormat() == COO )
    {
        std::cout << "COO Format" << std::endl;
    }

    Vector<double> b(5,1);

    Vector<double> result;

    //std::cout << result;

    b.MoveToDevice();
    A.MoveToDevice();

    result.MoveToDevice();
    std::cout << "***************************************" <<std::endl;
    result = A*b;
    std::cout << "***************************************" <<std::endl;
    std::cout << A;
    std::cout << b;
    std::cout << result;

    LinearSystem<Matrix<double>, Vector<double>> LS(A,b);
    std::cout << "**************************************" << std::endl;
    std::cout << "Empieza gradientes conjugados" << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << LS.SolveCG(40, 0.00001);

    return 0;
}
