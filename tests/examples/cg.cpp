#include "pssolver.hpp"
#include "../timer.hpp"
#include <iostream>
#include <unistd.h>

using namespace pssolver;

int main(int argc, char* argv[] )
{
    if (argc != 3)
    {
        std::cerr << "Usage gc [matrixfile] [vectorfile]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    std::string matrixfile = std::string(argv[1]);
    std::string vectorfile = std::string(argv[2]);

    Vector<double> b;
    b.ReadFile(vectorfile);

    Matrix<double> A;
    A.ReadFile(matrixfile);

    Vector<double> result;

    LinearSystem<Matrix<double>, Vector<double>, double> LS(A,b);

    high_resolution_timer timer;
    result = LS.SolveCG(1000, 1e-12);
    double hosttime = timer.elapsed();
    std::cout << "Host Time = " << hosttime << std::endl;

    A.MoveToDevice();
    b.MoveToDevice();
    LS.MoveToDevice();
    result.MoveToDevice();

    timer.restart();
    result = LS.SolveCG(1000, 1e-12);
    double devicetime = timer.elapsed();
    std::cout << "Device Time = " << devicetime;






}
