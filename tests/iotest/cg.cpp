#include "pssolver.hpp"
#include "../timer.hpp"
#include <iostream>
#include <unistd.h>

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

    LinearSystem<Matrix<double>, Vector<double>> LS(A,b);

    high_resolution_timer timer;
    LS.SolveCG(1000, 1e-12);
    double hosttime = timer.elapsed();

    std::cout << "Host Time = " << hosttime << std::endl;

    A.MoveToDevice();
    b.MoveToDevice();

    timer.restart();
    LS.SolveCG(1000, 1e-12);
    double devicetime = timer.elapsed();
    std::cout << "Device Time = " << devicetime;




}
