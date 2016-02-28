#include "pssolver.hpp"
#include "../timer.hpp"

#include <iostream>

using namespace pssolver;


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage matrixVector [matrixfile]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    Matrix<double> A;
    std::string filename = std::string(argv[1]);

    high_resolution_timer timer;
    A.ReadFile(filename);
    double time =  timer.elapsed();

    std::cout << "Time =  " << time << std::endl;


    

}
