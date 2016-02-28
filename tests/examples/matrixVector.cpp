#include "pssolver.hpp"
#include "../timer.hpp"

#include <iostream>

using namespace pssolver;


int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage matrixVector [matrixfile] [vectorfile]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    Matrix<double> A;
    std::string filename = std::string(argv[1]);
    A.ReadFile(filename);

    high_resolution_timer timer;
    Vector<double> b;
    filename = std::string(argv[2]);
    b.ReadFile(filename);
    double time =  timer.elapsed();
    std::cout << "Time =  " << time << std::endl;



    

}
