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

    Vector<double> b;
    filename = std::string(argv[2]);
    b.ReadFile(filename);

    Vector<double> result(b.GetSize());

    high_resolution_timer timer;
    A.MoveToDevice();
    b.MoveToDevice();
    result.MoveToDevice();
    double time =  timer.elapsed();
    std::cout << "Allocation Time =  " << time << std::endl;

    timer.restart();
    MatVec(A, b,result);
    time =  timer.elapsed();
    std::cout << "Matrix Vector Time =  " << time << std::endl;

    //Asigment operator
    timer.restart();
    result = b;
    time =  timer.elapsed();
    std::cout << "AssigmentTime =  " << time << std::endl;

    //Dot product
    timer.restart();
    double alpha =  result * result;
    time =  timer.elapsed();
    std::cout << "Dot Product =  " << time << std::endl;

    //Vector * escalar
    timer.restart();
    ScalarMul(b, alpha, result);
    time =  timer.elapsed();
    std::cout << "Vector * Scalar = " << time << std::endl;

    std::cout << "******************************************" << std::endl;
    Vector<double> x(b.GetSize());
    x.MoveToDevice();
    std::cout << "******************************************" << std::endl;
    std::cout << "Alpha = " << alpha << std::endl;

    std::cout << "Scalar Add" << std::endl;
    timer.restart();
    ScalarAdd(b, alpha, result, x);
    time =  timer.elapsed();
    std::cout << "Vector scalar add = " << time << std::endl;



    

}
