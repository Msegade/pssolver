#include "pssolver.hpp"
#include <iostream>
#include <fstream>
#include "../timer.hpp"

using namespace pssolver;

int main(void)
{
    Matrix<double> A;
    Vector<double> b;
    Vector<double> result;
    try 
    {
        A.ReadFile("../tests/dealii/matrix.mtx");
        b.ReadFile("../tests/dealii/vector.txt");

    }
    catch (std::exception  f)
    {
        std::cerr << "Error opening/reading/closing" << std::endl;
        return 1;
    }

    high_resolution_timer timer;
    timer.restart();
    MatVec(A, b, result);
//    std::cout << "Host Time = " << timer.elapsed() << std::endl;

 //   std::cout << "Device time *****************" << std::endl;
    A.MoveToDevice();
    b.MoveToDevice();
    result.MoveToDevice();
    timer.restart();
    MatVec(A, b, result);
  //  std::cout << "Device Time = " << timer.elapsed() << std::endl;



}
