#include "pssolver.hpp"
#include <iostream>
#include <fstream>
#include "../timer.hpp"

using namespace pssolver;

int main(void)
{
    Vector<double> b;
    try 
    {
        b.ReadFile("../tests/matrices/evenvector.txt");

    }
    catch (std::exception  f)
    {
        std::cerr << "Error opening/reading/closing" << std::endl;
        return 1;
    }

    Vector<double> aux(b);
    high_resolution_timer timer;
    timer.restart();
    std::cout << aux.Norm() << std::endl;
    std::cout << "Host Time = " << timer.elapsed() << std::endl;

    b.MoveToDevice();
    timer.restart();
    std::cout << b.Norm() << std::endl;
    std::cout << "Device Time = " << timer.elapsed() << std::endl;


}
