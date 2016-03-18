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
        b.ReadFile("../tests/matrices/step3vector.txt");
        //b.ReadFile("../tests/matrices/evenvector.txt");
        //b.ReadFile("../tests/matrices/step6vector.txt");

    }
    catch (std::exception  f)
    {
        std::cerr << "Error opening/reading/closing" << std::endl;
        return 1;
    }

    Vector<double> aux(b);
    high_resolution_timer timer;
    timer.restart();
    std::cout << aux.SumReduce() << std::endl;
    std::cout << "Host Time = " << timer.elapsed() << std::endl;

    b.MoveToDevice();
    timer.restart();
    std::cout << b.SumReduce() << std::endl;
    std::cout << b << std::endl;
    std::cout << "Device Time = " << timer.elapsed() << std::endl;


}
