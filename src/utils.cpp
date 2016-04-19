#include "utils.hpp"

#include <fstream>  
#include <limits>

namespace pssolver
{

std::ifstream& GoToLine(std::ifstream& file, unsigned int num)
{
    file.seekg(std::ios::beg);
    for(unsigned int i=0; i < num - 1; ++i)
    {
            file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

}
