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

}
