#pragma once

#include <fstream>
#include <string>

#ifndef NDEBUG

#include "../tests/timer.hpp"

extern int mlevel;

#define  DEBUGLOG( obj, fct, args, level )                    \
    high_resolution_timer debugtimer;    \
    mlevel++;                               \
    std::cout << std::string(mlevel-1, '\t')               \
              <<  "Obj Addr: " << obj                    \
              << "; fct: " << fct                         \
              << " " args << std::endl;                    \

#define  DEBUGEND()                    \
    mlevel--;                                   \
    std::cout << debugtimer.elapsed() << std::endl; \


#else
#define DEBUGLOG( obj, fct, args, level );
#define DEBUGEND();
#endif

namespace pssolver
{

    std::ifstream& GoToLine(std::ifstream& file, unsigned int num);

}

