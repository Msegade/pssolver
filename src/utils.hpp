#pragma once

#include <fstream>
#include <string>

#ifndef NDEBUG

#include "../tests/timer.hpp"

extern int mlevel;
extern std::ofstream logfile;
extern int linedebug;

#define  DEBUGLOG( obj, fct, args, level )                    \
    mlevel++;                                                \
    int locallinedebug;                                     \
    linedebug++;                                           \
    locallinedebug = linedebug;                           \
    logfile  << std::string(mlevel-1, '\t')              \
              <<  "Obj Addr: " << obj                   \
              << "; fct: " << fct                      \
              << " " << args << std::endl;            \
    high_resolution_timer debugtimer;                \

#define  DEBUGEND()                                  \
    double elapsedtime = debugtimer.elapsed();      \
    logfile << '\t' << locallinedebug              \
            << '\t' << elapsedtime << std::endl;  \
    mlevel--;                                    \


#else
#define DEBUGLOG( obj, fct, args, level );
#define DEBUGEND();
#endif

namespace pssolver
{

    std::ifstream& GoToLine(std::ifstream& file, unsigned int num);

}

