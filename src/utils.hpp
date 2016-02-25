#pragma once

#include <fstream>
#include <string>

#ifndef NDEBUG
#define  DEBUGLOG( obj, fct, args, level ) {                   \
    std::cout << std::string(level-1, '\t')                 \
               <<  "Obj Addr: " << obj                    \
             << "; fct: " << fct                         \
            << " " args << std::endl;                    \
}

#else
#define DEBUGLOG( obj, fct, args );
#endif

namespace pssolver
{

    std::ifstream& GoToLine(std::ifstream& file, unsigned int num);

}

