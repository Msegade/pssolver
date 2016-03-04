#pragma once

#include "base/Vector.hpp"
#include "base/host/HostVector.hpp"
#include "base/device/DeviceVector.hpp"

#include "base/Matrix.hpp"
#include "base/BaseMatrix.hpp"
#include "base/host/HostMatrix.hpp"
#include "base/host/HostCsrMatrix.hpp"
#include "base/host/HostCOOMatrix.hpp"

#include "solvers/LinearSystem.hpp"


#ifndef NDEBUG

#include <fstream>
extern std::ofstream logfile;
std::ofstream logfile("log.txt");

extern int mlevel;
int mlevel=0;

extern int linedebug;
int linedebug=0;

#endif
