#pragma once

#include "../pssolver.hpp"
#include <cassert>

namespace pssolver
{

template <class MatrixType, class VectorType>
class LinearSystem
{
private:
    int mSize;
    // This have to be pointers because they don't have a constructor with 0 arguments
    MatrixType* mpA;
    VectorType* mpb;

    // Don't allow the use of copy constructor -> private
    //LinearSystem(const LinearSystem& otherLinearSystem){};

public:
    LinearSystem(MatrixType &A, VectorType &b);
    // No objects allocated with new
//    ~LinearSystem();

    VectorType SolveCG(int maxiter, double tol);

};
}
