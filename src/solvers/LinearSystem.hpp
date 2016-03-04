#pragma once

#include <cassert>

namespace pssolver
{


template <class MatrixType, class VectorType, typename ValueType>
class LinearSystem
{
private:
    int mSize;
    // This have to be pointers because they don't have a constructor with 0 arguments
    MatrixType* mpA; 
    VectorType* mpb;

    // Aux vectors
    VectorType res;
    VectorType resold;
    VectorType s;
    VectorType x;
    VectorType z;

    bool mIsHost;

public:
    LinearSystem(MatrixType &A, VectorType &b);
    // No objects allocated with new
    void  MoveToDevice(void);

    VectorType SolveCG(int maxiter, double tol);

};
}
