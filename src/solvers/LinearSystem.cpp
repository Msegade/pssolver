#include "LinearSystem.hpp"
#include "../pssolver.hpp"

namespace pssolver
{
// Constructor - Matrix and vector remain unchanged
template <class MatrixType, class VectorType>
LinearSystem<MatrixType, VectorType>::LinearSystem (MatrixType& A, VectorType& b) 
{
    // Check size compatibility
    assert(A.GetNRows() == b.GetSize());
    mSize = A.GetNRows();
    mpA = &A;
    mpb = &b;
}


template <typename MatrixType, typename VectorType>
VectorType LinearSystem<MatrixType, VectorType>::SolveCG(int maxiter, double tol)
{
    // References to make the code easier
    MatrixType& rA = *mpA;
    VectorType& rb = *mpb;

    // Residue Vector and old residue
    VectorType res(mSize);
    VectorType resold(mSize);
    // Conjugated direction vector
    VectorType dir(mSize);
    // Initial guess
    VectorType x(mSize, 1.0);

    res = rb - rA*x;
    dir = res;

    double alpha, beta;
    
    for (int i=1; i<maxiter; i++)
    {
        alpha = res*res / (dir*(rA*dir));
        x = x + dir*alpha;
        resold = res;
        res = res - (rA*dir)*alpha;
        beta = res*res / (resold*resold);
        dir = res + dir*beta;
        if ( ( res.Norm() /  resold.Norm() ) < tol)
            break;
    }

    return x;
    
    
}

template class LinearSystem<Matrix<double>, Vector<double>>;
template class LinearSystem<Matrix<float>, Vector<float>>;

}
