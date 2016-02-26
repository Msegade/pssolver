#include "LinearSystem.hpp"
#include "../pssolver.hpp"
#include "../utils.hpp"

namespace pssolver
{
// Constructor - Matrix and vector remain unchanged
template <class MatrixType, class VectorType>
LinearSystem<MatrixType, VectorType>::LinearSystem (MatrixType& A, VectorType& b) 
{

    DEBUGLOG(this, "LinearSystem::LinearSystem()", 
                                        "A = " << &A << " b = " << &b, 1);
    // Check size compatibility
    assert(A.GetNRows() == b.GetSize());
    mSize = A.GetNRows();
    mpA = &A;
    mpb = &b;
}


template <typename MatrixType, typename VectorType>
VectorType LinearSystem<MatrixType, VectorType>::SolveCG(int maxiter, double tol)
{
    DEBUGLOG(this, "LinearSystem::SolveCG()", 
                                "maxiter = " << maxiter << " tol = " << tol, 1);
    assert(mpA->IsHost() && mpb->IsHost() ||
            mpA->IsDevice() && mpb->IsDevice());
    
    // References to make the code easier
    MatrixType& rA = *mpA;
    VectorType& rb = *mpb;

    // Auxiliary vectors
    // Residue Vector and old residue
    VectorType res(mSize);
    VectorType resold(mSize);
    // Conjugated direction vector
    VectorType dir(mSize);
    // Initial guess
    VectorType x(mSize, 1.0);
    // move to device if needed
    if (mpA->IsDevice() )
    {
        res.MoveToDevice();
        resold.MoveToDevice();
        dir.MoveToDevice();
        x.MoveToDevice();
    }


    //res = rb - rA*x;
    res = rA*x;
    res = rb - res;
    dir = res;

    double alpha, beta;
    
    for (int i=1; i<maxiter; i++)
    {
        alpha = res*res / (dir*(rA*dir));
        x = x + (dir*alpha);
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
