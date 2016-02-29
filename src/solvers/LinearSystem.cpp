#include "LinearSystem.hpp"
#include "../pssolver.hpp"
#include "../utils.hpp"

namespace pssolver
{
// Constructor - Matrix and vector remain unchanged
template <class MatrixType, class VectorType, typename ValueType>
LinearSystem<MatrixType, VectorType, ValueType>::LinearSystem (MatrixType& A, VectorType& b) 
{

    DEBUGLOG(this, "LinearSystem::LinearSystem()", 
                                        "A = " << &A << " b = " << &b, 1);
    // Check size compatibility
    assert(A.GetNRows() == b.GetSize());
    mSize = A.GetNRows();
    mpA = &A;
    mpb = &b;
}


template <class MatrixType, class VectorType, typename ValueType>
VectorType LinearSystem<MatrixType, VectorType, ValueType>::SolveCG(int maxiter, double tol)
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
    VectorType s(mSize);
    // Initial guess
    VectorType x(mSize, 1.0);
    // Aux vector
    VectorType z(mSize, 1.0);
    // move to device if needed
    if (mpA->IsDevice() )
    {
        res.MoveToDevice();
        resold.MoveToDevice();
        s.MoveToDevice();
        x.MoveToDevice();
        z.MoveToDevice();
    }

    ValueType sscalar;
    ValueType alpha, beta;

    //res = rb - rA*x;
    MatVec(rA, x, res);
    res -= rb;
    // Sin precondicionamiento
    s = res;
    MatVec(rA, s, z);
    sscalar = z*s;
    alpha = res*s;
    alpha = 1/alpha;

    ScalarAdd(x, alpha, s, x);
    resold = res;
    ScalarAdd(resold, alpha, z, res);

    
    for (int i=1; i<maxiter; i++)
    {
        // No preconditioner
        beta = z*res;
        beta = 1/beta;
        ScalarAdd(res, beta, s, s);
        MatVec(rA, s, z);
        sscalar = z*s;
        alpha = res*s;
        alpha = 1/alpha;
        ScalarAdd(x, alpha, s, x);
        resold = res;
        ScalarAdd(resold, alpha, z, res);
        if ( ( res.Norm() /  resold.Norm() ) < tol)
            break;
    }

    return x;
    
    
}

template class LinearSystem<Matrix<double>, Vector<double>, double>;
template class LinearSystem<Matrix<float>, Vector<float>, float>;

}
