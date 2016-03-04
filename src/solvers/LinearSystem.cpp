#include "LinearSystem.hpp"
#include "../base/Matrix.hpp"
#include "../base/Vector.hpp"
#include "../utils.hpp"
#include "../../tests/timer.hpp"

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
    
    // Auxiliary vectors
    res.Allocate(mSize);
    resold.Allocate(mSize);
    s.Allocate(mSize);
    x.Allocate(mSize);
    z.Allocate(mSize);

    mIsHost = true;

    DEBUGEND();
}

template <class MatrixType, class VectorType, typename ValueType>
void LinearSystem<MatrixType, VectorType, ValueType>::MoveToDevice()
{
    DEBUGLOG(this, "LinearSystem::MoveToDevice()", "Empty", 1);
    assert(mIsHost);

    res.MoveToDevice();
    resold.MoveToDevice();
    s.MoveToDevice();
    x.MoveToDevice();
    z.MoveToDevice();

    mIsHost = false;

    DEBUGEND();
}


template <class MatrixType, class VectorType, typename ValueType>
VectorType LinearSystem<MatrixType, VectorType, ValueType>::SolveCG(int maxiter, double tol)
{
    DEBUGLOG(this, "LinearSystem::SolveCG()", 
                                "maxiter = " << maxiter << " tol = " << tol, 1);
    assert(mpA->IsHost() && mpb->IsHost() && mIsHost ||
            mpA->IsDevice() && mpb->IsDevice() && (!mIsHost));
    
    // References to make the code easier
    MatrixType& rA = *mpA;
    VectorType& rb = *mpb;

    ValueType sscalar;
    ValueType alpha, beta;

    x.SetVal(1.0);

    //res = rb - rA*x;
    ValueType val = -1.0;
    MatVec(rA, x, val, res);
    res += rb;
    // Sin precondicionamiento
    s = res;
    MatVec(rA, s, val, z);
    sscalar = -(z*s);
    alpha = res*s;
    alpha = alpha/sscalar;

    ScalarAdd(x, alpha, s, x);
    resold = res;
    ScalarAdd(resold, alpha, z, res);

    for (int i=1; i<maxiter; i++)
    {
        // No preconditioner
            beta = z*res;
            beta = beta/sscalar;
            ScalarAdd(res, beta, s, s);
            ValueType val = -1.0;
            MatVec(rA, s, val, z);
            sscalar = -(z*s);
            alpha = res*s;
            alpha = alpha/sscalar;
            ScalarAdd(x, alpha, s, x);
            resold = res;
            ScalarAdd(resold, alpha, z, res);
            if ( ( res.Norm() /  resold.Norm() ) < tol)
                break;
    }

    return x;
    
    
    DEBUGEND();
}

template class LinearSystem<Matrix<double>, Vector<double>, double>;
template class LinearSystem<Matrix<float>, Vector<float>, float>;

}
