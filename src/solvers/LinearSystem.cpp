#include "LinearSystem.hpp"
#include "../pssolver.hpp"
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

    
    std::ofstream ofs;
    if (x.IsHost())
        ofs.open("HostTimes.txt");
    else
        ofs.open("DeviceTimes.txt");
    high_resolution_timer timer;
    for (int i=1; i<maxiter; i++)
    {
        // No preconditioner
        timer.restart();
            beta = z*res;
        ofs << "Dotproduct " << timer.elapsed() << std::endl;
            beta = beta/sscalar;
        timer.restart();
            ScalarAdd(res, beta, s, s);
        ofs << "ScalarAdd " << timer.elapsed() << std::endl;
            ValueType val = -1.0;
        timer.restart();
            MatVec(rA, s, val, z);
        ofs << "MatVec " << timer.elapsed() << std::endl;
        timer.restart();
            sscalar = -(z*s);
        ofs << "DotProductII " << timer.elapsed() << std::endl;
        timer.restart();
            alpha = res*s;
        ofs << "DotProductIII " << timer.elapsed() << std::endl;
            alpha = alpha/sscalar;
        timer.restart();
            ScalarAdd(x, alpha, s, x);
        ofs << "ScalarAddII " << timer.elapsed() << std::endl;
        timer.restart();
            resold = res;
        ofs << "Assigment " << timer.elapsed() << std::endl;
        timer.restart();
            ScalarAdd(resold, alpha, z, res);
        ofs << "ScalarAddIII " << timer.elapsed() << std::endl;
        timer.restart();
            if ( ( res.Norm() /  resold.Norm() ) < tol)
                break;
        ofs << "Norms " << timer.elapsed() << std::endl;
    }

    return x;
    
    
}

template class LinearSystem<Matrix<double>, Vector<double>, double>;
template class LinearSystem<Matrix<float>, Vector<float>, float>;

}
