#include "Matrix.hpp"
#include "host/HostCOOMatrix.hpp"
#include "host/HostCsrMatrix.hpp"

#include <cassert>

#include <string>

namespace pssolver
{
template <typename ValueType>
Matrix<ValueType>::Matrix()
{
    // Empty CSR matrix on host
    pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                        (new HostCsrMatrix<ValueType>());
    pImpl = pImplHost;
}

template <typename ValueType>
Matrix<ValueType>::Matrix(int nRows, int nCols, int nnz)
{
    // Empty CSR matrix on host
    pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                        (new HostCsrMatrix<ValueType>());
    pImplHost->Allocate(nRows, nCols, nnz);
    pImpl = pImplHost;
}


template <typename ValueType>
Matrix<ValueType>::~Matrix()
{
    // No-op
}

template <typename ValueType>
MatrixType Matrix<ValueType>::GetFormat(void) const
{
    return pImpl->GetFormat(); 
}

template <typename ValueType>
void Matrix<ValueType>::ReadFile(const std::string filename)
{
    if (GetFormat() == COO )
    {
        pImplHost.reset();
        pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                        (new HostCOOMatrix<ValueType>());
        
        pImplHost->ReadFile(filename);
        pImpl = pImplHost;
    }
    
}


template <typename ValueType>
int Matrix<ValueType>::GetNRows(void) const
{
    return pImpl->GetNRows();
}

template <typename ValueType>
int Matrix<ValueType>::GetNCols(void) const
{
    return pImpl->GetNCols();
}

template <typename ValueType>
int Matrix<ValueType>::GetNnz(void) const
{
    return pImpl->GetNnz();
}

template <typename ValueType>
void Matrix<ValueType>::AllocateCSR(int nRows, int nCols, int nnz)
{
    assert(nRows > 0 && nCols > 0 && nnz > 0);
    if (pImpl == pImplHost )
    {
        pImplHost.reset();
        pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                            (new HostCsrMatrix<ValueType>());
        pImplHost->Allocate(nRows, nCols, nnz);
        pImpl = pImplHost;
    }
    
}

template <typename ValueType>
void Matrix<ValueType>::AllocateCOO(int nRows, int nCols, int nnz)
{
    assert(nRows > 0 && nCols > 0 && nnz > 0 );
    if (pImpl == pImplHost )
    {
        pImplHost.reset();
        pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                            (new HostCOOMatrix<ValueType>());
        pImplHost->Allocate(nRows, nCols, nnz);
        pImpl = pImplHost;
    }

}

template <typename ValueType>
void Matrix<ValueType>::ConvertTo(MatrixType format)
{
    assert( format == COO || format == CSR );
    if (format == this->GetFormat() ) return;
    
}



template class Matrix<double>;
template class Matrix<float>;
}
