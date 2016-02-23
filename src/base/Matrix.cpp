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
        pImplHost->ReadFile(filename);
        pImpl = pImplHost;
    }
    if (GetFormat() == CSR)
    {
        std::shared_ptr<HostMatrix<ValueType>> tmpPtr
                        (new HostCOOMatrix<ValueType>());
        
        tmpPtr->ReadFile(filename);
        pImplHost->CopyFrom(*(tmpPtr));
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
void Matrix<ValueType>::Allocate(int nRows, int nCols, int nnz, MatrixType format)
{
    assert(nRows > 0 && nCols > 0 && nnz > 0);
    if (pImpl == pImplHost )
    {
        pImplHost.reset();
        if(format == CSR)
            pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                            (new HostCsrMatrix<ValueType>());
        if(format == COO)
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
    std::shared_ptr<HostMatrix<ValueType>> tmpPointer;
    if (format == CSR)
    {
        tmpPointer = std::shared_ptr<HostMatrix<ValueType>>
                                (new HostCsrMatrix<ValueType>());
    }
    tmpPointer->CopyFrom(*(pImplHost));
    std::swap(pImplHost, tmpPointer);
    tmpPointer.reset();
    pImpl = pImplHost;

    
}

template <typename ValueType>
ValueType Matrix<ValueType>::operator()(int i, int j)
{
    return pImpl->Read(i,j);
}

template <typename ValueType>
Matrix<ValueType>& Matrix<ValueType>::operator=(const Matrix<ValueType>& otherMatrix)
{
    assert(&otherMatrix != NULL);
    if (this == &otherMatrix)
        return *this;
    if ((pImpl == pImplHost) && (otherMatrix.pImpl == otherMatrix.pImplHost))
    {
        int NRows = otherMatrix.pImplHost->GetNRows();
        int NCols = otherMatrix.pImplHost->GetNCols();
        int Nnz = otherMatrix.pImplHost->GetNnz();
        MatrixType format = otherMatrix.pImplHost->GetFormat();
        this->Allocate(NRows, NCols, Nnz, format);
        pImpl->CopyFrom(*(otherMatrix.pImpl));
        return *this;
        
    }
    else
    {
        std::cerr << "Objects must be on the same place (device or host)"
                                                                 << std::endl;
            return *this;
    }
}


// Friend Functions
template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Matrix<ValueType> &Mat)
{
    Mat.pImpl->Print(os);
    return os;
}



template class Matrix<double>;
template class Matrix<float>;

template std::ostream& operator<<(std::ostream& os, const Matrix<double> &Mat);
template std::ostream& operator<<(std::ostream& os, const Matrix<float> &Mat);

}
