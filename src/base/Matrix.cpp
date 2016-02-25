#include "Matrix.hpp"
#include "host/HostCOOMatrix.hpp"
#include "host/HostCsrMatrix.hpp"
#include "device/DeviceCsrMatrix.hpp"

#include "../utils.hpp"

#include <cassert>

#include <string>

namespace pssolver
{
template <typename ValueType>
Matrix<ValueType>::Matrix()
{
    DEBUGLOG(this, "Matrix::Matrix()", "Empty", 1);
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
bool Matrix<ValueType>::IsHost(void) const
{
    return (pImpl == pImplHost);
}

template <typename ValueType>
bool Matrix<ValueType>::IsDevice(void) const
{
    return (pImpl == pImplDevice);
}

template <typename ValueType>
void Matrix<ValueType>::ReadFile(const std::string filename)
{
    DEBUGLOG(this, "Matrix::ReadFile()", "filename = " << filename, 1);
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
        pImplHost->CopyFromHost(*(tmpPtr));
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
    tmpPointer->CopyFromHost(*(pImplHost));
    std::swap(pImplHost, tmpPointer);
    tmpPointer.reset();
    pImpl = pImplHost;

    
}

template <typename ValueType>
void Matrix<ValueType>::MoveToDevice(void)
{
    DEBUGLOG(this, "Matrix::MoveToDevice()", "Empty", 1);
    assert(pImpl == pImplHost);
    if (this->GetFormat() == CSR)
    {
        pImplDevice = std::shared_ptr<DeviceCsrMatrix<ValueType>>
                        (new DeviceCsrMatrix<ValueType>());
        pImplHost->CopyToDevice(*pImplDevice);
        pImpl = pImplDevice;
    }

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
        pImpl->CopyFromHost(*(otherMatrix.pImpl));
        return *this;
        
    }
    else
    {
        std::cerr << "Objects must be on the same place (device or host)"
                                                                 << std::endl;
            return *this;
    }
}

template <typename ValueType>
Vector<ValueType> Matrix<ValueType>::operator*(const Vector<ValueType>& vec)
{
    DEBUGLOG(this, "Matrix::operator*", "vec = " << &vec, 1);
    assert(vec.GetSize() == this->GetNCols()); 
    assert( (this->IsHost() && vec.IsHost() ) ||
            (this->IsDevice() && vec.IsDevice() ) );

    Vector<ValueType> out(this->GetNRows());

    if ( this->IsDevice() ) out.MoveToDevice();
    this->pImpl->MatVec(*(vec.pImpl), *(out.pImpl), 1.0);

    return out;
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
