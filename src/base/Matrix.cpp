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
    DEBUGLOG(this, "Matrix::Allocate()", "nRows = " << nRows << " nCols = " << nCols
                            << " nnz = " << nnz  << " format = " << format, 1);
    assert(nRows > 0 && nCols > 0 && nnz > 0);
    if (pImpl == pImplHost )
    {
        pImplHost.reset();
        pImpl.reset();
        if(format == CSR)
            pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                            (new HostCsrMatrix<ValueType>());
        if(format == COO)
            pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                            (new HostCOOMatrix<ValueType>());
        pImplHost->Allocate(nRows, nCols, nnz);
        pImpl = pImplHost;
    }
    else if (pImpl == pImplDevice)
    {
        pImplDevice.reset();
        pImpl.reset();
        if(format == CSR)
            pImplDevice = std::shared_ptr<DeviceMatrix<ValueType>>
                            (new DeviceCsrMatrix<ValueType>());
        pImplDevice->Allocate(nRows, nCols, nnz);
        pImpl = pImplDevice;
        
    }
    
}


template <typename ValueType>
void Matrix<ValueType>::ConvertTo(MatrixType format)
{
    DEBUGLOG(this, "Matrix::ConvertTO()", "format = " << format, 1);
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
void Matrix<ValueType>::MoveToHost(void)
{
    DEBUGLOG(this, "Matrix::MoveToHost()", "Empty", 1);
    assert(pImpl == pImplDevice);
    if (this->GetFormat() == CSR)
    {
        pImplHost = std::shared_ptr<HostCsrMatrix<ValueType>>
                        (new HostCsrMatrix<ValueType>());
        pImplDevice->CopyToHost(*pImplHost);
        pImpl = pImplHost;
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

// Friend Function
template <typename ValueType>
void MatVec(const Matrix<ValueType>& mat, const Vector<ValueType>& invec, 
                                          Vector<ValueType>& outvec)
{
    DEBUGLOG(&mat, "MatVec()", "invec = " << &invec <<
                                    " outvec = "<< &outvec, 1);
    assert(invec.GetSize() == mat.GetNCols()); 
    if(invec.GetSize() != outvec.GetSize())
    {
        outvec.Allocate(invec.GetSize());
    }

    assert( (mat.IsHost() && invec.IsHost() ) ||
            (mat.IsDevice() && invec.IsDevice() ) );
    assert( (mat.IsHost() && outvec.IsHost() ) ||
            (mat.IsDevice() && outvec.IsDevice() ) );
    if ( mat.IsHost() )
    {
        if ( outvec.IsHost() ) {}
        else outvec.MoveToHost();
    }
    else if ( mat.IsDevice() )
    {
        if ( outvec.IsDevice() ) {}
        else outvec.MoveToDevice();
    }

    mat.pImpl->MatVec(*(invec.pImpl), *(outvec.pImpl), 1.0);

}

template <typename ValueType>
void MatVec(const Matrix<ValueType>& mat, const Vector<ValueType>& invec, 
                                const ValueType& val, Vector<ValueType>& outvec)
{
    DEBUGLOG(&mat, "MatVec()", "invec = " << &invec <<
                                    " outvec = "<< &outvec, 1);
    assert(invec.GetSize() == mat.GetNCols()); 
    if(invec.GetSize() != outvec.GetSize())
    {
        outvec.Allocate(invec.GetSize());
    }

    assert( (mat.IsHost() && invec.IsHost() ) ||
            (mat.IsDevice() && invec.IsDevice() ) );
    if ( mat.IsHost() )
    {
        if ( outvec.IsHost() ) {}
        else outvec.MoveToHost();
    }
    else if ( mat.IsDevice() )
    {
        if ( outvec.IsDevice() ) {}
        else outvec.MoveToDevice();
    }

    mat.pImpl->MatVec(*(invec.pImpl), *(outvec.pImpl), val);

}

template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Matrix<ValueType> &Mat)
{
    if(Mat.IsDevice())
    {
        Mat.pImpl->CopyToHost(*(Mat.pImplHost));
        Mat.pImplHost->Print(os);
        return os;
    }
    else
    {
        Mat.pImpl->Print(os);
        return os;
    }
}



template class Matrix<double>;
template class Matrix<float>;

template void MatVec(const Matrix<double>& mat, const Vector<double>& invec, Vector<double>& outvec);
template void MatVec(const Matrix<float>& mat, const Vector<float>& invec, Vector<float>& outvec);
template void MatVec(const Matrix<double>& mat, const Vector<double>& invec,
                                    const double& val, Vector<double>& outvec);
template void MatVec(const Matrix<float>& mat, const Vector<float>& invec,
                                    const float& val, Vector<float>& outvec);
template std::ostream& operator<<(std::ostream& os, const Matrix<double> &Mat);
template std::ostream& operator<<(std::ostream& os, const Matrix<float> &Mat);

}
