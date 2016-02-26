#include "HostCsrMatrix.hpp"
#include "HostCOOMatrix.hpp"

#include "../device/DeviceCsrMatrix.hpp"

#include "../../utils.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
HostCsrMatrix<ValueType>::HostCsrMatrix()
{
    DEBUGLOG(this, "HostCsrMatrix::HostCsrMatrix()", "Empty", 2);
    mData = new ValueType[0];
    mColInd = new int[0];
    mRowPtr = new int[0];
}

template <typename ValueType>
HostCsrMatrix<ValueType>::~HostCsrMatrix()
{
    DEBUGLOG(this, "HostCsrMatrix::~HostCsrMatrix()", "Empty", 2);
    delete[] mData;
    delete[] mRowPtr;
    delete[] mColInd;

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::Allocate(const int nRows,
                                        const int nCols,const int nnz)
{

    DEBUGLOG(this, "HostCsrMatrix::Allocate()", 
            "nRows = " << nRows << " nCols = " << nCols << " nnz = " << nnz, 2);
    assert ( nRows > 0 && nCols > 0 && nnz > 0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    mData = new ValueType[nnz];
    mColInd = new int[nnz];
    mRowPtr = new int[nRows+1];

    memset(mData, 0, nnz*sizeof(ValueType));
    memset(mColInd, 0, nnz*sizeof(int));
    memset(mRowPtr, 0, (nRows+1)*sizeof(int));
}

template <typename ValueType>
void HostCsrMatrix<ValueType>::Print(std::ostream& os)
{
    os << "Data" << "\t" << "Col Index" << "\t" << 
                                                    "Row Ptr" << std::endl;
    for (int i= 0; i< this->mNnz; i++)
    {
        if (i <= this->mNRows)
        {
        os << mData[i] << "\t" << mColInd[i] << "\t\t" << 
                                                    mRowPtr[i] << std::endl;

        }
        else 
        os << mData[i] << "\t" << mColInd[i] << "\t\t" << std::endl;

    }
}

template <typename ValueType>
void HostCsrMatrix<ValueType>::CopyFromHost(const BaseMatrix<ValueType> &mat)
{
    DEBUGLOG(this, "HostCsrMatrix::CopyFromHost()", "BaseMat = " << &mat, 2);
    this->Allocate(mat.GetNRows(), mat.GetNCols(), mat.GetNnz());
    if(mat.GetFormat() == COO)
    {
        const HostCOOMatrix<ValueType> *cast_mat = 
            dynamic_cast<const HostCOOMatrix<ValueType>*> (&mat);
        // ******************************************************
        // mRowPtr
        // Number of nnz per row in mRowPtr
        for (int i = 0; i < this->mNnz; i++)
        {
            this->mRowPtr[cast_mat->mRowInd[i]] = 
                        this->mRowPtr[cast_mat->mRowInd[i]] + 1;  
        }
        // Counting where the rows begin
        int count = 0;
        for (int i = 0; i < this->mNRows; i++)
        {
            int temp = this->mRowPtr[i];
            this->mRowPtr[i] = count;
            count += temp;
        }
        this->mRowPtr[this->mNRows] = this->mNnz;

        // *******************************************************
        // mColInd y mData
        for (int i = 0; i <this->mNnz; i++)
        {
            int row = cast_mat->mRowInd[i];
            int dest = this->mRowPtr[row];
            
            this->mColInd[dest] = cast_mat->mColInd[i];
            this->mData[dest] = cast_mat->mData[i];

            // Necesitamos un contador para saber cuantas veces entramos en 
            // esa fila, lo guardamos en mRowPtr y luego lo volvemos a calcular
            // para no tener que almacenar otro vector
            this->mRowPtr[row] = this->mRowPtr[row] + 1;
        }
        // Recalculamos mRowPtr
        // Tenemos desplazados todos los valores a la izquierda
        int last = 0;
        for (int i = 0; i < this->mNRows; i++)
        {
            int temp = this->mRowPtr[i];
            this->mRowPtr[i] = last;
            last = temp;
        }

    }
    else if(mat.GetFormat() == CSR)
    {
        const HostCsrMatrix<ValueType> *cast_mat = 
            dynamic_cast<const HostCsrMatrix<ValueType>*> (&mat);

        for (int i=0; i<this->mNnz; i++) 
        {
            this->mData[i] = cast_mat->mData[i];
            this->mColInd[i] = cast_mat->mColInd[i];
        }
        for (int i=0; i<(this->mNRows+1); i++) 
        {
            this->mRowPtr[i] = cast_mat->mRowPtr[i];
        }

    }

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::CopyFromDevice(const BaseMatrix<ValueType> &mat)
{
    DEBUGLOG(this, "HostCsrMatrix::CopyFromDevice()", "BaseMat = " << &mat, 2);
    const DeviceCsrMatrix<ValueType> *cast_device = 
        dynamic_cast<const DeviceCsrMatrix<ValueType>*> (&mat);
    cast_device->CopyToHost(*(this));


}

template <typename ValueType>
void HostCsrMatrix<ValueType>::CopyToHost(BaseMatrix<ValueType> &mat) const
{
    DEBUGLOG(this, "HostCsrMatrix::CopyToHost()", "BaseMat = " << &mat, 2);
    mat.Allocate(this->GetNRows(), this->GetNCols(), this->GetNnz());

    if(mat.GetFormat() == CSR)
    {
        HostCsrMatrix<ValueType> *cast_mat = 
            dynamic_cast<HostCsrMatrix<ValueType>*> (&mat);

        for (int i=0; i<this->mNnz; i++) 
        {
            cast_mat->mData[i] = this->mData[i];
            cast_mat->mColInd[i] = this->mColInd[i];
        }
        for (int i=0; i<(this->mNRows+1); i++) 
        {
            cast_mat->mRowPtr[i] = this->mRowPtr[i];
        }

    }

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::CopyToDevice(BaseMatrix<ValueType> &mat) const
{
    DEBUGLOG(this, "HostCsrMatrix::CopyToDevice()", "BaseMat = " << &mat, 2);
    DeviceCsrMatrix<ValueType> *cast_device = 
        dynamic_cast<DeviceCsrMatrix<ValueType>*> (&mat);
    cast_device->CopyFromHost(*(this));


}

template <typename ValueType>
ValueType HostCsrMatrix<ValueType>::Read(int i, int j) const
{
    assert (i >=0 && j>=0 && i < this->mNRows && j < this->mNCols);
    int aux = mRowPtr[i];
    int rowWidth = mRowPtr[i+1] - mRowPtr[i];
    int k;
    ValueType result;
    for (k = 0; k < rowWidth; k++)
    {
        if(mColInd[aux+k] == j) 
        {
            result = mData[aux+k];
            break;
        }
        else result = 0.0;
    }
    return result;

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::MatVec(BaseVector<ValueType>& invec, BaseVector<ValueType>& outvec,
                    ValueType scalar) const
{
    DEBUGLOG(this, "HostCsrMatrix::MatVec()", "InVec = " << &invec
                    << " OutVec = " << &outvec << " Scalar = " << scalar,  2);
    assert(invec.GetSize() == this->GetNCols());    
    assert(outvec.GetSize() == this->GetNRows());    

    const HostVector<ValueType> *cast_in =
                dynamic_cast<const HostVector<ValueType>*> (&invec);
    const HostVector<ValueType> *cast_out =
                dynamic_cast<const HostVector<ValueType>*> (&outvec);

    for (int i=0; i<this->mNRows; i++)
    {
        for (int j=this->mRowPtr[i]; j<this->mRowPtr[i+1]; j++)
        {
            cast_out->mData[i] += scalar*this->mData[j]*cast_in->mData[this->mColInd[j] ];
        }
    }
    


}

template class HostCsrMatrix<double>;
template class HostCsrMatrix<float>;

}
