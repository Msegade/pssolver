#include "HostCsrMatrix.hpp"
#include "HostCOOMatrix.hpp"

#include <cassert>
#include <cstring>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
HostCsrMatrix<ValueType>::HostCsrMatrix()
{
    mData = new ValueType[0];
    mColInd = new int[0];
    mRowPtr = new int[0];
}

template <typename ValueType>
HostCsrMatrix<ValueType>::~HostCsrMatrix()
{
    delete[] mData;
    delete[] mRowPtr;
    delete[] mColInd;

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::Allocate(const int nRows,
                                        const int nCols,const int nnz)
{

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
        os << mData[i] << "\t" << mColInd[i] << "\t" << 
                                                    mRowPtr[i] << std::endl;

        }
        else 
        os << mData[i] << "\t" << mColInd[i] << "\t" << std::endl;

    }
}

template <typename ValueType>
void HostCsrMatrix<ValueType>::CopyFrom(BaseMatrix<ValueType> &mat)
{
    BaseMatrix<ValueType>::CopyFrom(mat);
    this->Allocate(this->mNRows, this->mNCols, this->mNnz);
    if(mat.GetFormat() == COO)
    {
        HostCOOMatrix<ValueType> *cast_mat = 
            dynamic_cast<HostCOOMatrix<ValueType>*> (&mat);
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
