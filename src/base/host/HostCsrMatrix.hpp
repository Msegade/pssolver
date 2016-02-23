#pragma once

#include "../BaseMatrix.hpp"
#include "HostMatrix.hpp"

#include "HostVector.hpp"


#include <string>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
class HostCsrMatrix: public HostMatrix<ValueType>
{
public:
    HostCsrMatrix();
    virtual ~HostCsrMatrix();
    virtual MatrixType GetFormat(void) const { return CSR; }

    virtual void Allocate(const int nRows,const int nCols,const int nnz);
    virtual void ReadFile(const std::string filename) {}
    virtual void Print(std::ostream& os);

    virtual void CopyFromHost(const BaseMatrix<ValueType> &hostMatrix);
    virtual void CopyFromDevice(const BaseMatrix<ValueType> &deviceMatrix) {}
    virtual void CopyToHost(BaseMatrix<ValueType> &hostMatrix) const {}
    virtual void CopyToDevice(BaseMatrix<ValueType> &deviceMatrix) const {}

    virtual ValueType Read(int i, int j) const;

    virtual void MatVec(BaseVector<ValueType>& invec, 
                                    BaseVector<ValueType>& outvec,
                                    ValueType scalar) const;
private:
    int *mRowPtr;
    int *mColInd;
    ValueType *mData;

    friend class HostCOOMatrix<ValueType>;
    friend class DeviceCsrMatrix<ValueType>;
};

}
