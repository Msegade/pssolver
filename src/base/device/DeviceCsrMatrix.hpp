#pragma once

#include "../BaseMatrix.hpp"
#include "DeviceMatrix.hpp"

#include "DeviceVector.hpp"

#include <string>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
class DeviceCsrMatrix: public DeviceMatrix<ValueType>
{
public:
    DeviceCsrMatrix();
    virtual ~DeviceCsrMatrix();
    virtual MatrixType GetFormat(void) const { return CSR; }

    virtual void Allocate(const int nRows,const int nCols,const int nnz);
    virtual void ReadFile(const std::string filename) {}
    virtual void Print(std::ostream& os) {}

    virtual void CopyFrom(BaseMatrix<ValueType> &hostMatrix) {}

    virtual ValueType Read(int i, int j) const { return 0.0;}

    virtual void MatVec(BaseVector<ValueType>& invec, 
                                    BaseVector<ValueType>& outvec,
                                    ValueType scalar) const;
private:
    int *d_mRowPtr;
    int *d_mColInd;
    ValueType *d_mData;

};

}
