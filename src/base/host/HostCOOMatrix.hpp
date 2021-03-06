#pragma once

#include "../BaseMatrix.hpp"
#include "HostMatrix.hpp"


#include <string>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
class HostCOOMatrix: public HostMatrix<ValueType>
{
public:
    HostCOOMatrix();
    virtual ~HostCOOMatrix();
    virtual MatrixType GetFormat(void) const { return COO; }

    virtual void Allocate(const int nRows,const int nCols,const int nnz);

    virtual void CopyFromHost(const BaseMatrix<ValueType> &hostMatrix) {}
    virtual void CopyFromDevice(const BaseMatrix<ValueType> &deviceMatrix) {}
    virtual void CopyToHost(BaseMatrix<ValueType> &hostMatrix) const {}
    virtual void CopyToDevice(BaseMatrix<ValueType> &deviceMatrix) const {}

    virtual void ReadFile(const std::string filename);
    virtual void Print(std::ostream& os);

    virtual ValueType Read(int i, int j) const{ return 0.0;}

    virtual void MatVec(BaseVector<ValueType>& invec,
                            BaseVector<ValueType>& outvec,
                            ValueType scalar) const {}
private:
    int *mRowInd;
    int *mColInd;
    ValueType *mData;

    friend class HostCsrMatrix<ValueType>;
};

}
