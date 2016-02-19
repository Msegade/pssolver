#pragma once

#include "../BaseMatrix.hpp"
#include "HostMatrix.hpp"

#include <string>

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
private:
    int *mRowPtr;
    int *mColInd;
    ValueType *mData;
};

}
