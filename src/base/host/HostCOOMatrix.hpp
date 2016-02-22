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

    virtual void ReadFile(const std::string filename);
    virtual void Print(std::ostream& os);
private:
    int *mRowInd;
    int *mColInd;
    ValueType *mData;
};

}
