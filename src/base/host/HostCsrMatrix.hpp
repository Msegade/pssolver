#pragma once

#include "../BaseMatrix.hpp"
#include "HostMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
class HostCsrMatrix: public HostMatrix<ValueType>
{
public:
    HostCsrMatrix();
    virtual ~HostCsrMatrix();
private:
    int *mRowPtr;
    int *mColInd;
    ValueType *mData;
};

}
