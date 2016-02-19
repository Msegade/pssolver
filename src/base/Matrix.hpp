#pragma once

#include "BaseMatrix.hpp"
#include "host/HostMatrix.hpp"
#include "host/HostCsrMatrix.hpp"

#include <memory>

namespace pssolver
{

//Floats and doubles
template <typename ValueType>
class Matrix
{

public:
    Matrix();
    Matrix(int nRows, int nCols);
    virtual ~Matrix();

private:
    std::shared_ptr<BaseMatrix<ValueType>> pImpl;
    std::shared_ptr<HostMatrix<ValueType>> pImplHost;

};

}
