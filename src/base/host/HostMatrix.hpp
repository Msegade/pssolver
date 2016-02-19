#pragma once

#include "../BaseMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
class HostMatrix: public BaseMatrix<ValueType>
{
public:
    HostMatrix();
    virtual ~HostMatrix();
};

}
