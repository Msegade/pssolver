#pragma once

#include "../BaseMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
class DeviceMatrix: public BaseMatrix<ValueType>
{
public:
    DeviceMatrix();
    virtual ~DeviceMatrix();

};

}
