#include "DeviceMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
DeviceMatrix<ValueType>::DeviceMatrix() {}
template <typename ValueType>
DeviceMatrix<ValueType>::~DeviceMatrix() {}

template class DeviceMatrix<double>;
template class DeviceMatrix<float>;
    
}
