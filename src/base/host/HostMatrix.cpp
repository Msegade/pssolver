#include "HostMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
HostMatrix<ValueType>::HostMatrix() {}
template <typename ValueType>
HostMatrix<ValueType>::~HostMatrix() {}

template class HostMatrix<double>;
template class HostMatrix<float>;
    
}
