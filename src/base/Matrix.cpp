#include "Matrix.hpp"

namespace pssolver
{
template <typename ValueType>
Matrix<ValueType>::Matrix()
{
    // Empty CSR matrix on host
    pImplHost = std::shared_ptr<HostMatrix<ValueType>>
                        (new HostCsrMatrix<ValueType>());
    pImpl = pImplHost;
}

template <typename ValueType>
Matrix<ValueType>::~Matrix()
{
    // No-op
}


template class Matrix<double>;
template class Matrix<float>;
}
