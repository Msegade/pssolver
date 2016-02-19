#include "HostCsrMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
HostCsrMatrix<ValueType>::HostCsrMatrix() {}

template <typename ValueType>
HostCsrMatrix<ValueType>::~HostCsrMatrix()
{
    delete[] mData;
    delete[] mRowPtr;
    delete[] mColInd;

}

template class HostCsrMatrix<double>;
template class HostCsrMatrix<float>;

}
