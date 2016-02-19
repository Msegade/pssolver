#include "BaseMatrix.hpp"

namespace pssolver
{
template <typename ValueType>
BaseMatrix<ValueType>::BaseMatrix() : mNRows(0), mNCols(0), mNnz(0) {}

template <typename ValueType>
BaseMatrix<ValueType>::~BaseMatrix() {}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNRows(void) const
{
    return mNRows;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNCols(void) const
{
    return mNCols;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNnz(void) const
{
    return mNnz;
}

template class BaseMatrix<double>;
template class BaseMatrix<float>;

}
