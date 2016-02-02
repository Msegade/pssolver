#include "BaseVector.hpp"
#include <iostream>
namespace pssolver
{
template <typename ValueType>
BaseVector<ValueType>::BaseVector() : mSize(0) {}

template <typename ValueType>
BaseVector<ValueType>::~BaseVector()
{
    //std::cout << "BaseVector Destructor" << std::endl;
} 

template <typename ValueType>
inline int BaseVector<ValueType>::GetSize() const
{
    return mSize;
}

template class BaseVector<double>;
template class BaseVector<float>;
template class BaseVector<int>;

}
