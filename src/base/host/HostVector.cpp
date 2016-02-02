#include "HostVector.hpp"

#include <cassert>
#include <cstring>
#include <iostream>

namespace pssolver
{

template <typename ValueType>
HostVector<ValueType>::HostVector()
{

}

template <typename ValueType>
HostVector<ValueType>::~HostVector()
{

    delete[] mData;
    std::cout << "HostVector Destructor" << std::endl;


}

template <typename ValueType>
void HostVector<ValueType>::Allocate(const int size)
{
    assert(size > 0);
    mSize = size;
    mData = new ValueType[size];      
    // Set to 0
    memset(mData, 0, size*sizeof(ValueType));
}

template <typename ValueType>
void HostVector<ValueType>::SetVal(const ValueType val)
{
    for (int i=0; i<mSize; i++)
        mData[i] = val;
    
}

template <typename ValueType>
void HostVector<ValueType>::CopyFrom(
                        const BaseVector<ValueType> &otherVector)
{
    // To access private attributes of the derived class we need to 
    // downcast the object

    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_vec->mData[i];
    }

}
template class HostVector<double>;
template class HostVector<float>;
template class HostVector<int>;

}
