#include "HostVector.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>

namespace pssolver
{

template <typename ValueType>
HostVector<ValueType>::HostVector() 
{
    mData = new ValueType[0];

}

template <typename ValueType>
HostVector<ValueType>::~HostVector()
{
    delete[] mData;
    //std::cout << "HostVector Destructor" << std::endl;

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

template <typename ValueType>
void HostVector<ValueType>::Add(
                        const BaseVector<ValueType> &otherVector)
{
    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);
    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = mData[i] + cast_vec->mData[i];
    }

}

template <typename ValueType>
void HostVector<ValueType>::Add(
                        const BaseVector<ValueType> &v1,
                        const BaseVector<ValueType> &v2)
{
    const HostVector<ValueType> *cast_v1 = 
        dynamic_cast<const HostVector<ValueType>*> (&v1);
    const HostVector<ValueType> *cast_v2 = 
        dynamic_cast<const HostVector<ValueType>*> (&v2);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_v1->mData[i] + cast_v2->mData[i];
    }

}

template <typename ValueType>
double HostVector<ValueType>::Norm(void) const
{
    double norm = 0.0;
    for (int i=0; i<mSize; ++i)
    {
       norm += mData[i]*mData[i];
    }

    return std::sqrt((double)norm);

}

template class HostVector<double>;
template class HostVector<float>;
template class HostVector<int>;

}
