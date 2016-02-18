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
ValueType HostVector<ValueType>::Read(const int i) const
{
    return mData[i];
}

template <typename ValueType>
void HostVector<ValueType>::CopyFromHost(
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
void HostVector<ValueType>::CopyFromDevice(
                        const BaseVector<ValueType> &deviceVector)
{
    // To access private attributes of the derived class we need to 
    // downcast the object

    const DeviceVector<ValueType> *cast_vec = 
        dynamic_cast<const DeviceVector<ValueType>*> (&deviceVector);

    cast_vec->CopyToHost(*this);

}

template <typename ValueType>
void HostVector<ValueType>::CopyToHost(
                        BaseVector<ValueType> &otherVector) const
{

    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    for (int i=0; i<this->mSize; i++)
    {
        cast_vec->mData[i] = mData[i];
    }

}

template <typename ValueType>
void HostVector<ValueType>::CopyToDevice(
                        BaseVector<ValueType> &deviceVector) const
{

    DeviceVector<ValueType> *cast_vec = 
        dynamic_cast<DeviceVector<ValueType>*> (&deviceVector);

    cast_vec->CopyFromHost(*this);

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
    double result = 0.0;
    for (int i=0; i<mSize; ++i)
    {
       result += mData[i]*mData[i];
    }

    return std::sqrt((double)result);

}

template <typename ValueType>
ValueType HostVector<ValueType>::Dot(const BaseVector<ValueType>& otherVector)
{
    const HostVector<ValueType> *cast_v = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    ValueType result = 0.0;
    for (int i=0; i<this->mSize; i++)
    {
        result += mData[i]*cast_v->mData[i];
    }

    return result;

}

template class HostVector<double>;
template class HostVector<float>;
template class HostVector<int>;

}
