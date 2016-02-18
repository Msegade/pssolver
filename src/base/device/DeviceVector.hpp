#pragma once

#include "../BaseVector.hpp"

namespace pssolver
{
     
// Forward declaration for friend statement
template <typename ValueType>
class Vector;

template <typename ValueType>
class DeviceVector: public BaseVector<ValueType>
{
    using BaseVector<ValueType>::mSize;
    // For the [] operator
    friend class Vector<ValueType>;

public:
    DeviceVector();
    virtual ~DeviceVector();

    virtual void Allocate(const int size);
    virtual void SetVal(const ValueType val);
    virtual void CopyFrom(const BaseVector<ValueType> &hostVector);
    virtual void Add(const BaseVector<ValueType> &otherVector) {}
    virtual void Add(const BaseVector<ValueType>& v1,
                const BaseVector<ValueType>& v2) {}

    virtual double Norm(void) const { double result=0.0; return result;}
    virtual ValueType Dot(const BaseVector<ValueType>& otherVector)
    {
        ValueType result=0.0;
        return result;
    }
    
private:
    ValueType *d_mData;
};

}
