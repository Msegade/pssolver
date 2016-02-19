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
    DeviceVector(const DeviceVector<ValueType>& v);
    virtual ~DeviceVector();

    virtual void Allocate(const int size);
    virtual ValueType Read(const int i) const;
    virtual void SetVal(const ValueType val);
    virtual void CopyFromHost(const BaseVector<ValueType> &hostVector);
    virtual void CopyFromDevice(const BaseVector<ValueType> &deviceVector);
    virtual void CopyToHost(BaseVector<ValueType> &hostVector) const;
    virtual void CopyToDevice(BaseVector<ValueType> &deviceVector) const;
    virtual void Add(const BaseVector<ValueType> &otherVector);
    virtual void Add(const BaseVector<ValueType>& v1,
                const BaseVector<ValueType>& v2);

    virtual double Norm(void) const;
    virtual ValueType Dot(const BaseVector<ValueType>& otherVector);
    virtual void ScalarMul(const ValueType& val);
    
private:
    ValueType *d_mData;

    ValueType SumReduce(void);
};

}
