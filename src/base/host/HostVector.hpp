#pragma once

#include "../BaseVector.hpp"
#include "../device/DeviceVector.hpp"


namespace pssolver
{

// Forward declaration for friend statement
template <typename ValueType>
class Vector;

template <typename ValueType>
class HostVector: public BaseVector<ValueType>
{
    // Make HostVector use the dependent name mSize of the Base class
    // in his nondependent 
    using BaseVector<ValueType>::mSize;
    // For the [] operator
    friend class Vector<ValueType>;
    // Copying from host to device
    friend class DeviceVector<ValueType>;

public:
    HostVector();
    virtual ~HostVector();

    virtual void Allocate(const int size);
    virtual void SetVal(const ValueType val);
    virtual void CopyFromHost(const BaseVector<ValueType> &hostVector);
    virtual void CopyFromDevice(const BaseVector<ValueType> &deviceVector);
    virtual void CopyToHost(BaseVector<ValueType> &hostVector) const;
    virtual void CopyToDevice(BaseVector<ValueType> &deviceVector) const;
    virtual void Add(const BaseVector<ValueType> &otherVector);
    virtual void Add(const BaseVector<ValueType> &v1,
                    const BaseVector<ValueType> &v2);
    virtual double Norm(void) const;
    virtual ValueType Dot(const BaseVector<ValueType> &otherVector);

private:

    ValueType *mData;
    
};

}
