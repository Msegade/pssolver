#pragma once

namespace pssolver
{

// Base class for the implementations of host and device vectors
template <typename ValueType>
class BaseVector
{

public:
    BaseVector();
    virtual ~BaseVector();

    int GetSize(void) const;

    // Allocated in HostVector
    virtual void Allocate(const int size) = 0;

    virtual void SetVal(const ValueType val) = 0;

    virtual void CopyFromHost(const BaseVector<ValueType> &otherVector) = 0;
    virtual void CopyFromDevice(const BaseVector<ValueType> &otherVector) = 0;
    virtual void CopyToHost(BaseVector<ValueType> &otherVector) const = 0;
    virtual void CopyToDevice(BaseVector<ValueType> &otherVector) const = 0;

    virtual void Add(const BaseVector<ValueType> &otherVector) = 0;
    virtual void Add(const BaseVector<ValueType>& v1,
                const BaseVector<ValueType>& v2) = 0;

    virtual double Norm(void) const = 0;
    virtual ValueType Dot(const BaseVector<ValueType>& otherVector) = 0;

// Protected so the derived classes have acces to it
protected:
    int mSize;

};

}
