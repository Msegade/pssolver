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

    virtual void CopyFrom(const BaseVector<ValueType> &otherVector) = 0;

protected:
    int mSize;

};

}
