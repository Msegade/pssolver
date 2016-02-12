#pragma once

#include "../BaseVector.hpp"


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

public:
    HostVector();
    virtual ~HostVector();

    virtual void Allocate(const int size);
    virtual void SetVal(const ValueType val);
    virtual void CopyFrom(const BaseVector<ValueType> &otherVector);
    virtual void Add(const BaseVector<ValueType> &otherVector);
    virtual void Add(const BaseVector<ValueType> &v1,
                    const BaseVector<ValueType> &v2);

private:

    ValueType *mData;
    
};

}
