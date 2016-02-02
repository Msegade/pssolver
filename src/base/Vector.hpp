#pragma once

#include "BaseVector.hpp"
#include "host/HostVector.hpp"

namespace pssolver
{

// Integers Floats and doubles
template <typename ValueType>
class Vector
{

public:
    Vector();
    Vector(int size);
    virtual ~Vector();
    
    void Allocate(int size);
    int GetSize(void) const;
    void info(void) const;
    
    virtual void SetVal(const ValueType val);

    //Access operator - Modify
    ValueType& operator[](const int i);

    //Assignment
    Vector<ValueType>& operator=(const Vector<ValueType>& otherVector);


private:
    BaseVector<ValueType> *pImpl;
    HostVector<ValueType> *pImplHost;

};

}
