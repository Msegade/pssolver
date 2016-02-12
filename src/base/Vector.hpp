#pragma once

#include "BaseVector.hpp"
#include "host/HostVector.hpp"

#include <memory>

namespace pssolver
{

// Integers Floats and doubles
template <typename ValueType>
class Vector
{

public:
    Vector();
    Vector(int size);
    Vector(int size, const ValueType val);
    virtual ~Vector();
    
    void Allocate(int size);
    int GetSize(void) const;
    void info(void) const;
    
    virtual void SetVal(const ValueType val);

    //Access operator - Modify
    ValueType& operator[](const int i);

    //Assignment
    Vector<ValueType>& operator=(const Vector<ValueType>& otherVector);
    // a = a + b; --> Allocates a temporary vector
    // a += b; --> Doesn't allocate a temorary vector
    void operator+=(const Vector<ValueType>& otherVector);
    Vector operator+(const Vector<ValueType>& otherVector);

    double Norm(void) const;
     


private:
    std::shared_ptr<BaseVector<ValueType>> pImpl;
    std::shared_ptr<HostVector<ValueType>> pImplHost;

};

}
