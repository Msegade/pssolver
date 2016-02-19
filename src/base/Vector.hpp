#pragma once

#include "BaseVector.hpp"
#include "host/HostVector.hpp"
#include "device/DeviceVector.hpp"

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
    
    void MoveToDevice(void);
    bool IsHost(void) const;
    bool IsDevice(void) const;

    void Allocate(int size);
    int GetSize(void) const;
    void info(void) const;
    
    virtual void SetVal(const ValueType val);

    //Access operator - Modify
    ValueType operator[](const int i) const;

    //Assignment
    Vector<ValueType>& operator=(const Vector<ValueType>& otherVector);
    // a += b; --> Doesn't allocate a temorary vector
    void operator+=(const Vector<ValueType>& otherVector);
    // a = a + b; --> Allocates a temporary vector
    Vector<ValueType> operator+(const Vector<ValueType>& otherVector);

    double Norm(void) const;
    // Dot Product
    ValueType operator*(const Vector<ValueType>& otherVector) const;

    // Scalar multiplication 
    // a *= val --> Doesn't allocate a temporary vector
    void operator*=(const ValueType& val);
    // a = a * val --> Allocates a temporary vector



private:
    std::shared_ptr<BaseVector<ValueType>> pImpl;
    std::shared_ptr<HostVector<ValueType>> pImplHost;
    std::shared_ptr<DeviceVector<ValueType>> pImplDevice;

};

}
