#pragma once

#include "BaseVector.hpp"
#include "host/HostVector.hpp"
#include "device/DeviceVector.hpp"

#include <memory>
#include <iostream>

namespace pssolver
{

template <typename ValueType>
class Matrix;

// Integers Floats and doubles
template <typename ValueType>
class Vector
{

public:
    Vector();
    Vector(int size);
    Vector(int size, const ValueType val);
    Vector(const Vector<ValueType>& otherVector);
    virtual ~Vector();
    
    void ReadFile(const std::string filename);
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

    // Decrement and minus
    void operator-=(const Vector<ValueType>& otherVector);
    // a = a + b; --> Allocates a temporary vector
    Vector<ValueType> operator-(const Vector<ValueType>& otherVector);

    double Norm(void) const;
    // Dot Product
    ValueType operator*(const Vector<ValueType>& otherVector) const;
    // Scalar product
    Vector<ValueType> operator*(const ValueType& scalar) const;

    // Scalar multiplication 
    // a *= val --> Doesn't allocate a temporary vector
    void operator*=(const ValueType& val);
    // a = a * val --> Allocates a temporary vector

    template <typename T>
    friend void ScalarMul(const Vector<T>& invec, const T& val, Vector<T>& outvec);
    template <typename T>
    friend void MatVec(const Matrix<T>& mat, const Vector<T>& invec, Vector<T>& outvec);
    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const Vector<T>& vec);


private:
    std::shared_ptr<BaseVector<ValueType>> pImpl;
    std::shared_ptr<HostVector<ValueType>> pImplHost;
    std::shared_ptr<DeviceVector<ValueType>> pImplDevice;

    friend class Matrix<ValueType>;

};

template <typename ValueType>
void ScalarMul(const Vector<ValueType>& invec, const ValueType& val, Vector<ValueType>& outvec);
template <typename ValueType>
void MatVec(const Matrix<ValueType>& mat, const Vector<ValueType>& invec, Vector<ValueType>& outvec);
template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Vector<ValueType>& vec);

}
