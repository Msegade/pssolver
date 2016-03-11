#pragma once

#include "../BaseVector.hpp"
#include "../device/DeviceVector.hpp"


namespace pssolver
{

// Forward declaration for friend statement
template <typename ValueType>
class Vector;
template <typename ValueType>
class HostCsrMatrix;

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
    // Matrix vector multiplication
    friend class HostCsrMatrix<ValueType>;

public:
    HostVector();
    virtual ~HostVector();

    void ReadFile(const std::string filename);
    virtual void Allocate(const int size);
    virtual ValueType Read(const int i) const;
    virtual void Print(std::ostream& os);
    virtual void SetVal(const ValueType val);
    virtual void CopyFromHost(const BaseVector<ValueType> &hostVector);
    virtual void CopyFromDevice(const BaseVector<ValueType> &deviceVector);
    virtual void CopyToHost(BaseVector<ValueType> &hostVector) const;
    virtual void CopyToDevice(BaseVector<ValueType> &deviceVector) const;
    virtual void Add(const BaseVector<ValueType> &otherVector);
    virtual void Add(const BaseVector<ValueType> &v1,
                    const BaseVector<ValueType> &v2);
    virtual void ScalarAdd(const BaseVector<ValueType>& v1,
                    const BaseVector<ValueType>& v2, const ValueType& val);
    virtual void Substract(const BaseVector<ValueType> &otherVector);
    virtual void Substract(const BaseVector<ValueType> &v1,
                    const BaseVector<ValueType> &v2);
    virtual double Norm(void) const;
    virtual double Norm(BaseVector<ValueType>& aux) const;
    virtual ValueType Dot(const BaseVector<ValueType> &otherVector);
    virtual void ScalarMul(const ValueType& val);
    virtual void ScalarMul(const ValueType& val, BaseVector<ValueType>& outvec);

private:

    ValueType *mData;
    
};

}
