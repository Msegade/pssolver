#include "HostVector.hpp"

#include "../../utils.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>
#include <regex>

namespace pssolver
{

template <typename ValueType>
HostVector<ValueType>::HostVector() 
{
    mData = new ValueType[0];

}

template <typename ValueType>
HostVector<ValueType>::~HostVector()
{
    DEBUGLOG(this, "HostVector::~HostVector()", "Empty", 2);
    delete[] mData;
    //std::cout << "HostVector Destructor" << std::endl;
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::Allocate(const int size)
{
    DEBUGLOG(this, "HostVector::Allocate()", "size = " << size, 2);
    assert(size > 0);
    this->mSize = size;
    this->mData = new ValueType[size];      
    // Set to 0
    memset(mData, 0, size*sizeof(ValueType));
    DEBUGEND();
    
}

template <typename ValueType>
void HostVector<ValueType>::ReadFile(const std::string filename)
{
    DEBUGLOG(this, "HostVector::ReadFile()", "filename = " << filename, 2);
    
    std::ifstream mFile(filename);
    std::string line;
    std::getline(mFile, line);
    if ( !std::regex_match (line, std::regex("^[0-9]+")))
    {
        std::cerr << "Bad syntax line 1" << std::endl;
    }
    int size = std::stoi(line);
    this->Allocate(size);

    std::getline(mFile, line);
    int index = 0;
    if (!regex_match (line, std::regex("^(-?[0-9.]+e(?:\\+|\\-)[0-9]+)")))
    {
        std::cerr << "Bad syntax in line: " << index+2 << std::endl;
    }
    GoToLine(mFile, 2);
    // If the file its bigger than the size specified above -> no errors
    // if its smaller -> exception
    for (int i = 0; i < mSize; i++)
    {
        std::getline(mFile, line);
        mData[index] = std::stod(line);
        index++;
    }
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::SetVal(const ValueType val)
{
    DEBUGLOG(this, "HostVector::SetVal()", "val = " << val, 2);
    for (int i=0; i<mSize; i++)
        mData[i] = val;
    DEBUGEND();
    
}

template <typename ValueType>
ValueType HostVector<ValueType>::Read(const int i) const
{
    return mData[i];
}

template <typename ValueType>
void HostVector<ValueType>::Print(std::ostream& os)
{
    //os << "Data" << std::endl;
    for (int i = 0; i < this->mSize; i++)
    {
        os << mData[i] << std::endl;
    }
}

template <typename ValueType>
void HostVector<ValueType>::CopyFromHost(
                        const BaseVector<ValueType> &otherVector)
{

    DEBUGLOG(this, "HostVector::CopyFromHost()", "Vec = " << &otherVector, 2);
    this->Allocate(otherVector.GetSize());
    // To access private attributes of the derived class we need to 
    // downcast the object
    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_vec->mData[i];
    }
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::CopyFromDevice(
                        const BaseVector<ValueType> &deviceVector)
{
    DEBUGLOG(this, "HostVector::CopyFromDevice()", "Vec = " << &deviceVector, 2);
    const DeviceVector<ValueType> *cast_vec = 
        dynamic_cast<const DeviceVector<ValueType>*> (&deviceVector);

    cast_vec->CopyToHost(*this);
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::CopyToHost(
                        BaseVector<ValueType> &otherVector) const
{

    DEBUGLOG(this, "HostVector::CopyToHost()", "Vec = " << &otherVector, 2);
    otherVector.Allocate(this->GetSize());
    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    for (int i=0; i<this->mSize; i++)
    {
        cast_vec->mData[i] = mData[i];
    }
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::CopyToDevice(
                        BaseVector<ValueType> &deviceVector) const
{

    DEBUGLOG(this, "HostVector::CopyToDevice()", "Vec = " << &deviceVector, 2);
    DeviceVector<ValueType> *cast_vec = 
        dynamic_cast<DeviceVector<ValueType>*> (&deviceVector);

    cast_vec->CopyFromHost(*this);
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::Add(
                        const BaseVector<ValueType> &otherVector)
{
    DEBUGLOG(this, "HostVector::Add()", "Vec = " << &otherVector, 2);
    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);
    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = mData[i] + cast_vec->mData[i];
    }
    DEBUGEND();

}


template <typename ValueType>
void HostVector<ValueType>::Add(
                        const BaseVector<ValueType> &v1,
                        const BaseVector<ValueType> &v2)
{
    DEBUGLOG(this, "HostVector::Add()", "Vec1 = " << &v1
            << " Vec2 = " << &v2, 2);
    const HostVector<ValueType> *cast_v1 = 
        dynamic_cast<const HostVector<ValueType>*> (&v1);
    const HostVector<ValueType> *cast_v2 = 
        dynamic_cast<const HostVector<ValueType>*> (&v2);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_v1->mData[i] + cast_v2->mData[i];
    }
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::ScalarAdd(
                        const BaseVector<ValueType> &v1,
                        const BaseVector<ValueType> &v2, const ValueType& val)
{
    DEBUGLOG(this, "HostVector::ScalarAdd()", "Vec1 = " << &v1
            << " Vec2 = " << &v2 << " val = " << val, 2);
    const HostVector<ValueType> *cast_v1 = 
        dynamic_cast<const HostVector<ValueType>*> (&v1);
    const HostVector<ValueType> *cast_v2 = 
        dynamic_cast<const HostVector<ValueType>*> (&v2);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_v1->mData[i] + val*cast_v2->mData[i];
    }
    DEBUGEND();

}

template <typename ValueType>
void HostVector<ValueType>::Substract(
                        const BaseVector<ValueType> &otherVector)
{
    DEBUGLOG(this, "HostVector::Substract()", "Vec = " << &otherVector, 2);
    const HostVector<ValueType> *cast_vec = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);
    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = mData[i] - cast_vec->mData[i];
    }
    DEBUGEND();

}


template <typename ValueType>
void HostVector<ValueType>::Substract(
                        const BaseVector<ValueType> &v1,
                        const BaseVector<ValueType> &v2)
{
    DEBUGLOG(this, "HostVector::Substract()", "Vec1 = " << &v1
            << " Vec2 = " << &v2, 2);
    const HostVector<ValueType> *cast_v1 = 
        dynamic_cast<const HostVector<ValueType>*> (&v1);
    const HostVector<ValueType> *cast_v2 = 
        dynamic_cast<const HostVector<ValueType>*> (&v2);

    for (int i=0; i<this->mSize; i++)
    {
        mData[i] = cast_v1->mData[i] - cast_v2->mData[i];
    }
    DEBUGEND();

}

template <typename ValueType>
double HostVector<ValueType>::Norm(void) const
{
    DEBUGLOG(this, "HostVector::Norm()", "Empty" , 2);
    double result = 0.0;
    for (int i=0; i<mSize; ++i)
    {
       result += mData[i]*mData[i];
    }

    DEBUGEND();
    return std::sqrt((double)result);

}
template <typename ValueType>
double HostVector<ValueType>::Norm(BaseVector<ValueType>& aux) const
{
    // Host Vector doesn't need an aux vector
    this->Norm();
}

template <typename ValueType>
ValueType HostVector<ValueType>::Dot(const BaseVector<ValueType>& otherVector)
{
    DEBUGLOG(this, "HostVector::Dot()", "Vec = " << &otherVector, 2);
    const HostVector<ValueType> *cast_v = 
        dynamic_cast<const HostVector<ValueType>*> (&otherVector);

    ValueType result = 0.0;
    for (int i=0; i<this->mSize; i++)
    {
        result += mData[i]*cast_v->mData[i];
    }

    DEBUGEND();
    return result;

}
template <typename ValueType>
void HostVector<ValueType>::ScalarMul(const ValueType& val)
{
    DEBUGLOG(this, "HostVector::ScalarMul()", "val = " << val, 2);
    for (int i=0; i<this->mSize; i++)    
    {
        mData[i] = mData[i]*val;
    }
    DEBUGEND();
}

template <typename ValueType>
void HostVector<ValueType>::ScalarMul(const ValueType& val, BaseVector<ValueType>& outvec)
{
    DEBUGLOG(this, "HostVector::ScalarMul()", "val = " << val << 
                " outvec = " << &outvec, 2);
    const HostVector<ValueType> *cast_v = 
        dynamic_cast<const HostVector<ValueType>*> (&outvec);
    for (int i=0; i<this->mSize; i++)    
    {
        cast_v->mData[i] = mData[i]*val;
    }
    DEBUGEND();
}

template <typename ValueType>
ValueType HostVector<ValueType>::SumReduce(void)
{
    DEBUGLOG(this, "HostVector::SumReduce()", "Empty", 1);
    ValueType result = 0.0;
    for (int i = 0; i<this->mSize; i++)
    {
        result = result + mData[i];
    }

    DEBUGEND();
    return result;
}

template class HostVector<double>;
template class HostVector<float>;
template class HostVector<int>;

}
