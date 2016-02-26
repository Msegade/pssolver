#include "Vector.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>

#include "../utils.hpp"

namespace pssolver
{

template <typename ValueType>
Vector<ValueType>::Vector()
{
    DEBUGLOG( this, "Vector::Vector()", "Empty", 1);
    // Create empty vector on host
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImpl = pImplHost;

}

template <typename ValueType>
Vector<ValueType>::Vector(const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::Vector()", "Vector = " << &otherVector, 1);
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImplHost->CopyFromHost(*(otherVector.pImplHost));
    pImpl = pImplHost;
    if (otherVector.IsDevice()) 
    {
        pImplDevice = std::shared_ptr<DeviceVector<ValueType>>
                                (new DeviceVector<ValueType>());
        pImplDevice->CopyFromDevice(*(otherVector.pImplDevice)); 
        pImpl = pImplDevice;
    }
    
}

template <typename ValueType>
Vector<ValueType>::Vector(int size)
{
    DEBUGLOG( this, "Vector::Vector()", "size = " << size, 1);
    assert(size>0);
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImplHost->Allocate(size);
    pImpl = pImplHost;

}

template <typename ValueType>
Vector<ValueType>::Vector(int size, const ValueType val)
{
    DEBUGLOG( this, "Vector::Vector()", 
            "size = " << size << " val = " <<val, 1);
    assert(size>0);
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImplHost->Allocate(size);
    pImpl = pImplHost;
    pImpl->SetVal(val);

}

template <typename ValueType>
Vector<ValueType>::~Vector()
{
    // No-op -> pImplHost gets deleted automatically (smart pointer)
    //std::cout << "Vector Destructor" << std::endl;
}

template <typename ValueType>
void Vector<ValueType>::ReadFile(const std::string filename)
{
    DEBUGLOG(this, "Vector::ReadFile()", "filename = " << filename, 1);
    if (this->IsHost())
        pImplHost->ReadFile(filename);
    else
    {
        pImplHost->ReadFile(filename);
        pImplHost->CopyToDevice(*pImplDevice);
    }

    
}

template <typename ValueType>
void Vector<ValueType>::MoveToDevice(void)
{
    DEBUGLOG( this, "Vector::MoveToDevice()", "Empty", 1);
    pImplDevice = std::shared_ptr<DeviceVector<ValueType>>
                                (new DeviceVector<ValueType>());
    pImplDevice->CopyFromHost(*pImplHost);
    pImpl = pImplDevice;
}

template <typename ValueType>
bool Vector<ValueType>::IsHost(void) const
{
    return (pImpl == pImplHost);
}

template <typename ValueType>
bool Vector<ValueType>::IsDevice(void) const
{
    return (pImpl == pImplDevice);
}

template <typename ValueType>
void Vector<ValueType>::Allocate(int size)
{
    DEBUGLOG( this, "Vector::Allocate()", "Size = " << size, 1);
    assert(size>0);
    if ( pImpl == pImplHost )
    {
        pImplHost.reset();
        pImplHost = std::shared_ptr<HostVector<ValueType>>
                                    (new HostVector<ValueType>());
        assert(pImplHost != NULL);
        pImplHost->Allocate(size);
    }
    else if (pImpl == pImplDevice )
    {
        pImplDevice.reset();
        pImplDevice = std::shared_ptr<DeviceVector<ValueType>>
                                    (new DeviceVector<ValueType>());
        assert(pImplDevice != NULL);
        pImplDevice->Allocate(size);

    }

}

template <typename ValueType>
int Vector<ValueType>::GetSize(void) const
{
    return pImpl->GetSize();
}

template <typename ValueType>
void Vector<ValueType>::info(void) const
{
    if (pImpl == pImplHost)
    {
        std::cout << "Host Vector" << std::endl;
    }
    std::cout << "Size = " << this->GetSize() << std::endl;
    std::cout << "Precission = " << 8*sizeof(ValueType)<< std::endl;

}

template <typename ValueType>
void Vector<ValueType>::SetVal(const ValueType val)
{
    pImpl->SetVal(val);
}

template <typename ValueType>
ValueType Vector<ValueType>::operator[](const int i) const
{
    assert((i >= 0) && (i < pImpl->GetSize()));
    return pImpl->Read(i);
}

template <typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator=(
                                const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::operator=", "Vector = " << &otherVector, 1);
    assert(&otherVector != NULL);
    if (this == &otherVector)
       return *this;
    
    if((pImpl == pImplHost) && (otherVector.pImpl == otherVector.pImplHost))
    {
        int size = otherVector.GetSize();
        //this->Allocate(size);
        // we need to pass a reference (BaseVector&)
        // otherVector.pImpl is BaseVector*
        pImpl->CopyFromHost(*(otherVector.pImpl));

        // Returning a Vector object allows chaining assigment
        // a = b = c;
        // Returning by reference makes that no copy of the object is created
        // and destroyed
        return *this;
    }
    else if((pImpl == pImplDevice) && (otherVector.pImpl == otherVector.pImplDevice))
    {
        int size = otherVector.GetSize();
        //this->Allocate(size);
        pImpl->CopyFromDevice(*(otherVector.pImpl));
        return *this;
    }
    else
    {
        std::cerr << "Objects must be on the same place (device or host)"
                  << std::endl;
        return *this;
    }


}

template <typename ValueType>
void Vector<ValueType>::operator+=(
                                const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::operator+=", "Vec =" << &otherVector, 1);
    assert(GetSize() == otherVector.GetSize());
    pImpl->Add(*(otherVector.pImpl));
}

template <typename ValueType>
Vector<ValueType> Vector<ValueType>::operator+(
                                const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::operator+", "Vec =" << &otherVector, 1);
    assert(GetSize() == otherVector.GetSize());
    assert(( IsHost() && otherVector.IsHost() )|| 
            (IsDevice() && otherVector.IsDevice()) );

    Vector<ValueType> result(GetSize());

    if (pImpl == pImplHost)
        result.pImpl->Add(*(otherVector.pImpl), *pImpl);
    else if (pImpl == pImplDevice)
    {
        result.MoveToDevice();
        result.pImpl->Add(*(otherVector.pImpl), *pImpl);
    }

    return result;
        
}

template <typename ValueType>
void Vector<ValueType>::operator-=(
                                const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::operator-=", "Vec =" << &otherVector, 1);
    assert(GetSize() == otherVector.GetSize());
    pImpl->Substract(*(otherVector.pImpl));
}

template <typename ValueType>
Vector<ValueType> Vector<ValueType>::operator-(
                                const Vector<ValueType>& otherVector)
{
    DEBUGLOG( this, "Vector::operator-", "Vec =" << &otherVector, 1);
    assert(GetSize() == otherVector.GetSize());

    Vector<ValueType> result(GetSize());

    if (pImpl == pImplHost)
        result.pImpl->Substract(*pImpl, *(otherVector.pImpl));
    else if (pImpl == pImplDevice)
    {
        result.MoveToDevice();
        result.pImpl->Substract((*pImpl), *(otherVector.pImpl));
    }

    return result;
        
}

template <typename ValueType>
double Vector<ValueType>::Norm(void) const
{
    if (GetSize()>0)
    {
        return pImpl->Norm();
    }
    else
    {
        return 0.0;
    }

}

template <typename ValueType>
ValueType Vector<ValueType>::operator*(const Vector<ValueType>& otherVector) const 
{
    DEBUGLOG( this, "Vector::operator*", "Vec =" << &otherVector, 1);
    assert(GetSize() == otherVector.GetSize()); 
    return pImpl->Dot(*(otherVector.pImpl));
    
}

template <typename ValueType>
Vector<ValueType> Vector<ValueType>::operator*(const ValueType& val) const 
{
    DEBUGLOG( this, "Vector::operator*", "Scalar =" << val, 1);
    Vector<ValueType> result(*this);
    result.pImpl->ScalarMul(val);
    return result;
    
}

template <typename ValueType>
void Vector<ValueType>::operator*=(const ValueType& val)
{
    pImpl->ScalarMul(val);
}
                                
// Friend functions
template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Vector<ValueType> &Vec)
{
    if (Vec.IsHost())
        Vec.pImpl->Print(os);
    else
    {
        Vec.pImpl->CopyToHost(*(Vec.pImplHost));
        Vec.pImplHost->Print(os);
    }
    return os;
}

// Instantiate the class for the supported template type 
// parameters. If not done you have to include the 
// implementation in the header
template class Vector<double>;
template class Vector<float>;
template class Vector<int>;

template std::ostream& operator<<(std::ostream& os, const Vector<double> &Vec);
template std::ostream& operator<<(std::ostream& os, const Vector<float> &Vec);
template std::ostream& operator<<(std::ostream& os, const Vector<int> &Vec);
}
