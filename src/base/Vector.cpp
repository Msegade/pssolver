#include "Vector.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>

namespace pssolver
{

template <typename ValueType>
Vector<ValueType>::Vector()
{
    // Create empty vector on host
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImpl = pImplHost;

}

template <typename ValueType>
Vector<ValueType>::Vector(int size)
{
    assert(size>0);
    pImplHost = std::shared_ptr<HostVector<ValueType>>
                                (new HostVector<ValueType>());
    pImplHost->Allocate(size);
    pImpl = pImplHost;

}

template <typename ValueType>
Vector<ValueType>::Vector(int size, const ValueType val)
{
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
void Vector<ValueType>::MoveToDevice(void)
{
    pImplDevice = std::shared_ptr<DeviceVector<ValueType>>
                                (new DeviceVector<ValueType>());
    pImplDevice->Allocate(pImpl->GetSize());
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
    assert(size>0);
    if ( pImpl == pImplHost )
    {
        pImplHost.reset();
        pImplHost = std::shared_ptr<HostVector<ValueType>>
                                    (new HostVector<ValueType>());
        assert(pImplHost != NULL);
        pImplHost->Allocate(size);
        pImpl = pImplHost;
    }
    else if (pImpl == pImplDevice )
    {
        pImplDevice.reset();
        pImplDevice = std::shared_ptr<DeviceVector<ValueType>>
                                    (new DeviceVector<ValueType>());
        assert(pImplDevice != NULL);
        pImplDevice->Allocate(size);
        pImpl = pImplDevice;

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
    assert(&otherVector != NULL);
    if (this == &otherVector)
       return *this;
    
    int size = otherVector.GetSize();
    this->Allocate(size);
    // we need to pass a reference (BaseVector&)
    // otherVector.pImpl is BaseVector*
    pImpl->CopyFromHost(*(otherVector.pImpl));

    // Returning a Vector object allows chaining assigment
    // a = b = c;
    // Returning by reference makes that no copy of the object is created
    // and destroyed
    return *this;


}

template <typename ValueType>
void Vector<ValueType>::operator+=(
                                const Vector<ValueType>& otherVector)
{
    assert(GetSize() == otherVector.GetSize());
    pImpl->Add(*(otherVector.pImpl));
}

template <typename ValueType>
Vector<ValueType> Vector<ValueType>::operator+(
                                const Vector<ValueType>& otherVector)
{
    assert(GetSize() == otherVector.GetSize());

    Vector<ValueType> result(GetSize());

    result.pImpl->Add(*(otherVector.pImpl), *pImpl);

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
    assert(GetSize() == otherVector.GetSize()); 
    return pImpl->Dot(*(otherVector.pImpl));
    
}
                                

// Instantiate the class for the supported template type 
// parameters. If not done you have to include the 
// implementation in the header
template class Vector<double>;
template class Vector<float>;
template class Vector<int>;

}
