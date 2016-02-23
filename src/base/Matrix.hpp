#pragma once

#include "BaseMatrix.hpp"
#include "host/HostMatrix.hpp"
#include "host/HostCsrMatrix.hpp"
#include "device/DeviceMatrix.hpp"

#include "Vector.hpp"
#include "BaseVector.hpp"

#include <memory>
#include <string>
#include <iostream>

namespace pssolver
{

//Floats and doubles
template <typename ValueType>
class Matrix
{

public:
    Matrix();
    Matrix(int nRows, int nCols, int nnz);
    virtual ~Matrix();

    MatrixType GetFormat(void) const;

    bool IsHost(void) const;
    bool IsDevice(void) const;

    void ReadFile(const std::string filename);
    void Allocate(int nRows, int nCols, int nnz, MatrixType format);

    void ConvertTo(MatrixType format);

    int GetNRows(void) const;
    int GetNCols(void) const;
    int GetNnz(void) const;

    ValueType operator()(int i, int j);  // 0-base indexing

    Matrix<ValueType>& operator=(const Matrix<ValueType>& otherMatrix);
    Vector<ValueType> operator*(const Vector<ValueType>& vec);


    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& Mat);

private:
    std::shared_ptr<BaseMatrix<ValueType>> pImpl;
    std::shared_ptr<HostMatrix<ValueType>> pImplHost;
    std::shared_ptr<DeviceMatrix<ValueType>> pImplDevice;

};

template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Matrix<ValueType>& Mat);

}
