#pragma once

#include "BaseMatrix.hpp"
#include "host/HostMatrix.hpp"
#include "host/HostCsrMatrix.hpp"

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

    void ReadFile(const std::string filename);
    void AllocateCSR(int nRows, int nCols, int nnz);
    void AllocateCOO(int nRows, int nCols, int nnz);

    void ConvertTo(MatrixType format);

    int GetNRows(void) const;
    int GetNCols(void) const;
    int GetNnz(void) const;

    ValueType operator()(int i, int j);  // 0-base indexing


    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& Mat);

private:
    std::shared_ptr<BaseMatrix<ValueType>> pImpl;
    std::shared_ptr<HostMatrix<ValueType>> pImplHost;

};

template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Matrix<ValueType>& Mat);

}
