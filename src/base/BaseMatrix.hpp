#pragma once

#include <string>
#include <iostream>

namespace pssolver
{

struct MatrixProperties
{
    bool IsSymmetric;
    bool IsReal;
    bool IsSparse;
};
    
enum MatrixType { CSR, COO };

template <typename ValueType>
class HostCsrMatrix;
template <typename ValueType>
class HostCOOMatrix;
template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;
// Base class for the implementations of host and device vectors
template <typename ValueType>
class BaseMatrix
{

public:
    BaseMatrix();
    virtual ~BaseMatrix();

    virtual MatrixType GetFormat(void) const = 0;

    virtual void ReadFile(const std::string filename);

    virtual void Print(std::ostream& os) = 0;

    virtual void CopyFrom(BaseMatrix<ValueType> &mat);
    
    int GetNRows(void) const;
    int GetNCols(void) const;
    int GetNnz(void) const;

    virtual void Allocate(const int nRows, const int nCols, const int nNz) = 0;
    
    virtual ValueType Read(int i, int j) const = 0;

    virtual void MatVec(BaseVector<ValueType>& invec, 
                                        BaseVector<ValueType>& outvec,
                                        ValueType scalar) const = 0;


protected:
    int mNRows, mNCols, mNnz;

    MatrixProperties mProperties;


};



}
