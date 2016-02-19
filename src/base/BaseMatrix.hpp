#pragma once

namespace pssolver
{
    
enum MatrixType { CSR, COO };
// Base class for the implementations of host and device vectors
template <typename ValueType>
class BaseMatrix
{

public:
    BaseMatrix();
    virtual ~BaseMatrix();

    virtual MatrixType GetFormat(void) const = 0;
    
    int GetNRows(void) const;
    int GetNCols(void) const;
    int GetNnz(void) const;

    virtual void Allocate(const int nRows, const int nCols, const int nNz) = 0;


protected:
    int mNRows, mNCols, mNnz;

};



}
