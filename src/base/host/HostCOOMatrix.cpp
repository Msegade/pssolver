#include "HostCOOMatrix.hpp"
#include "../../utils.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <regex>

namespace pssolver
{
template <typename ValueType>
HostCOOMatrix<ValueType>::HostCOOMatrix()
{
    mData = new ValueType[0];
    mColInd = new int[0];
    mRowInd = new int[0];
}

template <typename ValueType>
HostCOOMatrix<ValueType>::~HostCOOMatrix()
{
    delete[] mData;
    delete[] mRowInd;
    delete[] mColInd;

}

template <typename ValueType>
void HostCOOMatrix<ValueType>::Allocate(const int nRows,
                                        const int nCols,const int nnz)
{
    DEBUGLOG(this, "HostCOOMatrix::Allocate()", 
            "nRows = " << nRows << " nCols = " << nCols << " nnz = " << nnz, 2);

    assert ( nRows > 0 && nCols > 0 && nnz > 0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    mData = new ValueType[nnz];
    mColInd = new int[nnz];
    mRowInd = new int[nnz];

    memset(mData, 0, nnz*sizeof(ValueType));
    memset(mColInd, 0, nnz*sizeof(int));
    memset(mRowInd, 0, nnz*sizeof(int));
}

template <typename ValueType>
void HostCOOMatrix<ValueType>::ReadFile(const std::string filename) 
{
    DEBUGLOG(this, "HostCOOMatrix::ReadFile()", "filename = " << filename, 2);
    BaseMatrix<ValueType>::ReadFile(filename);

    //Allocate matrix after reading the size from the file
    this->Allocate(this->mNRows, this->mNCols, this->mNnz);
    // Re open file
    std::ifstream mFile;
    mFile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    mFile.open(filename);
    // Go to line 3 where the data begins

    GoToLine(mFile, 3);
    std::string line;
    std::getline(mFile, line);
    // Only check syntax on first line for performance issues
    // Checking regex in each iteration very slow
    if (!regex_match (line, std::regex("^((?:(?:[0-9][0-9]*\\s+?){2})"
                                "-?[0-9\\.]+e(?:\\+|\\-)[0-9]+)")))
    {
        std::cerr << "Bad syntax in line: 3" << std::endl;
    }
    GoToLine(mFile, 3);
    std::istringstream linestream;
    for (int index=0; index<this->mNnz; index++)
    {
        std::getline(mFile, line);
        std::cout << line << std::endl;
        linestream.str(line); 
        int rowInd, colInd;
        linestream >> rowInd >> colInd >> mData[index];
        mRowInd[index] = rowInd - 1;
        mColInd[index] = colInd - 1;
        linestream.clear();
        
    }
    mFile.close();

}

template <typename ValueType>
void HostCOOMatrix<ValueType>::Print(std::ostream& os)
{
    os << "Number of Rows: " << this->mNRows << std::endl;
    os << "Number of Columns: " << this->mNCols << std::endl;
    os << "Number of Non zero: " << this->mNnz << std::endl;
    os << "Row Index" << "\t" << "Col Index" << "\t" << 
                                                    "Data" << std::endl;
    for (int i= 0; i< this->mNnz; i++)
    {
        os << mRowInd[i] << "\t" << mColInd[i] << "\t" << 
                                                    mData[i] << std::endl;
    }
}

template class HostCOOMatrix<double>;
template class HostCOOMatrix<float>;

}
