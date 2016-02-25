#include "BaseMatrix.hpp"

#include <iostream>
#include <fstream>
#include <regex>


namespace pssolver
{
template <typename ValueType>
BaseMatrix<ValueType>::BaseMatrix() : mNRows(0), mNCols(0), mNnz(0) {}

template <typename ValueType>
BaseMatrix<ValueType>::~BaseMatrix() {}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNRows(void) const
{
    return mNRows;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNCols(void) const
{
    return mNCols;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNnz(void) const
{
    return mNnz;
}

template <typename ValueType>
void BaseMatrix<ValueType>::ReadFile(const std::string filename)
{
    
    std::ifstream mFile(filename);
    std::string line;
    std::getline(mFile, line);
    int linenumber = 1;
    if ( !regex_match ( line, std::regex("%%MatrixMarket matrix"
                                    " coordinate (real|complex)"
                                    " (symmetric|unsymmetric)")))
    {
        std::cerr << "Bad syntax line 1" << std::endl;
    }
    else if ( regex_search (line, std::regex("symmetric")) )
        mProperties.IsSymmetric = true;
    else if ( regex_search (line, std::regex("real")) )
        mProperties.IsReal = true;
    else
    {
        mProperties.IsSymmetric = false;
        mProperties.IsReal = false;
    }

    // Next line
    linenumber++;
    std::getline(mFile, line);
    // Check syntax 
    // ?: make the inner group a non-capturing group
    // \\s double escape to get the escape character to the engine
    // Accepts any number of blanks between numbers
    if ( !regex_match (line, std::regex("^((?:[1-9][0-9]*\\s*?){3})")))
    {
        std::cerr << "Bad syntax line 2" << std::endl;
    }
    std::istringstream linestream(line);
    linestream >> mNRows >> mNCols >> mNnz;
    // The stream is in position for reading the data (line 3)
    // That is done in the subclasses
    // getline(mFile,line);

}

template class BaseMatrix<double>;
template class BaseMatrix<float>;

}
