#!/bin/bash

file="$1"
output=$(basename $file .txt).mtx
echo $output

awk -F'[(,)]' '{printf "%s %s %8.3e\n", ++$2, ++$3, $4}' $file > $output

nnz=$(wc -l $output | cut -d\  -f1)
nrows=$(tail -1 $output | cut -d\   -f1)

sed -i '1i%%MatrixMarket matrix coordinate real unsymmetric' $output
sed -i "2i$nrows $nrows $nnz" $output

vectorfile="$2"
sed -i '$ d' $vectorfile
size=$(wc -l $vectorfile | cut -d\  -f1)
sed -i "1i$size" $vectorfile
