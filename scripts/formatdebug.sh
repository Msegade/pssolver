#!/bin/bash

awk '$1 in a {print a[$1], $0; next}; {a[++l] = $0}' < "$1"
