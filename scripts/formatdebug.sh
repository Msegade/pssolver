#!/bin/bash

#awk '$1 in a {print a[$1], $0; next}; {a[++l] = $0}' < "$1"
gawk '
  $1 in a { a[$1] = a[$1] " " $0; next }
  { a[++l] = $0 }
  END { for (i = 1; i <= length(a); i++) print a[i] }
' < "$1"
