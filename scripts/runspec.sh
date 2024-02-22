#!/bin/bash

if [ $# -lt 2 ]
  then
    # Setting ncpu manually
    ncpu=8
  else
    ncpu=$2
fi

mpirun -n $ncpu ~/codes/SPEC/xspec $1
