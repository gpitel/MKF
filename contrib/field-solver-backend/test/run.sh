#!/bin/bash
# Standalone verification of CpuFieldSolver against the mock MKF types in test/mock.
# Proves the extracted loop compiles as valid C++23 and is bit-identical serial vs
# OpenMP. Does NOT need the MKF/MAS headers (which are mid-migration on this
# checkout and don't compile). Run: bash test/run.sh
set -e
cd "$(dirname "$0")/.."
FLAGS="-std=c++23 -O2 -Wall -Wextra -I test/mock -I ."
echo "=== serial ==="
g++ $FLAGS test/verify_standalone.cpp CpuFieldSolver.cpp -o /tmp/cfs_serial && /tmp/cfs_serial
echo "=== openmp (8 threads) ==="
g++ $FLAGS -fopenmp test/verify_standalone.cpp CpuFieldSolver.cpp -o /tmp/cfs_omp && OMP_NUM_THREADS=8 /tmp/cfs_omp
