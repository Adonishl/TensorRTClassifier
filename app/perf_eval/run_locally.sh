#!/bin/bash

mkdir -p build  && cd build && \
cmake .. &&  make -j && \
echo "flags: $@" && \
./PerfEval  $@
