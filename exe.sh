#!/bin/bash

mkdir -p build
cd build
cmake ..
make
./main > result.csv
python3 ../plot.py