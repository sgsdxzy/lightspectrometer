#!/bin/bash

g++ -Wall -march=native -O3 -std=c++11 edpsolver.cpp get_solver_data.cpp magnet.cpp spectrometer.cpp -o get_solver_data -lboost_serialization
