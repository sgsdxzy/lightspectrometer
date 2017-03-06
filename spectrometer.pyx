# distutils: language = c++
# distutils: sources = cspectrometer.cpp
# distutils: extra_compile_args = -fopenmp -march=native -O3 -Wall
# distutils: extra_link_args = -fopenmp

import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "cspectrometer.h":
    cdef cppclass Magnet:
        vector[double] B;
        int x_grid, y_grid;
        double x_delta, y_delta;

        double getB(double x, double y) const;
 
    cdef cppclass Spectrometer:
        Magnet mag;
        double x_offset, y_offset; 
        double maxtime;

        void initdt(double dt_multiplier);
        #int run(Particle& par) const;                         
        #int condition(Particle& par) const; 
        void getSolverData(vector[double] &Ens, vector[double] &divergences, EDPSolver& side, EDPSolver& front) const;
     
    cdef cppclass EDPSolver:
        vector[double] divergences, energies, times, positions;

        double div_min, div_max, en_min, en_max, pos_min, pos_max;
        double ddiv, den;
        int div_size, en_size, pos_size, div_0_index;

        void init();
        void clear();

        double position(int en_index, int div_index) const;
        
        double getP(double E, double D) const;
        double getE(double P, double D) const;
        double getT(double E) const;

cdef class pyEDPSolver:
    cdef EDPSolver solver

    def save(self, f):
        pass

    def load(self, f):
        pass

    def init(self):
        self.solver.init()

    def clear(self):
        self.solver.clear()

    def getP(self, E, D):
        return self.solver.getP(E, D)

    def getE(self, P, D):
        return self.solver.getE(P, D)

    def getT(self, E):
        return self.solver.getT(E)

 
cdef class pySpectrometer:
    cdef Spectrometer spec

    def init(self, np.ndarray[double, ndim=2, mode="c"] B not None, x_delta=0.002, y_delta=0.0025, x_offset=140e-3, y_offset=-110e-3, dt_multiplier=0.0001, maxtime=1):
        self.spec.mag.B.assign(&B[0, 0], &B[0, 0] + B.shape[0]*B.shape[1])
        self.spec.mag.x_grid = B.shape[1]
        self.spec.mag.y_grid = B.shape[0]
        self.spec.mag.x_delta = x_delta
        self.spec.mag.y_delta = y_delta
        self.spec.x_offset = x_offset
        self.spec.y_offset = y_offset
        self.spec.initdt(dt_multiplier)
        self.spec.maxtime = maxtime

    def save(self, f):
        pass

    def load(self, f):
        pass

    def getSolvers(self, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None):
        side = pyEDPSolver()
        front = pyEDPSolver()
        cdef vector[double] Ens#(energies.shape[0], &energies[0])
        cdef vector[double] divs#(divergences.shape[0], &divergences[0])
        Ens.assign(&energies[0], &energies[0]+energies.shape[0])
        divs.assign(&divergences[0], &divergences[0]+divergences.shape[0])
        self.spec.getSolverData(Ens, divs, side.solver, front.solver)
        return side, front
