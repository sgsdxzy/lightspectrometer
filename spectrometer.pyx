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
        double *B;
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
        void getSolverData(double *Ens, int en_size, double *divergences, int div_size, double *x_pos, double *y_pos, int* results, double* times, int central_index) const;

    cdef cppclass EDPSolver:
        double *energies;
        double *divergences;
        double *times;
        double *positions;
        int div_size, en_size, central_index;

        double getP(double E, double D) const;
        double getE(double P, double D) const;
        double getT(double E) const;

cdef class pyEDPSolver:
    cdef EDPSolver solver
    cdef public np.ndarray energies, divergences, times, positions

    @classmethod
    def fromdata(cls, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None,
            np.ndarray[double, ndim=1, mode="c"] times not None, np.ndarray[double, ndim=2, mode="c"] positions not None):
        newsolver = cls()
        newsolver.init(energies, divergences, times, positions)
        return newsolver

    @classmethod
    def fromfile(cls, f):
        newsolver = cls()
        newsolver.load(f)
        return newsolver

    def init(self, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None,
            np.ndarray[double, ndim=1, mode="c"] times not None, np.ndarray[double, ndim=2, mode="c"] positions not None):
        #check if input is valid
        if not ((energies.shape[0] == times.shape[0]) and (energies.shape[0] == positions.shape[0]) and (divergences.shape[0] == positions.shape[1])) :
            raise ValueError("Unmatched input size")
        #TODO: check if sorted

        self.energies = energies
        self.divergences = divergences
        self.times = times
        self.positions = positions

        self.solver.energies = &energies[0]
        self.solver.divergences = &divergences[0]
        self.solver.times = &times[0]
        self.solver.positions = &positions[0, 0]
        self.solver.div_size = self.divergences.shape[0]
        self.solver.en_size = self.energies.shape[0]
        self.solver.central_index = np.argmin(np.abs(divergences))

    def save(self, f):
        np.savez(f, energies = self.energies, divergences = self.divergences, times = self.times, positions = self.positions)

    def load(self, f):
        datas = np.load(f)
        energies = datas["energies"]
        divergences = datas["divergences"]
        times = datas["times"]
        positions = datas["positions"]
        self.init(energies, divergences, times, positions)
        datas.close()

    def getP(self, E, D):
        return self.solver.getP(E, D)

    def getE(self, P, D):
        return self.solver.getE(P, D)

    def getT(self, E):
        return self.solver.getT(E)


cdef class pySpectrometer:
    cdef Spectrometer spec
    cdef public np.ndarray B
    cdef double dt_multiplier

    @classmethod
    def fromdata(cls, np.ndarray[double, ndim=2, mode="c"] B not None, x_delta, y_delta, x_offset=0, y_offset=0, dt_multiplier=0.0001, maxtime=1e-6):
        newspec = cls()
        newspec.init(B, x_delta, y_delta, x_offset, y_offset, dt_multiplier, maxtime)
        return newspec

    @classmethod
    def fromfile(cls, f):
        newspec = cls()
        newspec.load(f)
        return newspec

    def init(self, np.ndarray[double, ndim=2, mode="c"] B not None, x_delta, y_delta, x_offset=0, y_offset=0, dt_multiplier=0.0001, maxtime=1e-4):
        self.B = B
        self.dt_multiplier = dt_multiplier
        self.spec.mag.B = &B[0, 0]
        self.spec.mag.x_grid = self.B.shape[1]
        self.spec.mag.y_grid = self.B.shape[0]
        self.spec.mag.x_delta = x_delta
        self.spec.mag.y_delta = y_delta
        self.spec.x_offset = x_offset
        self.spec.y_offset = y_offset
        self.spec.initdt(self.dt_multiplier)
        self.spec.maxtime = maxtime

    def settimemultiplyer(self, dt_multiplier):
        self.dt_multiplier = dt_multiplier
        self.spec.initdt(self.dt_multiplier)

    def setmaxtime(self, maxtime):
        self.spec.maxtime = maxtime

    def save(self, f):
        np.savez(f, B = self.B, paras = (self.spec.mag.x_delta, self.spec.mag.y_delta, self.spec.x_offset, self.spec.y_offset, self.dt_multiplier, self.spec.maxtime))

    def load(self, f):
        datas = np.load(f)
        B = datas["B"]
        paras = datas["paras"]
        self.init(B, *paras)
        datas.close()

    def getSolvers(self, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None):
        cdef np.ndarray[double, ndim=2, mode="c"] x_pos = np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=float, order="c")
        cdef np.ndarray[double, ndim=2, mode="c"] y_pos = np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=float, order="c")
        cdef np.ndarray[int, ndim=2, mode="c"] results =  np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=np.int32, order="c")
        cdef np.ndarray[double, ndim=1, mode="c"] times = np.ndarray(shape=(energies.shape[0], ), dtype=float, order="c")
        cdef int central_index = np.argmin(np.abs(divergences))
        self.spec.getSolverData(&energies[0], energies.shape[0], &divergences[0], divergences.shape[0], &x_pos[0, 0], &y_pos[0, 0], &results[0, 0], &times[0], central_index)

        side = None
        front = None
        side_en = (results == 1).all(axis=1)
        front_en = (results == 2).all(axis=1)
        if side_en.any() :
            side = pyEDPSolver()
            side.init(energies[side_en], divergences, times[side_en], x_pos[side_en])
        if front_en.any() :
            front = pyEDPSolver()
            front.init(energies[front_en], divergences, times[front_en], y_pos[front_en])
        return side, front
