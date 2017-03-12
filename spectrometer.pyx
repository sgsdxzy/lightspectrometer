# distutils: extra_compile_args = -fopenmp -march=native -O3 -Wall
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
# cython: initializedcheck = False

import cython
from cython.parallel import parallel, prange
cimport openmp
import numpy as np
cimport numpy as np
cimport libc.math as cmath
from libc.stdio cimport printf

cdef:
    double cc = 299792458;    #speed of light in vaccum
    double ee = 1.602e-19;    #charge of electron
    double me = 9.109e-31;    #mass of electron

cdef struct Particle:
    double x, y;        #m
    double vx, vy;      #m/s
    double m;           #kg
    double q;           #ee
    double t;       #flight time, s

cdef void setElectron(Particle *par, double energy, double divergence) nogil:     #Energy in MeV, divergence in mrad
    cdef double gamma = energy*1e6*ee/(me*cc*cc);
    cdef double beta = cmath.sqrt(1-1/(gamma*gamma));
    par.q = -1;
    par.m = gamma * me;
    par.vx = beta * cc * cmath.cos(divergence/1000);
    par.vy = beta * cc * cmath.sin(divergence/1000);
    par.x = 0;
    par.y = 0;
    par.t = 0;


cdef class pySpectrometer:
    cdef:
        double[:,:] B                       #The magnetic filed block, unit in T
        double dt
        double dt_multiplier
        int x_grid, y_grid                  #The number of grids in x and y direction
        double x_delta, y_delta             #The step of grids in x and y direction, units in SI(m)
        double x_offset, y_offset           #The position of magnet (0,0) in physical space
        double maxtime                      #Maximum running time for a single particle

    @classmethod
    def fromdata(cls, double[:, :] B not None, x_delta, y_delta, x_offset=0, y_offset=0, dt_multiplier=0.01, maxtime=1e-3):
        newspec = cls()
        newspec.init(B, x_delta, y_delta, x_offset, y_offset, dt_multiplier, maxtime)
        return newspec

    @classmethod
    def fromfile(cls, f):
        newspec = cls()
        newspec.load(f)
        return newspec

    def init(self, double[:, :] B not None, x_delta, y_delta, x_offset=0, y_offset=0, dt_multiplier=0.01, maxtime=1e-3):
        self.B = B
        self.dt_multiplier = dt_multiplier
        self.x_grid = self.B.shape[1]
        self.y_grid = self.B.shape[0]
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.maxtime = maxtime
        self.setdt()

    def settimemultiplyer(self, dt_multiplier):
        self.dt_multiplier = dt_multiplier
        self.setdt()

    cdef setdt(self):
        self.dt = self.dt_multiplier/cmath.sqrt((cc/self.x_delta)*(cc/self.x_delta)+(cc/self.y_delta)*(cc/self.y_delta));

    def setmaxtime(self, maxtime):
        self.maxtime = maxtime

    def save(self, f):
        np.savez(f, B = self.B, paras = (self.x_delta, self.y_delta, self.x_offset, self.y_offset, self.dt_multiplier, self.maxtime))

    def load(self, f):
        datas = np.load(f)
        self.init(datas["B"], *datas["paras"])
        datas.close()

    cdef double accessB(self, int x, int y) nogil:
        if ((x<0) or (x>=self.x_grid) or (y<0) or (y>=self.y_grid)) :
            return 0
        return self.B[y, x];

    cdef double getB(self, double x, double y) nogil:
        '''Get the magnetic field at physical space (x,y)'''
        x -= self.x_offset
        y -= self.y_offset
        cdef int x_left = <int>cmath.floor(x/self.x_delta);
        cdef int y_left = <int>cmath.floor(y/self.y_delta);
        cdef double pr = self.accessB(x_left, y_left)
        cdef double qr = self.accessB(x_left+1, y_left);
        cdef double ps = self.accessB(x_left, y_left+1);
        cdef double qs = self.accessB(x_left+1, y_left+1);
        cdef double p = x - x_left*self.x_delta;
        cdef double q = (x_left+1)*self.x_delta - x;
        cdef double r = y - y_left*self.y_delta;
        cdef double s = (y_left+1)*self.y_delta - y;
        return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(self.x_delta*self.y_delta);

    cdef int run(self, Particle *par) nogil:
        '''Push the particle in magnetic field until condition() is met, return result: 0-3 same as condition(), 4 = timeout'''
        cdef:
            double B, t, s, vpx, vpy
            int result = 0

        while (par.t <= self.maxtime):
            B = self.getB(par.x, par.y)
            t = (par.q*ee*B/par.m)*self.dt/2
            s = 2*t/(1+t*t)
            vpx = par.vx+t*par.vy
            vpy = par.vy-t*par.vx
            par.vx += s*vpy
            par.vy -= s*vpx
            par.x += par.vx*self.dt
            par.y += par.vy*self.dt
            par.t += self.dt
            result = self.condition(par)
            if (result != 0):
                return result
        return 4

    cdef int condition(self, Particle* par) nogil:
        '''Whether a partile hits detector or is lost and stops running, return status: 0 = keep running, 1 = hit side, 2 = hit front, 3 = hit other'''
        if ((par.x < -10e-3) or (par.y < -110e-3)):
            return 3
        if ((par.x >= 152e-3) and (par.x <= 522e-3) and (par.y >= 100e-3)):
            return 1
        if ((par.x >= 622e-3)):
            return 2
        return 0;

    cdef getSolverData(self, double[:] energies, double[:] divergences):
        cdef:
            double En, div;
            int i, j, en_size, div_size;
            np.ndarray[double, ndim=2] x_pos, y_pos, times
            np.ndarray[int, ndim=2] results
            int num_threads = openmp.omp_get_num_threads()
            Particle test, blank
        en_size = energies.shape[0]
        div_size = divergences.shape[0]
        x_pos = np.ndarray(shape=(en_size, div_size), dtype=float)
        y_pos = np.ndarray(shape=(en_size, div_size), dtype=float)
        times = np.ndarray(shape=(en_size, div_size), dtype=float)
        results = np.ndarray(shape=(en_size, div_size), dtype=np.int32)

        for i in prange(en_size, nogil=True, schedule='dynamic'):
            En = energies[i]
            j = 0
            test = blank
            for j in range(div_size):
                div = divergences[j]
                setElectron(&test, En, div)
                results[i, j] = self.run(&test)
                x_pos[i, j] = test.x
                y_pos[i, j] = test.y

        return x_pos, y_pos, times, results

    def getSolvers(self, double[:] energies, double[:] divergences):
        return self.getSolverData(energies, divergences)

#     def getSolvers(self, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None):
#         cdef np.ndarray[double, ndim=2, mode="c"] x_pos = np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=float, order="c")
#         cdef np.ndarray[double, ndim=2, mode="c"] y_pos = np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=float, order="c")
#         cdef np.ndarray[int, ndim=2, mode="c"] results =  np.ndarray(shape=(energies.shape[0], divergences.shape[0]), dtype=np.int32, order="c")
#         cdef np.ndarray[double, ndim=1, mode="c"] times = np.ndarray(shape=(energies.shape[0], ), dtype=float, order="c")
#         cdef int central_index = np.argmin(np.abs(divergences))
#         self.spec.getSolverData(&energies[0], energies.shape[0], &divergences[0], divergences.shape[0], &x_pos[0, 0], &y_pos[0, 0], &results[0, 0], &times[0], central_index)
#
#         side = None
#         front = None
#         side_en = (results == 1).all(axis=1)
#         front_en = (results == 2).all(axis=1)
#         if side_en.any() :
#             side = pyEDPSolver()
#             side.init(energies[side_en], divergences, times[side_en], x_pos[side_en])
#         if front_en.any() :
#             front = pyEDPSolver()
#             front.init(energies[front_en], divergences, times[front_en], y_pos[front_en])
#         return side, front
#
# cdef class pyEDPSolver:
#     cdef EDPSolver solver
#     cdef public np.ndarray energies, divergences, times, positions
#
#     @classmethod
#     def fromdata(cls, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None,
#             np.ndarray[double, ndim=1, mode="c"] times not None, np.ndarray[double, ndim=2, mode="c"] positions not None):
#         newsolver = cls()
#         newsolver.init(energies, divergences, times, positions)
#         return newsolver
#
#     @classmethod
#     def fromfile(cls, f):
#         newsolver = cls()
#         newsolver.load(f)
#         return newsolver
#
#     def init(self, np.ndarray[double, ndim=1, mode="c"] energies not None, np.ndarray[double, ndim=1, mode="c"] divergences not None,
#             np.ndarray[double, ndim=1, mode="c"] times not None, np.ndarray[double, ndim=2, mode="c"] positions not None):
#         #check if input is valid
#         if not ((energies.shape[0] == times.shape[0]) and (energies.shape[0] == positions.shape[0]) and (divergences.shape[0] == positions.shape[1])) :
#             raise ValueError("Unmatched input size")
#         #TODO: check if sorted
#
#         self.energies = energies
#         self.divergences = divergences
#         self.times = times
#         self.positions = positions
#
#         self.solver.energies = &energies[0]
#         self.solver.divergences = &divergences[0]
#         self.solver.times = &times[0]
#         self.solver.positions = &positions[0, 0]
#         self.solver.div_size = self.divergences.shape[0]
#         self.solver.en_size = self.energies.shape[0]
#         self.solver.central_index = np.argmin(np.abs(divergences))
#
#     def save(self, f):
#         np.savez(f, energies = self.energies, divergences = self.divergences, times = self.times, positions = self.positions)
#
#     def load(self, f):
#         datas = np.load(f)
#         energies = datas["energies"]
#         divergences = datas["divergences"]
#         times = datas["times"]
#         positions = datas["positions"]
#         self.init(energies, divergences, times, positions)
#         datas.close()
#
#     def getP(self, E, D):
#         return self.solver.getP(E, D)
#
#     def getE(self, P, D):
#         return self.solver.getE(P, D)
#
#     def getT(self, E):
#         return self.solver.getT(E)
