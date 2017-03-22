# distutils: extra_compile_args = -fopenmp -march=native -O3 -Wall
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
# cython: initializedcheck = False

import cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
cimport openmp
import numpy as np
cimport numpy as np
cimport libc.math as cmath
from libc.stdio cimport printf

cdef:
    double cc = 299792458    #speed of light in vaccum
    double ee = 1.602e-19    #charge of electron
    double me = 9.109e-31    #mass of electron

cdef struct Particle:
    double x, y        #m
    double vx, vy      #m/s
    double m           #kg
    double q           #ee
    double t           #flight time, s

cdef Particle setElectron(Particle *par, double energy, double divergence) nogil:     #Energy in MeV, divergence in mrad
    cdef double gamma = energy*1e6*ee/(me*cc*cc)
    cdef double beta = cmath.sqrt(1-1/(gamma*gamma))
    par.q = -1
    par.m = gamma * me
    par.vx = beta * cc * cmath.cos(divergence/1000)
    par.vy = beta * cc * cmath.sin(divergence/1000)
    par.x = 0
    par.y = 0
    par.t = 0

cdef int find_nearest(double[:] data, double needle) nogil:
    cdef:
        int size = data.shape[0]
        int i, arg
        double diff, newdiff
    arg = 0
    diff = cmath.fabs(data[0] - needle)
    for i in range(1, size):
        newdiff = cmath.fabs(data[i] - needle)
        if  newdiff < diff:
            diff = newdiff
            arg = i
    return arg

cdef int find_nearest_sorted(double[:] data, double needle, int start, int stop):
    #TODO
    if (data[start]-needle)*(data[stop]-needle) > 0 :
        #not found
        pass
    arg = 0
    return arg

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

    cdef inline double accessB(self, int x, int y) nogil:
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
            Particle test
        en_size = energies.shape[0]
        div_size = divergences.shape[0]
        x_pos = np.ndarray(shape=(en_size, div_size), dtype=float)
        y_pos = np.ndarray(shape=(en_size, div_size), dtype=float)
        times = np.ndarray(shape=(en_size, div_size), dtype=float)
        results = np.ndarray(shape=(en_size, div_size), dtype=np.int32)

        with nogil, parallel():
            test = test
            for i in prange(en_size, schedule='dynamic'):
                En = energies[i]
                for j in range(div_size):
                    div = divergences[j]
                    setElectron(&test, En, div)
                    results[i, j] = self.run(&test)
                    x_pos[i, j] = test.x
                    y_pos[i, j] = test.y

        return x_pos, y_pos, times, results

    def getSolvers(self, np.ndarray[double, ndim=1] energies, np.ndarray[double, ndim=1] divergences):
        x_pos, y_pos, times, results = self.getSolverData(energies, divergences)
        side = None
        front = None
        side_en = (results == 1).any(axis=1)
        front_en = (results == 2).any(axis=1)
        if side_en.any() :
            side = pyEDPSolver()
            valid = results[side_en] == 1
            side_valid = np.array([ np.nonzero(row)[0][[1,-1]] for row in valid ], dtype=np.int32)
            side.init(energies[side_en], divergences, times[side_en], x_pos[side_en], side_valid)
        if front_en.any() :
            front = pyEDPSolver()
            valid = results[front_en] == 2
            front_valid = np.array([ np.nonzero(row)[0][[1,-1]] for row in valid ], dtype=np.int32)
            front.init(energies[front_en], divergences, times[front_en], y_pos[front_en], front_valid)
        return side, front


cdef class pyEDPSolver:
    cdef:
        double[:] energies, divergences
        double[:, :] times, positions
        int[:, :] valid

    @classmethod
    def fromdata(cls, double[:] energies not None,double[:] divergences not None,
            double[:, :] times not None, double[:, :] positions not None, int[:, :] valid not None):
        newsolver = cls()
        newsolver.init(energies, divergences, times, positions, valid)
        return newsolver

    @classmethod
    def fromfile(cls, f):
        newsolver = cls()
        newsolver.load(f)
        return newsolver

    def init(self, double[:] energies not None,double[:] divergences not None,
            double[:, :] times not None, double[:, :] positions not None, int[:, :] valid not None):
        self.energies = energies
        self.divergences = divergences
        self.times = times
        self.positions = positions
        self.valid = valid
        #check if input is valid
        #if not ((energies.shape[0] == times.shape[0]) and (energies.shape[0] == positions.shape[0]) and (divergences.shape[0] == positions.shape[1])) :
        #    raise ValueError("Unmatched input size")
        #TODO: check if sorted

    def save(self, f):
        np.savez(f, energies = self.energies, divergences = self.divergences, times = self.times, positions = self.positions, valid=self.valid)

    def load(self, f):
        datas = np.load(f)
        energies = datas["energies"]
        divergences = datas["divergences"]
        times = datas["times"]
        positions = datas["positions"]
        valid = datas["valid"]
        self.init(energies, divergences, times, positions, valid)
        datas.close()

    cpdef double getP(self, double E, double D):
        cdef:
            int en_nearest, div_nearest
            double result
        en_nearest = find_nearest(self.energies, E)
        div_nearest = find_nearest(self.divergences, D)
        result = self.positions[en_nearest, div_nearest]
        return result

    cpdef double getE(self, double P, double D):
        cdef:
            int pos_nearest, div_nearest
        div_nearest = find_nearest(self.divergences, D)
        pos_nearest = find_nearest(self.positions[:, div_nearest], P)
        return self.energies[pos_nearest]

    cpdef double getT(self, double E, double D):
        cdef:
            int en_nearest, div_nearest
        en_nearest = find_nearest(self.energies, E)
        div_nearest = find_nearest(self.divergences, D)
        return self.times[en_nearest, div_nearest]
