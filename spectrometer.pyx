# distutils: extra_compile_args = -fopenmp -march=native -O3 -Wall
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
# cython: initializedcheck = False

import cython
from cython cimport view
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
cimport openmp
import numpy as np
cimport numpy as cnp
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

cdef inline int min(int a, int b) nogil:
    if (a>b) :
        return b
    else :
        return a

cdef inline int max(int a, int b) nogil:
    if (a>b) :
        return a
    else :
        return b

cdef int find_nearest(double[:] data, double needle, int start, int stop) nogil:
    cdef:
        int i, arg
        double diff, newdiff
    arg = start
    diff = cmath.fabs(data[start] - needle)
    for i in range(start, stop+1):
        newdiff = cmath.fabs(data[i] - needle)
        if  newdiff < diff:
            diff = newdiff
            arg = i
    return arg

cdef inline int find_nearest_evenly_spaced(double first, double delta, double needle) nogil:
    return <int>cmath.round((needle-first)/delta)

cdef int find_nearest_sorted(double[:] data, double needle, int start, int stop) nogil:
    if (data[start]-needle)*(data[stop]-needle) > 0 :
        #not in range
        return -1
    return find_nearest_iter(data, needle, start, stop)

cdef int find_nearest_iter(double[:] data, double needle, int start, int stop) nogil:
    if stop - start == 1:
        if cmath.fabs(data[start]-needle) <= cmath.fabs(data[stop]-needle) :
            return start
        else :
            return stop
    cdef int middle = (start+stop)/2
    if (data[start]-needle)*(data[middle]-needle) <= 0 :
        return find_nearest_iter(data, needle, start, middle)
    else :
        return find_nearest_iter(data, needle, middle, stop)

cdef inline int find_left_evenly_spaced(double first, double delta, double needle) nogil:
    return <int>cmath.floor((needle-first)/delta)

cdef inline int find_left_sorted(double[:] data, double needle, int start, int stop) nogil:
    if (start == stop) or (data[start]-needle)*(data[stop]-needle) > 0 :
        #not in range
        return -1
    return find_left_iter(data, needle, start, stop)

cdef int find_left_iter(double[:] data, double needle, int start, int stop) nogil:
    if stop - start == 1:
        return start
    cdef int middle = (start+stop)/2
    if (data[start]-needle)*(data[middle]-needle) <= 0 :
        return find_left_iter(data, needle, start, middle)
    else :
        return find_left_iter(data, needle, middle, stop)

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

    #cdef inline double accessB(self, int x, int y) nogil:
        #if ((x<0) or (x>=self.x_grid) or (y<0) or (y>=self.y_grid)) :
        #    return 0
        #return self.B[y, x];

    cdef double getB(self, double x, double y) nogil:
        '''Get the magnetic field at physical space (x,y)'''
        x -= self.x_offset
        y -= self.y_offset
        cdef int x_left = <int>cmath.floor(x/self.x_delta)
        cdef int y_left = <int>cmath.floor(y/self.y_delta)
        cdef double pr = self.B[y_left, x_left]
        cdef double qr = self.B[y_left, x_left+1]
        cdef double ps = self.B[y_left+1, x_left]
        cdef double qs = self.B[y_left+1, x_left+1]
        cdef double p = x - x_left*self.x_delta
        cdef double q = self.x_delta - p
        cdef double r = y - y_left*self.y_delta
        cdef double s = self.y_delta - r
        return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(self.x_delta*self.y_delta)

    cdef int run(self, Particle *par) nogil:
        '''Push the particle in magnetic field until condition() is met, return result: 0-4 same as condition(), 5 = timeout'''
        cdef:
            double B, t, s, vpx, vpy, time
            int result = 0

        #push particle from source to left edge of B region
        if (par.x < self.x_offset) :
            time = (self.x_offset - par.x)/par.vx
            par.x = self.x_offset
            par.y += par.vy * time
            par.t += time
        result = self.condition(par)
        if (result != 0):
            return result

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
            if (result == 4):
                time = (622e-3 - par.x)/par.vx
                par.x = 622e-3
                par.y += par.vy * time
                par.t += time
                return 2
            if (result != 0):
                return result
        return 5

    cdef int condition(self, Particle* par) nogil:
        '''Whether a partile hits detector or is lost and stops running, return status: 0 = keep running, 1 = hit side, 2 = hit front, 3 = hit other, 4 = exit B region'''
        if ((par.x >= 152e-3) and (par.x <= 522e-3) and (par.y >= 100e-3)):
            return 1
        if ((par.x < self.x_offset) or (par.y <= self.y_offset) or (par.y >= 100e-3)):
            return 3
        if ((par.x >= 532e-3)):
            return 4
        return 0;

    cdef getSolverData(self, double[:] energies, double[:] divergences):
        cdef:
            double En, div;
            int i, j, en_size, div_size;
            cnp.ndarray[double, ndim=2] x_pos, y_pos, times
            cnp.ndarray[int, ndim=2] results
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

    def getSolvers(self, cnp.ndarray[double, ndim=1] energies, cnp.ndarray[double, ndim=1] divergences):
        '''Get side and front solvers. energeis and divergences must be in increasing order and evenly spaced.'''
        x_pos, y_pos, times, results = self.getSolverData(energies, divergences)
        side = None
        front = None
        side_en = (results == 1).any(axis=1)
        front_en = (results == 2).any(axis=1)
        if side_en.any() :
            side = pyEDPSolver()
            valid = results[side_en] == 1
            en_valid = np.array([ np.nonzero(row)[0][[0,-1]] for row in valid.T ], dtype=np.int32)
            div_valid = np.array([ np.nonzero(row)[0][[0,-1]] for row in valid ], dtype=np.int32)
            side.init(energies[side_en], divergences, times[side_en], x_pos[side_en], en_valid, div_valid)
        if front_en.any() :
            front = pyEDPSolver()
            valid = results[front_en] == 2
            en_valid = np.array([ np.nonzero(row)[0][[0,-1]] for row in valid.T ], dtype=np.int32)
            div_valid = np.array([ np.nonzero(row)[0][[0,-1]] for row in valid ], dtype=np.int32)
            front.init(energies[front_en], divergences, times[front_en], y_pos[front_en], en_valid, div_valid)
        return side, front


cdef class pyEDPSolver:
    cdef:
        double[:] energies, divergences
        double[:, :] times, positions
        int[:, :] en_valid, div_valid
        int en_size, div_size
        double en_delta, div_delta
        double en_min, en_max, div_min, div_max

    @classmethod
    def fromdata(cls, double[:] energies not None,double[:] divergences not None, double[:, :] times not None,
            double[:, :] positions not None, int[:, :] en_valid not None, int[:, :] div_valid not None):
        newsolver = cls()
        newsolver.init(energies, divergences, times, positions, en_valid, div_valid)
        return newsolver

    @classmethod
    def fromfile(cls, f):
        newsolver = cls()
        newsolver.load(f)
        return newsolver

    def init(self, double[:] energies not None,double[:] divergences not None, double[:, :] times not None,
            double[:, :] positions not None, int[:, :] en_valid not None, int[:, :] div_valid not None):
        self.energies = energies
        self.divergences = divergences
        self.times = times
        self.positions = positions
        self.en_valid = en_valid
        self.div_valid = div_valid
        self.en_size = self.energies.shape[0]
        self.div_size = self.divergences.shape[0]
        self.en_min = self.energies[0]
        self.en_max = self.energies[self.en_size-1]
        self.div_min = self.divergences[0]
        self.div_max = self.divergences[self.div_size-1]
        if self.en_size > 1:
            self.en_delta = (self.en_max - self.en_min)/(self.en_size-1)
        if self.div_size > 1:
            self.div_delta = (self.div_max - self.div_min)/(self.div_size-1)

    def save(self, f):
        np.savez(f, energies = self.energies, divergences = self.divergences, times = self.times, positions = self.positions, en_valid=self.en_valid, div_valid=self.div_valid)

    def load(self, f):
        datas = np.load(f)
        self.init(**datas)
        datas.close()

    #TODO: special case for 1-element data
    #In the series of get functions, a return value of 0 means invalid!
    cpdef double getP_fast(self, double E, double D) :
        cdef:
            int en_nearest, div_nearest
        if (E<self.en_min) or (E>self.en_max) or (D<self.div_min) or (D>self.div_max):
            return 0
        en_nearest = find_nearest_evenly_spaced(self.en_min, self.en_delta, E)
        div_nearest = find_nearest_evenly_spaced(self.div_min, self.div_delta, D)
        if (div_nearest < self.div_valid[en_nearest, 0]) or (div_nearest > self.div_valid[en_nearest, 1]):
            return 0
        return self.positions[en_nearest, div_nearest]

    cpdef double getE_fast(self, double P, double D) :
        cdef:
            int pos_nearest, div_nearest
        if (D<self.div_min) or (D>self.div_max):
            return 0
        div_nearest = find_nearest_evenly_spaced(self.div_min, self.div_delta, D)
        pos_nearest = find_nearest_sorted(self.positions[:, div_nearest], P, self.en_valid[div_nearest, 0], self.en_valid[div_nearest, 1])
        if pos_nearest == -1 :
            return 0
        else :
            return self.energies[pos_nearest]

    cpdef double getT_fast(self, double E, double D) :
        cdef:
            int en_nearest, div_nearest
        if (E<self.en_min) or (E>self.en_max) or (D<self.div_min) or (D>self.div_max):
            return 0
        en_nearest = find_nearest_evenly_spaced(self.en_min, self.en_delta, E)
        div_nearest = find_nearest_evenly_spaced(self.div_min, self.div_delta, D)
        if (div_nearest < self.div_valid[en_nearest, 0]) or (div_nearest > self.div_valid[en_nearest, 1]):
            return 0
        return self.times[en_nearest, div_nearest]

    cpdef double getP(self, double E, double D) :
        cdef:
            int en_left, div_left
            double pr, qr, ps, qs, p, q, r, s
        en_left = find_left_evenly_spaced(self.en_min, self.en_delta, E)
        div_left = find_left_evenly_spaced(self.div_min, self.div_delta, D)

        if ((en_left >=0) and (en_left+1<self.en_size) and (div_left >= self.div_valid[en_left, 0]) and (div_left+1 <= self.div_valid[en_left, 1])\
                and (div_left >= self.div_valid[en_left+1, 0]) and (div_left+1 <= self.div_valid[en_left+1, 1])):
            pr = self.positions[en_left, div_left]
            qr = self.positions[en_left, div_left+1]
            ps = self.positions[en_left+1, div_left]
            qs = self.positions[en_left+1, div_left+1]
            p = D - self.divergences[div_left]
            q = self.div_delta - p
            r = E - self.energies[en_left]
            s = self.en_delta - r
            return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(self.en_delta*self.div_delta)
        else :
            return 0

    cpdef double getE(self, double P, double D) :
        cdef:
            int pos_left, div_left
            double p, q
            int i, start, end, newsize
            double[:] pos_int
        if (D<self.div_min) or (D>=self.div_max) :
            return 0
        div_left = find_left_evenly_spaced(self.div_min, self.div_delta, D)
        start = max(self.en_valid[div_left, 0], self.en_valid[div_left+1, 0])
        end = min(self.en_valid[div_left, 1], self.en_valid[div_left+1, 1])
        newsize = end - start + 1
        if (newsize <= 1):
            return 0

        pos_int = np.ndarray(shape=(newsize, ), dtype=float)
        p = D - self.divergences[div_left]
        q = self.div_delta - p
        for i in range(newsize) :
            pos_int[i] = (self.positions[i+start, div_left]*q+self.positions[i+start, div_left+1]*p)/self.div_delta
        pos_left = find_left_sorted(pos_int, P, 0, newsize-1)
        if pos_left == -1 :
            return 0
        pos_left += start
        p = P - pos_int[pos_left]
        q = pos_int[pos_left+1] - P
        return (self.energies[pos_left]*q+self.energies[pos_left+1]*p)/(p+q)

    cpdef double getT(self, double E, double D) :
        cdef:
            int en_left, div_left
            double pr, qr, ps, qs, p, q, r, s
        en_left = find_left_evenly_spaced(self.en_min, self.en_delta, E)
        div_left = find_left_evenly_spaced(self.div_min, self.div_delta, D)

        if (en_left >=0) and (en_left+1<self.en_size) and (div_left >= self.div_valid[en_left, 0]) and (div_left+1 <= self.div_valid[en_left, 1])\
                and (div_left >= self.div_valid[en_left+1, 0]) and (div_left+1 <= self.div_valid[en_left+1, 1]):
            pr = self.times[en_left, div_left]
            qr = self.times[en_left, div_left+1]
            ps = self.times[en_left+1, div_left]
            qs = self.times[en_left+1, div_left+1]
            p = D - self.divergences[div_left]
            q = self.div_delta - p
            r = E - self.energies[en_left]
            s = self.en_delta - r
            return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(self.en_delta*self.div_delta)
        else :
            return 0
