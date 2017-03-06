#ifndef LIGHT_SPECTROMETER
#define LIGHT_SPECTROMETER

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using std::endl;
using std::ostream;
using std::istream;
using std::vector;
using std::distance;
using std::lower_bound;

const double cc = 299792458;    //speed of light in vaccum
const double ee = 1.602e-19;    //charge of electron
const double me = 9.109e-31;    //mass of electron

class Particle
{
public:
    double x, y;        //m
    double vx, vy;      //m/s
    double m;           //kg
    double q;           //ee

    double t = 0;           //flight time, s

    void setElectron(double energy, double divergence);     //Energy in MeV, divergence in mrad
};

class Magnet 
{
private:
    double accessB(int x, int y) const;   //return B[y][x]

public:
    double* B;                      //The magnetic filed block, unit in T
    int x_grid, y_grid;             //The number of grids in x and y direction
    double x_delta, y_delta;        //The step of grids in x and y direction, units in SI(m)

    double getB(double x, double y) const;  //Get the magnetic field at physical space (x,y)
    //friend ostream& operator<<(ostream& out, const Magnet& mag);
    //friend istream& operator>>(istream& in, Magnet& mag);
};

class EDPSolver;

class Spectrometer
{
private:
    double dt;          //s
public:
    Magnet mag;
    double x_offset = 0, y_offset = 0;      //The position of magnet (0,0) in physical space
    double maxtime = 1;

    void initdt(double dt_multiplier);                                    //Calculate dt
    int run(Particle& par) const;                         //Push the particle in magnetic field until condition is met, 4 = timeout
    virtual int condition(Particle& par) const;           //Whether a partile hits detector or is lost and stops running, 0 = keep running, 1 = hit side, 2 = hit front, 3 = hit other
    void getSolverData(double *Ens, int en_size, double *divergences, int div_size, double *x_pos, double *y_pos, int* results, double* times) const;
};

class EDPSolver
{
public :
    double *energies, *divergences, *times, *positions; //data
    int div_size, en_size, div_0_index;

    //double div_min, div_max, en_min, en_max, pos_min, pos_max; // all ranges are [min, max)
    double ddiv, den;   //deltas

    double position(int en_index, int div_index) const;
    
    double getP(double E, double D) const;
    double getE(double P, double D) const;
    double getT(double E) const;

    //template<class Archive>
    //void serialize(Archive & ar, const unsigned int version)
    //{
    //    ar & divergences;
    //    ar & energies;
    //    ar & times;
    //    ar & positions;
    //}    
};

#endif //LIGHT_SPECTROMETER
