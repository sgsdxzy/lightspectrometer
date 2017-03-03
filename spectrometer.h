#ifndef LIGHT_SPECTROMETER
#define LIGHT_SPECTROMETER

#include "magnet.h"

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
};

class Spectrometer
{
private:
    double dt;          //s
public:
    Magnet mag;
    double maxtime = 1;

    void init(istream& data);                                    //Calculate dt
    int run(Particle& par);                         //Push the particle in magnetic field until condition is met, 4 = timeout
    virtual int condition(Particle& par);           //Whether a partile hits detector or is lost and stops running, 0 = keep running, 1 = hit side, 2 = hit front, 3 = hit other
};

#endif //LIGHT_SPECTROMETER
