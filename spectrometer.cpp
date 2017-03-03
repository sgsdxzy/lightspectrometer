#include <cmath>
#include "spectrometer.h"

void Spectrometer::init(istream& data)
{
    data>>mag;
    mag.x_offset = 140e-3;
    mag.y_offset = -110e-3;
    dt = 0.1/sqrt((cc/mag.x_delta)*(cc/mag.x_delta)+(cc/mag.y_delta)*(cc/mag.y_delta));
}

int Spectrometer::run(Particle& par)
{
    double B, t, s, vpx, vpy;
    int result = 0;

    while (par.t <= maxtime) {
        B = mag.getB(par.x, par.y);
        t = (par.q*ee*B/par.m)*dt/2;
        s = 2*t/(1+t*t);
        vpx = par.vx+t*par.vy;
        vpy = par.vy-t*par.vx;
        par.vx += s*vpy;
        par.vy -= s*vpx;
        par.x += par.vx*dt;
        par.y += par.vy*dt;

        par.t += dt;
        result = condition(par);
        if (result != 0) {
            return result;
        }
    }
    return 4;
}

int Spectrometer::condition(Particle& par)
{
    if ((par.x < -10e-3) || (par.y < -110e-3)) {
        return 3;
    }
    if ((par.x >= 152e-3) && (par.x <= 522e-3) && (par.y >= 100e-3)) {
        return 1;
    }
    if ((par.x >= 622e-3)) {
        return 2;
    }
    return 0;
}
