#include <cmath>
#include "spectrometer.h"

void Spectrometer::init()
{
    dt = 0.1/sqrt((cc/mag.x_delta)*(cc/mag.x_delta)+(cc/mag.y_delta)*(cc/mag.y_delta));
}

int Spectrometer::run(Particle& par)
{
    double B, t, s, vpx, vpy;
    double time = 0;
    int result = 0;

    while (time <= maxtime) {
        B = mag.getB(par.x, par.y);
        t = (par.q*ee*B/par.m)*dt/2;
        s = 2*t/(1+t*t);
        vpx = par.vx+t*par.vy;
        vpy = par.vy-t*par.vx;
        par.vx += s*vpy;
        par.vy -= s*vpx;
        par.x += par.vx*dt;
        par.y += par.vy*dt;

        time += dt;
        result = condition(par);
        if (result != 0) {
            return result;
        }
    }

    return 4;
}

int Spectrometer::condition(Particle& par)
{
    if ((par.x < -10e-3) || (par.y < -60e-3)) {
        return 3;
    }
    if ((par.x <= 300e-3) && (par.y >= 100e-3)) {
        return 1;
    }
    if ((par.x >= 300e-3) && (par.x <= 330e-3) && (par.y >= 60e-3) && (par.y <= 100e-3)) {
        return 3;
    }
    if ((par.x >= 466e-3)) {
        return 2;
    }
    return 0;
}
