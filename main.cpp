#include <fstream>
#include <cmath>
#include <iostream>

#include "magnet.h"
#include "spectrometer.h"

//Energy in MeV
void setElectron(Particle& par, double energy)
{
    par.q = -1;
    double gamma = energy/0.511;
    double beta = sqrt(1-1/(gamma*gamma));
    par.m = gamma * me;
    par.vx = beta * cc;
    par.vy = 0;
    par.x = 0;
    par.y = 0;
}
    

int main(int argc, char** argv)
{
    Spectrometer spec;

    std::ifstream m1i ("magnet1.txt");
    m1i>>spec.mag;
    m1i.close();

    spec.mag.x_offset = 0;
    spec.mag.y_offset = -60e-3;
    spec.init();

    Particle test;
    double En;
    int result;
    //std::cout << spec.mag.getB(450e-3, 60e-3) << std::endl;
    while (true) {
        std::cin >> En;
        setElectron(test, En);

        result = spec.run(test);
        std::cout << test.vx << ' ' << result << ' ' << test.x << ' ' << test.y << std::endl;
    }
    

    return 0;
}
