#include <fstream>
#include <cmath>
#include <iostream>

#include "magnet.h"
#include "spectrometer.h"

using std::ofstream;

//Energy in MeV, divergence in mrad
void placeElectron(Particle& par, double energy, double divergence)
{
    par.q = -1;
    double gamma = energy*1e6*ee/(me*cc*cc);
    double beta = sqrt(1-1/(gamma*gamma));
    par.m = gamma * me;
    par.vx = beta * cc * cos(divergence/1000);
    par.vy = beta * cc * sin(divergence/1000);
    par.x = 0;
    par.y = 0;
}
    

int main(int argc, char** argv)
{
    const char sep = ' ';
    Spectrometer spec;

    std::ifstream mag_data ("magnet2.txt");
    spec.init(mag_data);
    mag_data.close();

    Particle test;
    int result, div;

    for (div=-4;div<-4;div++) {
        placeElectron(test, 140, div);
        result = spec.run(test);
        std::cout << 300 << sep << div << sep << test.x << sep << test.y << std::endl;
    }

    div = 0;
    for (double En=290;En<=310;En+=0.1) {
        placeElectron(test, En, 0);
        result = spec.run(test);
        std::cout << En << sep << div << sep << test.x << sep << test.y << std::endl;
    }

  
    return 0;
}
