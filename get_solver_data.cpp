#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>

#include "magnet.h"
#include "spectrometer.h"
#include "edpsolver.h"

using std::ofstream;
using std::vector;

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
    Spectrometer spec;

    std::ifstream mag_data ("magnet2.txt");
    spec.init(mag_data);
    mag_data.close();

    Particle test;
    int result;
    double time;
    vector<double> divergences;
    EDPSolver side, front;
    vector<int> results;
    vector<double> x_pos, y_pos;
    

    ofstream oside("data_side.txt");
    ofstream ofront("data_front.txt");

    for (double div=-10;div<=10;div+=0.5) {
        divergences.push_back(div);
    }
    side.divergences = divergences;
    front.divergences = divergences;

    x_pos.resize(divergences.size());
    y_pos.resize(divergences.size());
    results.resize(divergences.size());

    for (double En=30;En<400;En+=0.1) {
        x_pos.clear();
        y_pos.clear();
        results.clear();
        for (double &div : divergences) {
            placeElectron(test, En, div);
            result = spec.run(test);
            results.push_back(result);
            x_pos.push_back(test.x);
            y_pos.push_back(test.y);
            if (div == 0) time = test.t;
        }
        if (std::all_of(results.begin(), results.end(), [](int i){return i==1;})) {
            //hit side
            side.energies.push_back(En);
            side.times.push_back(time);
            side.positions.insert(side.positions.end(), x_pos.begin(), x_pos.end());
            continue;
        }
        if (std::all_of(results.begin(), results.end(), [](int i){return i==2;})) {
            //hit front
            front.energies.push_back(En);
            front.times.push_back(time);
            front.positions.insert(front.positions.end(), y_pos.begin(), y_pos.end());
            continue;
        }
    }

    boost::archive::text_oarchive oaside(oside);
    boost::archive::text_oarchive oafront(ofront);

    oaside << side;
    oafront << front;

    oside.close();
    ofront.close();

    return 0;
}
