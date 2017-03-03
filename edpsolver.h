// Energy-Divergence-Position solver, read two as input and get the third from data
// Read energy, divergence and get position, read position, divergence and get energy.
#ifndef LIGHT_EDP_SOLVER
#define LIGHT_EDP_SOLVER

#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

using std::vector;

class EDPSolver
{
public :
    double div_min, div_max, en_min, en_max, pos_min, pos_max; // all ranges are [min, max)
    double ddiv, den;
    int div_size, en_size, pos_size, div_0_index;
    vector<double> divergences, energies, times, positions;

    void init();

    double position(int en_index, int div_index) const;
    
    double getP(double E, double D) const;
    double getE(double P, double D) const;
    double getT(double E) const;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & divergences;
        ar & energies;
        ar & times;
        ar & positions;
    }    
};



#endif //LIGHT_EDP_SOLVER
