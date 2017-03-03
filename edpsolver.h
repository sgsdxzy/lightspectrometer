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
    vector<double> divergences, energies, times, positions;
       
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
