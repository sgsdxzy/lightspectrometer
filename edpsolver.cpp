#include <algorithm>
#include "edpsolver.h"

using std::vector;
using std::distance;
using std::lower_bound;

void EDPSolver::init()
{
    div_size = divergences.size();
    en_size = energies.size();
    pos_size = positions.size();
    div_min = divergences.front();
    div_max = divergences.back();
    div_0_index = distance(divergences.begin(), lower_bound(divergences.begin(), divergences.end(), 0));
    en_min = energies.front();
    en_max = energies.back();
    pos_min = position(0, div_0_index);
    pos_max = position(en_size-1, div_0_index);
}


double EDPSolver::position(int en_index, int div_index) const
{
    int index = div_size * en_index + div_index;
    //if ((index<0) || (index>=pos_size)) return 0;
    return positions[index];
}

double EDPSolver::getP(double E, double D) const
{
    int en_left = distance(energies.begin(), lower_bound(energies.begin(), energies.end(), E));
    int en_right = en_left + 1;
    int div_left = distance(divergences.begin(), lower_bound(divergences.begin(), divergences.end(), D));
    int div_right = div_left + 1;
    double pr = position(en_left, div_left);
    double qr = position(en_right, div_left);
    double ps = position(en_left, div_right);
    double qs = position(en_right, div_right);
    double p = E - energies[en_left];
    double q = energies[en_right] - E;
    double r = D - divergences[div_left];
    double s = divergences[div_right] - D;
    double dx = p+q;
    double dy = r+s;
    
    return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(dx*dy);
}

double EDPSolver::getE(double P, double D) const
{
    int div_left = distance(divergences.begin(), lower_bound(divergences.begin(), divergences.end(), D));
    int div_right = div_left + 1;
    double r = D - divergences[div_left];
    double s = divergences[div_right] - D;
    double dy = r+s;

    vector<double> P_int(en_size);
    for (int i=0;i<en_size;i++) {
        P_int[i] = (position(i, div_left)*s + position(i, div_right)*r)/dy;
    }
    
    int pos_left = distance(P_int.begin(), lower_bound(P_int.begin(), P_int.end(), P));
    int pos_right = pos_left + 1;
    double p = P - P_int[pos_left];
    double q = P_int[pos_right] - P;
    double dx = p+q;

    return (energies[pos_left]*q+energies[pos_right]*p)/dx;
}

double EDPSolver::getT(double E) const
{
    int en_left = distance(energies.begin(), lower_bound(energies.begin(), energies.end(), E));
    int en_right = en_left + 1;
    double pp = times[en_left];
    double qq = times[en_right];
    double p = E - energies[en_left];
    double q = energies[en_right] - E;
    double dx = p+q;

    return (p*qq+q*pp)/dx;
}
