#include <cmath>
#include <limits>
#include "cspectrometer.h"

void Particle::setElectron(double energy, double divergence)
{
    q = -1;
    double gamma = energy*1e6*ee/(me*cc*cc);
    double beta = sqrt(1-1/(gamma*gamma));
    m = gamma * me;
    vx = beta * cc * cos(divergence/1000);
    vy = beta * cc * sin(divergence/1000);
    x = 0;
    y = 0;
}


double Magnet::accessB(int x, int y) const
{
    if ((x<0) || (x>=x_grid) || (y<0) || (y>=y_grid)) {
        return 0;
    }
    return B[x_grid*y+x];
}

double Magnet::getB(double x, double y) const
{
    int x_left = floor(x/x_delta);
    int y_left = floor(y/y_delta);
    double pr = accessB(x_left, y_left);
    double qr = accessB(x_left+1, y_left);
    double ps = accessB(x_left, y_left+1);
    double qs = accessB(x_left+1, y_left+1);
    double p = x - x_left*x_delta;
    double q = (x_left+1)*x_delta - x;
    double r = y - y_left*y_delta;
    double s = (y_left+1)*y_delta - y;

    return (r*p*qs+r*q*ps+s*p*qr+s*q*pr)/(x_delta*y_delta);
}

ostream& operator<<(ostream& out, const Magnet& mag)
{
    const char sep = ' ';
    out<<"#Light-magnet data file"<<endl;
    out<<"#Version: 0.1"<<endl;
    out<<mag.x_grid<<sep<<mag.y_grid<<endl;
    out<<mag.x_delta<<sep<<mag.y_delta<<endl;
    for (int i=0;i<mag.y_grid;i++) {
        for (int j=0;j<mag.x_grid;j++) {
            out<<mag.B[mag.x_grid*i+j]<<sep;
        }
        out<<endl;
    }
    return out;
}

istream& operator>>(istream& in, Magnet& mag)
{
    //skip first 2 lines
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    in>>mag.x_grid>>mag.y_grid;
    in>>mag.x_delta>>mag.y_delta;
    mag.B.clear();
    mag.B.resize(mag.x_grid*mag.y_grid);
    for (int i=0;i<mag.y_grid;i++) {
        for (int j=0;j<mag.x_grid;j++) {
            in>>mag.B[mag.x_grid*i+j];
        }
    }
    return in;
}

void Spectrometer::initdt(double dt_multiplier)
{
    //x_offset = 140e-3;
    //y_offset = -110e-3;
    dt = dt_multiplier/sqrt((cc/mag.x_delta)*(cc/mag.x_delta)+(cc/mag.y_delta)*(cc/mag.y_delta));
}

int Spectrometer::run(Particle& par) const
{
    double B, t, s, vpx, vpy;
    int result = 0;

    while (par.t <= maxtime) {
        B = mag.getB(par.x-x_offset, par.y-y_offset);
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

int Spectrometer::condition(Particle& par) const
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

void Spectrometer::getSolverData(vector<double> &Ens, vector<double> &divergences, EDPSolver& side, EDPSolver& front) const
{
    Particle test;
    double time;
    double En;
    vector<int> results;
    vector<double> x_pos, y_pos;
    
    side.clear();
    front.clear();
    side.divergences = divergences;
    front.divergences = divergences;

    #pragma omp parallel for ordered schedule(dynamic) private(En, x_pos, y_pos, results, test, time)
    for (auto it = Ens.begin(); it < Ens.end(); it++) {
        En = *it;
        x_pos.clear();
        y_pos.clear();
        results.clear();
        for (double &div : divergences) {
            test.setElectron(En, div);
            results.push_back(run(test));
            x_pos.push_back(test.x);
            y_pos.push_back(test.y);
            if (div == 0) time = test.t;
        }
        #pragma omp ordered
        {
            if (std::all_of(results.begin(), results.end(), [](int i){return i==1;})) {
                //hit side
                side.energies.push_back(En);
                side.times.push_back(time);
                side.positions.insert(side.positions.end(), x_pos.begin(), x_pos.end());
            }
            if (std::all_of(results.begin(), results.end(), [](int i){return i==2;})) {
                //hit front
                front.energies.push_back(En);
                front.times.push_back(time);
                front.positions.insert(front.positions.end(), y_pos.begin(), y_pos.end());
            }
        }
    }

}

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

void EDPSolver::clear()
{
    divergences.clear();
    energies.clear();
    times.clear();
    positions.clear();
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
