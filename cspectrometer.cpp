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
    t = 0;
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

/*
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
}*/

void Spectrometer::initdt(double dt_multiplier)
{
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

void Spectrometer::getSolverData(double *Ens, int en_size, double *divergences, int div_size, double *x_pos, double *y_pos, int *results, double *times, int central_index) const
{
    Particle test;
    int result;
    double En, div;
    int i, j;

    #pragma omp parallel for schedule(dynamic) private(En, div, test, i ,j)
    for (i = 0; i < en_size; i++) {
        En = Ens[i];
        for (j = 0; j < div_size; j++) {
            div = divergences[j];
            test.setElectron(En, div);
            result = run(test);
            results[i*div_size+j] = result;
            x_pos[i*div_size+j] = test.x;
            y_pos[i*div_size+j] = test.y;
            if (j == central_index) times[i] = test.t;
        }
    }

}


double EDPSolver::position(int en_index, int div_index) const
{
    int index = div_size * en_index + div_index;
    //if ((index<0) || (index>=pos_size)) return 0;
    return positions[index];
}

double EDPSolver::getP(double E, double D) const
{
    int en_right = distance(energies, lower_bound(energies, energies+en_size, E));
    int en_left = en_right - 1;
    int div_right = distance(divergences, lower_bound(divergences, divergences+div_size, D));
    int div_left = div_right - 1;
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
    int div_right = distance(divergences, lower_bound(divergences, divergences+div_size, D));
    int div_left = div_right - 1;
    double r = D - divergences[div_left];
    double s = divergences[div_right] - D;
    double dy = r+s;

    vector<double> P_int(en_size);
    for (int i=0;i<en_size;i++) {
        P_int[i] = (position(i, div_left)*s + position(i, div_right)*r)/dy;
    }

    int pos_right = distance(lower_bound(P_int.rbegin(), P_int.rend(), P), P_int.rend());
    int pos_left = pos_right - 1;
        std::cout << P << ' ' << pos_right << ' ' << *lower_bound(P_int.begin(), P_int.end(), P) << std::endl;
    double p = P - P_int[pos_left];
    double q = P_int[pos_right] - P;
    double dx = p+q;

    return (energies[pos_left]*q+energies[pos_right]*p)/dx;
}

double EDPSolver::getT(double E) const
{
    int en_right = distance(energies, lower_bound(energies, energies+en_size, E));
    int en_left = en_right - 1;
    double pp = times[en_left];
    double qq = times[en_right];
    double p = E - energies[en_left];
    double q = energies[en_right] - E;
    double dx = p+q;

    return (p*qq+q*pp)/dx;
}
