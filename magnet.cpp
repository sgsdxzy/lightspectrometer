#include <cmath>
#include <limits>
#include "magnet.h"

Magnet::~Magnet()
{
    if (allocated) {
        delete B;
    }
}


double Magnet::accessB(int x, int y)
{
    if ((x<0) || (x>=x_grid) || (y<0) || (y>=y_grid)) {
        return 0;
    }
    return B[x_grid*y+x];
}

double Magnet::getB(double x, double y)
{
    x -= x_offset;
    y -= y_offset;
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
    if (mag.allocated) {
        for (int i=0;i<mag.y_grid;i++) {
            for (int j=0;j<mag.x_grid;j++) {
                out<<mag.B[mag.x_grid*i+j]<<sep;
            }
            out<<endl;
        }
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
    if (mag.allocated) {
        delete mag.B;
    }
    mag.B = new double[mag.x_grid*mag.y_grid];
    mag.allocated = true;
    for (int i=0;i<mag.y_grid;i++) {
        for (int j=0;j<mag.x_grid;j++) {
            in>>mag.B[mag.x_grid*i+j];
        }
    }
    return in;
}
