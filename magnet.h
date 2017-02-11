#ifndef LIGHT_MAGNET
#define LIGHT_MAGNET

#include <iostream>

using std::ostream;
using std::istream;
using std::endl;

class Magnet 
{
private:
    double *B;                      //The magnetic filed block, unit in T
    bool allocated = false;         //Whether *B is allocated

    double accessB(int x, int y);   //return B[y][x]

public:
    int x_grid, y_grid;             //The number of grids in x and y direction
    double x_delta, y_delta;        //The step of grids in x and y direction, units in SI(m)
    double x_offset = 0, y_offset = 0;      //The position of magnet (0,0) in physical space

    ~Magnet();
    double getB(double x, double y);  //Get the magnetic field at physical space (x,y)
    friend ostream& operator<<(ostream& out, const Magnet& mag);
    friend istream& operator>>(istream& in, Magnet& mag);
};


#endif //LIGHT_MAGNET
