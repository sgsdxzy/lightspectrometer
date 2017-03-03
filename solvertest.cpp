#include <iostream>
#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "edpsolver.h"

using namespace std;

int main(int argc, char** argv)
{
    EDPSolver front;

    ifstream iside("data_side.txt");
    ifstream ifront("data_front.txt");
    boost::archive::text_iarchive iaside(iside);
    boost::archive::text_iarchive iafront(ifront);
    
    iafront >> front;
    front.init();
    iside.close();
    ifront.close();

    for (double P=0.08;P<0.09;P+=0.0001) {
        cout << front.getE(P, 0) << endl;
    }

    return 0;
}
