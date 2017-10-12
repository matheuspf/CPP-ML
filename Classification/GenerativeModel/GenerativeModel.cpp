#include "GenerativeModel.h"
#include <gnuplot-iostream.h>

#include "../../Distributions/Student/Student.h"


struct Plotting
{
    static constexpr const char* surface = "set pm3d\n"
                                           "set dgrid3d 30,30\n"
                                           "splot '-' u 1:2:3 with lines\n";
                             
    static constexpr const char* colorMap = "set palette rgb 7,5,15\n"
                                            "plot '-' using 0:2:3:xticlabels(1) w points lc palette notitle\n";
};


template <class F>
void plotSurface (F f, string cmd, double l = -5.0, double u = 5.0, int pts = 200)
{
    double d = (u - l) / pts;

    vector<tuple<double, double, double>> plot;
    
    FOR(i, pts) FOR(j, pts)
    {
        Vec x(2); x << l + i * d, l + j * d;

        plot.emplace_back(l + i * d, l + j * d, f(x));
    }


    Gnuplot gp;

    gp << cmd;

    gp.send1d(plot);

    cin.get();
}



void test1 ()
{
    int N = 2, M = 1e3;

    Mat X(M, N);
    Veci y(M);

    // Gaussian g1(Vec::Constant(N, 1.5), 0.5);
    // Gaussian g2(Vec::Constant(N, -1.5), 2.0);
    Student g1(Vec::Constant(N, 1.5), 1.0, 2.0);
    Student g2(Vec::Constant(N, -1.5), 0.5, 2.0);


    FOR(i, M)
    {
        X.row(i) = g1();
        y(i) = 0;
    }

    FORR(i, M/2, M)
    {
        X.row(i) = g2();
        y(i) = 1;
    }


    //plotSurface([&](const Vec& x){ return g1(x) + g2(x); }, Plotting::surface, -5.0, 5.0, 500); exit(0);


    GenerativeModel<Student> genModel;

    genModel.fit(X, y);


    plotSurface([&](const Vec& x){ return genModel.predict(x); }, Plotting::colorMap, -5.0, 5.0, 200);
}


int main ()
{
    test1();


    return 0;
}