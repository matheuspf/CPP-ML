#include "GenerativeModel.h"
#include <gnuplot-iostream.h>

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"

#include "../../Distributions/Student/Student.h"

#include "../../Distributions/KernelDensity/KernelDensity.h"

#include "../../Distributions/KNN/KNN.h"



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



void test2 ()
{
    // auto [X, y] = pickTarget(readMat("../../Data/mushroom.txt", ','), 0);
    // // auto [X, y] = pickTarget(readMat("../../Data/abalone.txt", ','), 0);

    // OneHotEncoding ohe;
    // X = ohe.fitTransform(X);


    auto [X, y] = pickTarget(readMat("../../Data/Wine.txt", ','), 0);


    // Mat X = readMat("../../Data/Wine.txt", ',');

    // X = X.block(0, 0, 130, X.cols()-1);

    // Veci y = X.col(0).cast<int>();

    // X = X.block(0, 1, X.rows(), X.cols()-1);



    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.5, 1);


    Standardize stz;
    X_train = stz.fitTransform(X_train);
    X_test = stz.transform(X_test);
    


    // GenerativeModel<KernelDensity> genModel;

    // genModel.fit(X_train, y_train, KernelDensity(5.0));

    
    GenerativeModel<KNN> genModel;

    
    //genModel.fit(X_train, y_train);


    vector<int> Ks;

    for(int k = 1; k <= 10; ++k)
        Ks.pb(k);

    auto gs = makeGridsearchCV(genModel, make_pair([](auto& cls, int k){ cls.baseConditional.K = k; }, Ks), 5, Accuracy());
    
    gs.fit(X_train, y_train);

    genModel = gs.bestEstimator;

    genModel.fit(X_train, y_train);

    


    Veci y_pred_train = genModel.predict(X_train);

    Veci y_pred = genModel.predict(X_test);

    
    //db("Sigma:  ", genModel.baseConditional.sigma, "\n");
    db("K:   ", genModel.baseConditional.K, "\n");

    db("Train:  ", (y_pred_train.array() == y_train.array()).cast<double>().sum() / y_train.rows());
    db("Test:  ", (y_pred.array() == y_test.array()).cast<double>().sum() / y_test.rows());
}




int main ()
{
    test2();


    return 0;
}