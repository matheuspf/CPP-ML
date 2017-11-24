#include "FishersLDA.h"
#include "../../Preprocessing/Preprocess.h"

#include <gnuplot-iostream.h>


void plot2d (const Mat& X, Veci y)
{
    LabelEncoder lenc;
    y = lenc.fitTransform(y);

    int M = X.rows(), N = X.cols(), K = lenc.numClasses;

    std::vector<std::vector<std::tuple<double, double, double>>> vplot(K);

    for(int i = 0; i < M; ++i)
        vplot[y(i)].emplace_back(X(i, 0), X(i, 1), X(i, 2));


    Gnuplot gp;

    gp << "splot '-' with points";

    for(int i = 1; i < K; ++i)
        gp << ", '-' with points";

    gp << "\n";

    for(const auto& v : vplot)
        gp.send1d(v);


    cin.get();
}


int main ()
{    
    // auto [X, y] = pickTarget(readMat("../../Data/Wine.txt", ','));

    //auto [X, y] = pickTarget(readMat("../../Data/car.txt", ','), 1);

    auto [X, y] = pickTarget(readMat("../../Data/abalone.txt", ','), 1);


    OneHotEncoding ohe({0});

    X = ohe.fitTransform(X);



    // Standardize stdz;
    
    // X = stdz.fitTransform(X);



    FishersLDAMultiClass flda(3);

    X = flda.fitTransform(X, y);


    plot2d(X, y);




    return 0;
}