#include "PCA.h"
#include "../../Preprocessing/Preprocess.h"

#include <gnuplot-iostream.h>



void plotd (const Mat& X, Veci y)
{
    LabelEncoder<int> lenc;
    y = lenc.fitTransform(y);

    int M = X.rows(), N = X.cols(), K = lenc.numClasses;

    std::vector<std::vector<std::pair<double, double>>> vplot2d(K);
    std::vector<std::vector<std::tuple<double, double, double>>> vplot3d(K);

    for(int i = 0; i < M; ++i)
    {
        if(N == 2)
            vplot2d[y(i)].emplace_back(X(i, 0), X(i, 1));

        else
            vplot3d[y(i)].emplace_back(X(i, 0), X(i, 1), X(i, 2));            
    }

    Gnuplot gp;

    if(N == 2)
        gp << "plot";
    
    else
        gp << "splot";
    
         
    gp << " '-' with points";

    for(int i = 1; i < K; ++i)
        gp << ", '-' with points";

    gp << "\n";
    
    if(N == 2)
        for(const auto& v : vplot2d)
            gp.send1d(v);

    else
        for(const auto& v : vplot3d)
            gp.send1d(v);


    cin.get();
}





int main ()
{    
    //auto [X, y] = pickTarget(readMat("../../Data/Wine.txt", ','));
    
    auto [X, y] = pickTarget(readMat("../../Data/mushroom.txt", ','), 0);

    //auto [X, y] = pickTarget(readMat("../../Data/car.txt", ','), 1);

    //auto [X, y] = pickTarget(readMat("../../Data/abalone.txt", ','), 1);


    OneHotEncoding ohe;

    X = ohe.fitTransform(X);



    Standardize stdz;
    
    X = stdz.fitTransform(X);



    PCA pca(2);

    X = pca.fitTransform(X);


    plotd(X, y);




    return 0;
}