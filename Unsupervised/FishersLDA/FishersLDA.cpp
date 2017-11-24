#include "FishersLDA.h"
#include "../../Preprocessing/Preprocess.h"

#include <gnuplot-iostream.h>


void plot2d (const Mat& X, Veci y)
{
    LabelEncoder lenc;
    y = lenc.fitTransform(y);

    int M = X.rows(), N = X.cols(), K = lenc.numClasses;

    std::vector<std::vector<std::pair<double, double>>> vplot(K);

    for(int i = 0; i < M; ++i)
        vplot[y(i)].emplace_back(X(i, 0), X(i, 1));


    Gnuplot gp;

    gp << "plot '-' with points";

    for(int i = 1; i < K; ++i)
        gp << ", '-' with points";

    gp << "\n";

    for(const auto& v : vplot)
        gp.send1d(v);


    cin.get();
}


int main ()
{
    Mat X = readMat("../../Data/Wine.txt", ',');

    //X = X.block(0, 0, 130, X.cols()-1);

    Veci y = X.col(0).cast<int>();

    X = X.block(0, 1, X.rows(), X.cols()-1);


    FishersLDAMultiClass flda;

    X = flda.fitTransform(X, y);

    plot2d(X, y);





    return 0;
}