#include "ProbitRegression.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"

#include "../../Optimization/LineSearch/Backtracking/Backtracking.h"

#include "../../Optimization/CG/CG.h"

#include "../../Optimization/BFGS/BFGS.h"

#include "../../Unsupervised/PCA/PCA.h"



int main ()
{
    Mat Z = readMat("../../Data/Wine.txt", ',');

    Mat Y = Z.block(0, 0, 130, Z.cols());

    Veci y = Y.col(0).cast<int>();

    Mat X = Z.block(0, 1, 130, Z.cols()-1);


    //auto [X, y] = pickTarget(readMat("../../Data/Wine.txt", ','), 0);
    //auto [X, y] = pickTarget(readMat("../../Data/mushroom.txt", ','), 0);
    //auto [X, y] = pickTarget(readMat("../../Data/abalone.txt", ','), 1);
    

    // OneHotEncoding ohe;

    // X = ohe.fitTransform(X);


    // PCA pca(20);

    // X = pca.fitTransform(X);



    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 1);
    // Mat X_train = X;
    // Mat X_test = X;
    // Veci y_train = y;
    // Veci y_test = y;


    // Standardize st;

    // X_train = st.fitTransform(X_train);
    // X_test = st.transform(X_test);


    //using Optimizer = Newton<StrongWolfe, SimplyInvert>;
    using Optimizer = BFGS<StrongWolfe, BFGS_Diagonal>;

    Optimizer opt(StrongWolfe(1.0, 1e-4));

    opt.maxIter = 20;
    opt.gTol = 1e-6;


    // using Optimizer = CG<HS, Backtracking>;

    // Optimizer opt;

    // opt.maxIterations = 20;
    // opt.xTol = 1e-6;


    ProbitRegression<L2, Optimizer> lr(1e-7, opt);


    double runTime = benchmark([&]
    {
        lr.fit(X_train, y_train);
    });

    Veci y_pred = lr.predict(X_test);

    Veci y_pred_train = lr.predict(X_train);



    // ProbitRegression<L2, Optimizer> lr(opt);

    // vector<double> alphas;

    // for(double x = -10; x <= 10; x += 0.2)
    //     alphas.pb(pow(10, x));

    // auto gs = makeGridsearchCV(lr, make_pair([](auto& cls, double a){ cls.alpha = a; }, alphas), 10, Accuracy());

    // gs.fit(X_train, y_train);

    // lr = gs.bestEstimator;

    // lr.fit(X_train, y_train);


    // Veci y_pred = lr.predict(X_test);

    // Veci y_pred_train = lr.predict(X_train);
    

    

    //db("\n\n\n", lr.alpha, "\n");

    db(y_test.transpose().head(min(int(y_test.size()), 20)), "\n\n", y_pred.transpose().head(min(int(y_pred.size()), 20)), "\n\n\n");


    db("Train error:    ", (y_train.array() == y_pred_train.array()).cast<double>().sum() / y_train.rows(), "\n");

    db("Test error:     ", (y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n");

    db(runTime);



    return 0;
}