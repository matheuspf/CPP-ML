#include "IncrementalFitting.h"

#include "../Preprocessing/Preprocess.h"

#include "../Preprocessing/ModelSelection.h"

#include "../Optimization/LineSearch/Backtracking/Backtracking.h"

#include "../Optimization/CG/CG.h"

#include "../Optimization/BFGS/BFGS.h"

#include "../Optimization/Newton/Newton.h"



int main ()
{
    Mat X = readMat("../Data/Wine.txt", ',');

    X = X.block(0, 0, 130, X.cols()-1);

    Veci y = X.col(0).cast<int>();

    X = X.block(0, 1, X.rows(), X.cols()-1);

    //X = polyExpansion(X, 2, true);

    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.5, 1);


    Standardize st;

    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    using Optimizer = Newton<StrongWolfe, CholeskyIdentity>;
    //using Optimizer = BFGS<StrongWolfe, BFGS_Diagonal>;

    Optimizer opt(StrongWolfe(1.0, 1e-2));

    opt.maxIterations = 10;
    //opt.gTol = 1e-3;


    // using Optimizer = CG<HS, StrongWolfe>;

    // Optimizer opt(StrongWolfe(1.0, 1e-2));

    // opt.maxIterations = 100;
    // opt.gTol = 1e-3;


    IncrementalFitting<RBFBase, Optimizer> incf(opt);


    // double runTime = benchmark([&]
    // {
        incf.fit(X_train, y_train);
    // });

    Veci y_pred = incf.predict(X_test);



    // vector<double> alphas;

    // for(double x = -5; x <= 5; x += 0.5)
    //     alphas.pb(pow(10, x));

    // auto gs = makeGridsearchCV(incf, make_pair([](auto& cls, double a){ cls.alpha = a; }, alphas), 5, Accuracy());

    // gs.fit(X_train, y_train);

    // incf = gs.bestEstimator;

    // incf.fit(X_train, y_train);

    // Veci y_pred = incf.predict(X_test);
    


    //db(incf.W, "\n\n", incf.intercept.transpose(), "\n\n", incf.alpha, "\n\n\n");

    db(y_test.transpose(), "\n", y_pred.transpose(), "\n");

    db((y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n");

    //db(runTime);



    return 0;
}