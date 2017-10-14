#include "LogisticRegression.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Optimization/LineSearch/Backtracking/Backtracking.h"

#include "../../Optimization/CG/CG.h"

#include "../../Optimization/BFGS/BFGS.h"

#include "../OVA/OVA.h"



int main ()
{
    // Mat X = readMat("../../Data/Iris.txt", ',');

    // X = X.block(50, 0, X.rows()-50, X.cols());

    // Veci y = X.col(X.cols()-1).cast<int>();

    // X.conservativeResize(Eigen::NoChange, X.cols()-1);

    // transform(begin(y), end(y), begin(y), [](int x){ return x == 2 ? 0 : x; });

    Mat X = readMat("../../Data/Wine.txt", ',');

    Veci y = X.col(0).cast<int>();

    X = X.block(0, 1, X.rows(), X.cols()-1);


    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.5, 1);


    Standardize st;

    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    //using Optimizer = Newton<StrongWolfe, SimplyInvert<Eigen::LLT>>;
    //using Optimizer = CG<HS, Backtracking>;
    using Optimizer = BFGS<StrongWolfe, BFGS_Diagonal>;

    Optimizer opt(StrongWolfe(1.0, 1e-2));

    opt.maxIter = 10;
    opt.gTol = 1e-3;

    OVA<LogisticRegression<Optimizer>> lr(1e-1, opt);




    Veci y_pred;


    double runTime = benchmark([&]
    {
        lr.fit(X_train, y_train);
    });

    y_pred = lr.predict(X_test);
    


    //db(lr.w.transpose(), "     ", lr.intercept, "\n");

    db(y_test.transpose(), "\n", y_pred.transpose(), "\n");

    db((y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n");

    db(runTime);



    return 0;
}