#include "LogisticRegression.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"

#include "../../Optimization/LineSearch/Backtracking/Backtracking.h"

#include "../../Optimization/CG/CG.h"

#include "../../Optimization/BFGS/BFGS.h"

#include "../../Unsupervised/PCA/PCA.h"



int main ()
{
    // Mat X = readMat("../../Data/Iris.txt", ',');

    // X = X.block(50, 0, X.rows()-50, X.cols());

    // Veci y = X.col(X.cols()-1).cast<int>();

    // X.conservativeResize(Eigen::NoChange, X.cols()-1);

    // transform(begin(y), end(y), begin(y), [](int x){ return x == 2 ? 0 : x; });

    



    auto [X, y] = pickTarget(readMat("../../Data/Wine.txt", ','), 0);
    //auto [X, y] = pickTarget(readMat("../../Data/mushroom.txt", ','), 0);
    //auto [X, y] = pickTarget(readMat("../../Data/abalone.txt", ','), 1);
    
    //X = X.block(0, 0, 130, X.cols()-1);    


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


    // using Optimizer = Newton<StrongWolfe, SimplyInvert>;
    // //using Optimizer = BFGS<StrongWolfe, BFGS_Diagonal>;

    // Optimizer opt(StrongWolfe(1.0, 1e-4));

    // opt.maxIterations = 20;
    // opt.gTol = 1e-6;


    // using Optimizer = CG<HS, Backtracking>;

    // Optimizer opt;

    // opt.maxIterations = 20;
    // opt.gTol = 1e-6;


    //LogisticRegression<L2, Optimizer> lr(1e-1, opt, "OVA");
    // auto cls = std::make_unique<poly::LogisticRegression<L2, Optimizer>>(1e-1, opt, "OVA");
    // poly::Classifier<>& lr = *cls;



    // LogisticRegression<L2> lr("Multi", 1e-7);


    // double runTime = benchmark([&]
    // {
    //     lr.fit(X_train, y_train);
    // });

    // Veci y_pred = lr.predict(X_test);

    // Veci y_pred_train = lr.predict(X_train);



    LogisticRegression<L2> lr("Multi");

    vector<double> alphas;

    for(double x = -10; x <= 10; x += 0.2)
        alphas.pb(pow(10, x));

    auto gs = makeGridsearchCV(lr, make_pair([](auto& cls, double a){ cls.alpha = a; }, alphas), 10, Accuracy());

    gs.fit(X_train, y_train);

    lr = gs.bestEstimator;

    lr.fit(X_train, y_train);


    Veci y_pred = lr.predict(X_test);

    Veci y_pred_train = lr.predict(X_train);
    

    

    db("\n\n\n", lr.alpha, "\n");

    db(y_test.transpose().head(min(int(y_test.size()), 20)), "\n\n", y_pred.transpose().head(min(int(y_pred.size()), 20)), "\n\n\n");


    db("Train error:    ", (y_train.array() == y_pred_train.array()).cast<double>().sum() / y_train.rows(), "\n");

    db("Test error:     ", (y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n");

    //db(runTime);



    return 0;
}