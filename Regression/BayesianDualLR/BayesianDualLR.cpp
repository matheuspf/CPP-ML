#include "BayesianDualLR.h"

#include "../../Preprocessing/Preprocess.h"
#include "../../Preprocessing/ModelSelection.h"


int main ()
{
    Mat X = readMat("../../Data/Airfoil.txt", '\t');
    
    Vec y = X.col(X.cols()-1);

    X.conservativeResize(Eigen::NoChange, X.cols()-1);
    
    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 270001);

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    BayesianDualLR<> bdlr;

    //bdlr.kernel.gamma = 1e-2;

    bdlr.fit(X_train, y_train);

    db("\n", (y_test - bdlr.predict(X_test)).squaredNorm() / y_test.rows(), "\n");
    
    // 9.64423           0.001       1       0.1



    // BayesianDualLR<RBFKernel> dlr;
    
    // vector<double> alphas, betas, gammas;

    // for(double x = -2.0; x <= 2.0; x += 1.0)
    //     gammas.push_back(pow(10, x));

    // for(double x = -3.0; x <= 1.0; x += 1.0)
    //     alphas.push_back(pow(10, x));

    // for(double x = -5.0; x <= 0.0; x += 1.0)
    //     betas.push_back(pow(10, x));


    // auto gs = makeGridsearchCV(dlr, make_tuple(make_pair([](auto& est, double g){ est.kernel.gamma = g; }, gammas),
    //                                            make_pair([](auto& est, double a){ est.alpha = a; },        alphas),
    //                                            make_pair([](auto& est, double b){ est.beta = b; },         betas )),
    //                                            5);
    
    // gs.fit(X_train, y_train);

    // dlr = gs.bestEstimator;

    // dlr.fit(X_train, y_train);

    // db((y_test - dlr.predict(X_test)).squaredNorm() / y_test.rows(), "\n");
    // db(dlr.alpha, "     ", dlr.beta, "     ", dlr.kernel.gamma);
    

    

    return 0;
}