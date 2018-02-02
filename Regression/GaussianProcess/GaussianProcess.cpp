#include "GaussianProcess.h"

#include "../../Preprocessing/ModelSelection.h"
#include "../../Preprocessing/Preprocess.h"



auto gridSearch (const Mat& X_train, const Vec& y_train)
{
    GaussianProcess<> gp;
    
    std::vector<double> beta, alpha;

    for(double x = -4; x <= 0; x += 0.5)
        beta.push_back(pow(10.0, x));

    for(double x = -4; x <= 4; x += 1.0)
        alpha.push_back(pow(10.0, x));

    auto gs = makeGridsearchCV(gp, make_tuple(make_pair([](auto& cls, double b){ cls.beta = b; }, beta),
                                              make_pair([](auto& cls, double a){ cls.kernel.phi(1) = a; }, alpha)),
                                              5, SquaredError());

    gs.fit(X_train, y_train);

    return gs.bestEstimator;
}


int main ()
{
    auto [X, y] = pickTarget<double>(readMat("../../Data/Airfoil.txt", '\t'), 1);

    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 270001);


    
    GaussianProcess<> gp(1e0);


    gp.fit(X_train, y_train); 

    Vec y_pred = gp.predict(X_test);
    Vec y_pred_train = gp.predict(X_train);
    

    db("Train error:  ", (y_train - y_pred_train).squaredNorm() / y_train.rows(), "\n");
    db("Test error:  ", (y_test - y_pred).squaredNorm() / y_test.rows(), "\n");
    

    db("\n\n", gp.beta, "   ", gp.kernel.phi(1), "\n");





    return 0;
}