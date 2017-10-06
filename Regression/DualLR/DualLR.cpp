#include "DualLR.h"

#include "../../Preprocessing/Preprocess.h"
#include "../../Preprocessing/ModelSelection.h"


int main ()
{
    Mat X = readMat("data.txt", '\t');
    
    Vec y = X.col(X.cols()-1);

    X.conservativeResize(Eigen::NoChange, X.cols()-1);
    
    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 0);

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    // DualLR<> dlr;

    // dlr.fit(X_train, y_train);

    // db((y_test - dlr.predict(X_test)).squaredNorm() / y_test.rows(), "\n");

    

    DualLR<RBFKernel> dlr;

    vector<double> vals;

    for(double x = -5.0; x <= 5.0; x += 1.0)
        vals.push_back(pow(10, x));


    auto gs = makeGridsearchCV(dlr, make_tuple(make_pair([](auto& est, double g){ est.kernel.gamma = g; }, vals),
                                               make_pair([](auto& est, double a){ est.alpha = a; }, vals)), 10);
    
    gs.fit(X_train, y_train);

    dlr = gs.bestEstimator;

    dlr.fit(X_train, y_train);

    db((y_test - dlr.predict(X_test)).squaredNorm() / y_test.rows(), "\n");
    db(dlr.alpha, "     ", dlr.kernel.gamma);


    //8.71799    0.001       0.1

    return 0;
}