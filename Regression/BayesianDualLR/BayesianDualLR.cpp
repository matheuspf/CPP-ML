#include "BayesianDualLR.h"

#include "../../Preprocessing/Preprocess.h"
#include "../../Preprocessing/ModelSelection.h"


int main ()
{
    Mat X = readMat("../../Data/Airfoil.txt", '\t');
    
    Vec y = X.col(X.cols()-1);

    X.conservativeResize(Eigen::NoChange, X.cols()-1);
    
    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.2, 270001);

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    BayesianDualLR<> bdlr;

    bdlr.fit(X_train, y_train);

    db((y_test - bdlr.predict(X_test)).squaredNorm() / y_test.rows(), "\n");

    

    return 0;
}