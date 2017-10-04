#include "../../../Modelo.h"
#include "../../../Preprocessing/Preprocess.h"
#include "../../../Preprocessing/ModelSelection.h"

#include "../../LinearRegression/LinearRegression.h"
#include "../../BayesianLR/BayesianLR.h"





int main ()
{
    Mat X = readMat("data.txt", '\t');

    Vec y = X.col(X.cols()-1);

    X.conservativeResize(Eigen::NoChange, X.cols()-1);
    
    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.2, 270001);
    

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    BayesianLR reg;
    //LinearRegression reg;

    reg.fit(X_train, y_train);

    db((y_test - reg.predict(X_test)).squaredNorm() / y_test.rows(), "\n");




    return 0;
}