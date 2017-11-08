#include "Stump.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"



int main ()
{
    Mat X = readMat("../../Data/Wine.txt", ',');

    X = X.block(0, 0, 130, X.cols()-1);

    Veci y = X.col(0).cast<int>();

    X = X.block(0, 1, X.rows(), X.cols()-1);


    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.5, 1);

    X_train = -X_train;



    Stump<> stump;


    double runTime = benchmark([&]
    {
        stump.fit(X_train, y_train);
    });

    Veci y_pred = stump.predict(X_test);


    db(y_test.transpose(), "\n", y_pred.transpose(), "\n");

    db((y_test.array() == y_pred.array()).cast<double>().sum() / y_pred.rows(), "\n");

    db(stump.index, "   ", stump.stump, "   ", stump.direction, "\n");
    
    db(runTime);



    return 0;
}