#include "BayesianLogisticRegression.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"

#include "../../Unsupervised/PCA/PCA.h"



int main ()
{
    // Mat W = readMat("../../Data/Wine.txt", ',');

    // Mat Z = W.block(0, 0, 130, W.cols());

    // Veci y = Z.col(0).cast<int>();

    // Mat X = Z.block(0, 1, Z.rows(), Z.cols()-1);


    // X = polyExpansion(X, 2, true);



    auto [X, y] = pickTarget(readMat("../../Data/mushroom.txt", ','), 0);

    OneHotEncoding ohe;

    X = ohe.fitTransform(X);

    //X = polyExpansion(X, 2, true);


    // PCA pca(5);

    // X = pca.fitTransform(X);



    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.9, 1);



    BayesianLogisticRegression<> blr;

    double runtime = benchmark([&]
    {
        blr.fit(X_train, y_train, 1e-6, 1e-6, 50, 1);
    });

    Veci y_pred_train = blr.predict(X_train);
    Veci y_pred = blr.predict(X_test);



    db("a:  ", blr.alpha, "\n");
    //db("w:  ", blr.w.transpose(), "\n");


    db("Train error:    ", (y_train.array() == y_pred_train.array()).cast<double>().sum() / y_train.rows(), "\n");

    db("Test error:     ", (y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n\n");

    db("Runtime:   ", runtime, "\n");



    return 0;
}