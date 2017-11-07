#include "RLSC.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Preprocessing/ModelSelection.h"

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


    Mat X = readMat("../../Data/Wine.txt", ',');

    //X = X.block(0, 0, 130, X.cols());

    Veci y = X.col(0).cast<int>();

    X = X.block(0, 1, X.rows(), X.cols()-1);



    // LabelEncoder lenc;

    // y = lenc.fitTransform(y, {-1, 1});


    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.5, 0);


    Standardize st;

    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);



    //RLSC<LinearKernel> rlsc(1e-1);
    // poly::Classifier<>* cls = new poly::RLSC<LinearKernel>(1e-1);
    // poly::Classifier<>& rlsc = *cls;
    

    //OVA<poly::RLSC<LinearKernel>> rlsc(1e-1);
    poly::Classifier<>* cls = new OVA<poly::RLSC<LinearKernel>>(1e-1);
    poly::Classifier<>& rlsc = *cls;


    // vector<double> alphas;

    // for(double x = -10; x <= 0; x += 0.5)
    //     alphas.push_back(pow(10, x));

    // auto gs = makeGridsearchCV(rlsc, make_tuple(make_pair([](auto& cls, double a){ cls.alpha = a; }, alphas)),
    //                                             //make_pair([](auto& cls, double g){ cls.kernel.gamma = g; }, alphas)),
    //                                             3, Accuracy());

    // gs.fit(X_train, y_train);

    // rlsc = gs.bestEstimator;


    Veci y_pred;

    double runTime = benchmark([&]
    {
        rlsc.fit(X_train, y_train);
    });

    y_pred = rlsc.predict(X_test);
    


    //db(rlsc.w.transpose(), "         ", rlsc.alpha, "\n");

    db(y_test.transpose(), "\n", y_pred.transpose(), "\n");

    db((y_test.array() == y_pred.array()).cast<double>().sum() / y_test.rows(), "\n");

    db(runTime);



    return 0;
}