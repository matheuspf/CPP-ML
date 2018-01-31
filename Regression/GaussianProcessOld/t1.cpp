#include "GaussianProcess.h"

#include "../../Preprocessing/Preprocess.h"


auto readData (string str)
{
    ifstream in(str);

    int N = 0, M = 0;

    vector<double> data; data.reserve(1e4);


    while(getline(in, str))
    {
        stringstream ss(str);

        int cnt = 0;
        double aux;

        while(ss >> aux) data.push_back(aux), cnt++;

        if(!N) N = cnt;
        ++M;
    }

    Mat X(M, N-1);
    Vec y(M);
    

    FOR(i, data.size())
    {
        if(i % N == 0) y(i / N) = data[i];

        else X(i / N, (i % N)-1) = data[i];
    }

    return make_tuple(X, y);
}




int main ()
{
    auto [X, y] = readData("reg.txt");

    OneHotEncoding ohe({0, 5, 6});
    X = ohe.fitTransform(X);


    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 1);

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    GaussianProcess gp;

    gp.fit(X_train, y_train);

    Vec y_pred(y_test.rows());
    
    FOR(i, y_test.rows())
        y_pred(i) = gp(Vec(X_test.row(i)));

    db((y_test - y_pred).norm());



    // vector<double> ww = {1e-3, 5e-3, 9e-3, 1e-2, 1.1e-2, 5e-2};

    // double best = 1e20, a;


    // for(double w : ww)
    // {
    //     GaussianProcess gp;

    //     gp.sigma = 1.0;
    //     gp.sigmaPrior = 1.0;
    //     gp.kernel.gamma = w;

    //     gp.fit(X_train, y_train);

    //     Vec y_pred(y_test.rows());
        
    //     FOR(i, y_test.rows())
    //         y_pred(i) = gp(Vec(X_test.row(i)));
    
    //     double val = (y_test - y_pred).norm();

    //     if(val < best)
    //         best = val, a = w;
    // }


    // db(best, a);
    

    return 0;
}