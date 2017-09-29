#include "SparseBayesianLR.h"

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

    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 0);

    Standardize st;
    X_train = st.fitTransform(X_train);
    X_test = st.transform(X_test);


    SparseBayesianLR blr;

    blr.fit(X_train, y_train);

    Vec y_pred(y_test.rows());
    
    FOR(i, y_test.rows())
        y_pred(i) = blr(Vec(X_test.row(i)));

    db((y_test - y_pred).norm());
    


    // double best = 1e20, a, b;

    // vector<double> ww = {1e-2, 1e-1, 0.5, 1.0, 5.0, 7.0, 10.0, 15.0, 20.0};

    // for(auto x : ww) for(auto z : ww)
    // {
    //     BayesianLR blr;

    //     blr.learn(X_train, y_train, x, z);

    //     Vec y_pred(y_test.rows());
        
    //     FOR(i, y_test.rows())
    //         y_pred(i) = blr(Vec(X_test.row(i)));
    
    //     if((y_test - y_pred).norm() < best)
    //         best = (y_test - y_pred).norm(), a = x, b = z;
    // }

    // db(best, a, b);


    return 0;
}