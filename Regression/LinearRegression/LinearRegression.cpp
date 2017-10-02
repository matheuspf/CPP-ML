#include "LinearRegression.h"

#include "../../Preprocessing/Preprocess.h"
#include "../../Preprocessing/ModelSelection.h"


void toyTest ()
{
	Mat A(3, 2);
	Vec b(3);

	A << 1, 2, 2, 4, 3, 7;
	b << 10, 2, 1;


	LinearRegression lr(1e1);

	lr.fit(A, b);

	DB(lr.predict(A) << "\n\n");

	DB(lr.phi << "    " << lr.bias);
}




auto readData (string str, char del = ' ')
{
    ifstream in(str);

    int N = 0, M = 0;

    vector<double> data; data.reserve(1e4);


    while(getline(in, str))
    {
        stringstream ss(str);
        string value;
        
        int cnt = 0;

        while(getline(ss, value, del))
            data.push_back(stod(value)), cnt++;

        if(!N) N = cnt;
        ++M;
    }

    Mat X(M, N-1);
    Vec y(M);
    

    FOR(i, data.size())
    {
        // if(i % N == 0) y(i / N) = data[i];

        // else X(i / N, (i % N)-1) = data[i];

        if(i % N == N-1) y(i / N) = data[i];
        
        else X(i / N, (i % N)) = data[i];
    }

    return make_tuple(X, y);
}




int main ()
{
    auto [X, y] = readData("reg2.txt", ',');

    // OneHotEncoding ohe({0, 5, 6});
    // X = ohe.fitTransform(X);

    auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3, 42);

    // Standardize st;
    // X_train = st.fitTransform(X_train);
    // X_test = st.transform(X_test);



    LinearRegression lr;

    vector<double> C;

    for(double x = -5; x <= 6; x += 0.5)
        C.push_back(pow(10.0, x));


    auto gs = makeGridsearchCV(lr, make_pair([](auto& est, double x){ est.sigmaP = x; }, C), 10);

    gs.fit(X_train, y_train);

    lr = gs.bestEstimator;


    Vec y_pred(y_test.rows());
    
    FOR(i, y_test.rows())
        y_pred(i) = lr.predict(Vec(X_test.row(i)));

    db((y_test - y_pred).norm() / y_test.rows(), "\n");
    db(lr.sigmaP);
    

    return 0;
}