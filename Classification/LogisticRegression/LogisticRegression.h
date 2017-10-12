#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "../../Modelo.h"

#include "../../Optimization/Newton/Newton.h"


template <class Optimizer = Newton<Goldstein, CholeskyIdentity>>
struct LogisticRegression
{
    LogisticRegression (double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        alpha(alpha), optimizer(optimizer) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8) :
                        alpha(alpha), optimizer(optimizer) {}


    LogisticRegression& fit (Mat X, Veci y)
    {
        assert(X.rows() == y.rows() && "Observation matrix and labels differ in number of samples.");

        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;
        
        M = X.rows(), N = X.cols();



        ArrayXd sig;


        auto func = [&](const Vec& w) -> double
        {
            sig = sigmoid((X * w).array());

            return -(y.array().cast<double>() * log(sig) + (1.0 - y.array().cast<double>()) * log(1.0 - sig)).sum()
                   //+ 0.5 * alpha * (w.array().abs().sum() - abs(w(N-1)));
                   //+ 0.5 * alpha * (w.squaredNorm() - pow(w(N-1), 2));
                   + 0.5 * alpha * w.squaredNorm();
        };

        auto grad = [&](const Vec& w) -> Vec
        {
            return X.transpose() * (sig.matrix() - y.cast<double>()) + alpha * w;
            //return X.transpose() * (sigmoid((X * w).array()).matrix() - y.cast<double>()) + alpha * w;
        };

        auto hess = [&](Vec w) -> Mat
        {
            Vec sig = sigmoid((X * w).array());

            return X.transpose() * sig.asDiagonal() * X;
        };


        w = Vec::Constant(N, 0.0);

        //generate(w.data(), w.data() + N, [&]{ return randDouble(-1.0/sqrt(N), 1.0/sqrt(N)); });

        
        w = optimizer(func, grad, w);

        transform(begin(w), end(w), begin(w), [](double x){ return abs(x) < 1e-7 ? 0.0 : x; });


        intercept = w(N-1);

        w.conservativeResize(N-1);
    }



    int predict (const Vec& x)
    {
        return w.dot(x) + intercept > 0.0;
    }

    auto predict (const Mat& X)
    {
        return ((X * w).array() + intercept > 0.0).cast<int>();
    }





    template <class T>
    auto sigmoid (const T& x)
    {
        using std::exp;
        using Eigen::exp;

        return 1.0 / (1.0 + exp(-x));
    }


    


    double alpha;

    Optimizer optimizer;


    Vec w;

    double intercept;

    
    int M, N;


    unordered_map<int, int> classMap;
    unordered_map<int, int> reverseClassMap;

    RandDouble randDouble;
};


#endif // CPP_ML_LOGISTIC_REGRESSION_H