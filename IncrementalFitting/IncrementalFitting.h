#ifndef CPP_ML_INCREMENTAL_FITTING_H
#define CPP_ML_INCREMENTAL_FITTING_H

#include "../Modelo.h"

#include "../Classification/Classifier.h"

#include "../Optimization/Newton/Newton.h"

#include "../Regularizers.h"



double logLikelihood (Vec x, const Veci& y)
{
    x = 1.0 / (1.0 + Eigen::exp(-x.array()));

    return -    (y.array().cast<double>() * log(x.array()) + 
             (1.0 - y.array().cast<double>()) * log(1.0 - x.array())).sum();
}


struct RBFBase
{
};


template <class Function = RBFBase, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
struct IncrementalFitting : public Classifier<IncrementalFitting<Function, Optimizer>>
{
    IncrementalFitting (const Function& function = Function(), const Optimizer& optimizer = Optimizer()) :
                        function(function), optimizer(optimizer) {}

    IncrementalFitting (const Optimizer& optimizer, const Function& function = Function()) :
                        function(function), optimizer(optimizer) {}
    

    void fit_ (const Mat& X, const Veci& y, int K = 10, double minGap = 1e-5)
    {
        int M = X.rows(), N = X.cols();

        alphas = Mat::Constant(K, N, 0.0);
        gammas = Vec::Constant(K, 0.0);

        weights = Vec::Constant(K, 0.0);
        intercept = 0.0;

        Vec predictions = Vec::Constant(M, 0.0);


        for(int k = 0; k < K; ++k)
        {
            predictions = predictions.array() - intercept;

            auto func = [&](Vec w) -> double
            {
                double w0 = w(N), wk = w(N+1), g = w(N+2);
                
                w.conservativeResize(N);

                Vec activations = predictions.array() + w0 +
                                  wk * (X * w).array().unaryExpr([](double x){ return ::std::atan(x); });
                                  //wk * Eigen::exp(-(X.rowwise() - w.transpose()).rowwise().squaredNorm().array() * g);
                                  
                return logLikelihood(activations, y);
            };


            RandDouble randDouble;


            Vec w = Vec::NullaryExpr(N+3, [&](int){ return randDouble(-1.0, 1.0); });

            w(N+2) = pow(10.0, randDouble(-3, 3));

            w = optimizer(func, w);


            alphas.row(k) = w.head(N);

            intercept = w(N);

            weights(k) = w(N+1);

            gammas(k) = w(N+2);


            w.conservativeResize(N);

            predictions = predictions.array() + intercept + weights(k) *
                          (X * w).array().unaryExpr([](double x){ return ::std::atan(x); });
                          //Eigen::exp(-(X.rowwise() - w.transpose()).rowwise().squaredNorm().array() * gammas(k));
            
            
            db("Train accuracy:   ", ((predictions.array() > 0.0).cast<int>() == y.array()).cast<double>().sum() / M, "\n");

            // db("W:   ", alphas.row(k), "\n\npred:   ", predictions.transpose(), "\n\ngamma:   ", gammas(k), 
            //    "\n\nweight k:   ", weights(k), "\n\nintercept:   ", intercept, "\n\n\n\n");
        }

        Veci y_pred = predict_(X);

        db("\n\n Train Accuracy:     ", (y.array() == y_pred.array()).cast<double>().sum() / M, "\n\n");
    }


    int predict_ (Vec x)
    {
        x = (alphas * x).array().unaryExpr([](double x){ return ::std::atan(x); });;
        //x = Eigen::exp(-(alphas.rowwise() - x.transpose()).rowwise().squaredNorm().array() * gammas.array());

        return weights.dot(x) + intercept > 0;
    }

    Veci predict_ (const Mat& X)
    {
        Veci pred(X.rows());

        for(int i = 0; i < X.rows(); ++i)
        {
            Vec x = X.row(i);
            pred(i) = predict_(x);
        }

        return pred;
    }



    Mat alphas;

    Vec gammas;
    

    Vec weights;

    double intercept;


    Function function;
    
    Optimizer optimizer;
};



#endif // CPP_ML_INCREMENTAL_FITTING_H