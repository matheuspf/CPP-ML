#ifndef CPP_ML_INCREMENTAL_FITTING_H
#define CPP_ML_INCREMENTAL_FITTING_H

#include "../Modelo.h"

#include "../Classification/Classifier.h"

#include "../Optimization/Newton/Newton.h"

#include "../Kernels.h"


double logLikelihood (Vec x, const Veci& y)
{
    x = 1.0 / (1.0 + Eigen::exp(-x.array()));

    return -    (y.array().cast<double>() * log(x.array()) + 
             (1.0 - y.array().cast<double>()) * log(1.0 - x.array())).sum();
}



namespace impl
{

template <class Function = AtanKernel, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool Polymorphic = false>
struct IncrementalFitting : public PickClassifierBase<IncrementalFitting<Function, Optimizer, Polymorphic>, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<IncrementalFitting<Function, Optimizer, Polymorphic>, Polymorphic>);


    IncrementalFitting (const Function& function = Function(), const Optimizer& optimizer = Optimizer()) :
                        function(function), optimizer(optimizer) {}

    IncrementalFitting (const Optimizer& optimizer, const Function& function = Function()) :
                        function(function), optimizer(optimizer) {}



    void params (int maxBasis_, double minGap_ = 1e-5)
    {
        maxBasis = maxBasis_;
        minGap = minGap_;
    }


    void fit_ (const Mat& X, const Veci& y, int maxBasis_, double minGap_ = 1e-5)
    {
        params(maxBasis_, minGap_);

        fit_(X, y);
    }


    void fit_ (const Mat& X, const Veci& y)
    {
        int M = X.rows(), N = X.cols();

        alphas = Mat::Constant(maxBasis, N, 0.0);

        gammas = Vec::Constant(maxBasis, 0.0);

        weights = Vec::Constant(maxBasis, 0.0);

        intercept = 0.0;

        Vec predictions = Vec::Constant(M, 0.0);

        double accuracy = 0.0, prevAccuracy = 0.0, prevIntercept;


        for(int k = 0; k < maxBasis; ++k)
        {
            predictions = predictions.array() - intercept;

            auto func = [&](Vec w) -> double
            {
                double w0 = w(N), wk = w(N+1);
                
                w.conservativeResize(N);

                Vec activations = predictions.array() + w0 + wk * function(X, w).array();

                return logLikelihood(activations, y);
            };


            RandDouble randDouble;

            Vec w = Vec::NullaryExpr(N+2, [&](int){ return randDouble(-1.0, 1.0); });

            w = optimizer(func, w);


            alphas.row(k) = w.head(N);

            weights(k) = w(N+1);

            prevIntercept = intercept;

            intercept = w(N);

            
            w.conservativeResize(N);

            predictions = predictions.array() + intercept + weights(k) * function(X, w).array();


            prevAccuracy = accuracy;

            accuracy = ((predictions.array() > 0.0).cast<int>() == y.array()).cast<double>().sum() / M;

            db("Train accuracy:   ", accuracy, "\n");
            


            if(accuracy - prevAccuracy < minGap)
            {
                alphas.conservativeResize(k, N);
                
                weights.conservativeResize(k);

                intercept = prevIntercept;

                break;
            }
        }

        Veci y_pred = predict_(X);

        db("\n\n Train Accuracy:     ", (y.array() == y_pred.array()).cast<double>().sum() / M, "\n\n");
    }



    int predict_ (const Vec& x)
    {
        return weights.dot(function(alphas, x)) + intercept > 0.0;
    }

    Veci predict_ (const Mat& X)
    {
        return ((ArrayXd(function(X, alphas) * weights).array()) + intercept > 0.0).matrix().cast<int>();
    }



    Mat alphas;

    Vec gammas;
    

    Vec weights;

    double intercept;


    Function function;
    
    Optimizer optimizer;


    int maxBasis = 10;

    double minGap = 1e-5;
};

} // namespace impl



template <class Function = AtanKernel, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
using IncrementalFitting = impl::IncrementalFitting<Function, Optimizer, false>;


namespace poly
{
    template <class Function = AtanKernel, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
    using IncrementalFitting = impl::IncrementalFitting<Function, Optimizer, true>;
}




#endif // CPP_ML_INCREMENTAL_FITTING_H