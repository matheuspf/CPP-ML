#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H

#include "../../Modelo.h"

#include "../Classifier.h"

#include "LogisticRegressionBase.h"



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>,
          bool EncodeLabels = true, bool Polymorphic = false>
struct LogisticRegressionMultiClass : public LogisticRegressionBase<Regularizer, Optimizer>,
                                      PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer,
                                                         EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer>);
    USING_CLASSIFIER(PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer,
                                        EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();


        auto func = [&](const Vec& w) -> double
        {
            Mat W = Mat::Map(&w(0), numClasses, N);

            double res = 0;

            for(int i = 0; i < M; ++i)
                res += logSoftmax(W * X.row(i).transpose(), y(i));

            //db(-res, "     ", 0.5 * alpha * regularizer(w), "\n");

            return -res + 0.5 * alpha * regularizer(w);
        };

        auto grad = [&](const Vec& w) -> Vec
        {
            Mat W = Mat::Map(&w(0), numClasses, N);
            
            Mat grad = Mat::Constant(numClasses, N, 0.0);

            for(int i = 0; i < M; ++i)
            {
                Vec sm = logSoftmax(W * X.row(i).transpose());

                sm(y(i)) -= 1.0;

                grad += sm * X.row(i);
            }
            
            return Vec::Map(&grad(0), numClasses * N) + 0.5 * alpha * regularizer.gradient(w);
        };


        RandDouble randDouble(0);

        Vec w0 = Vec::NullaryExpr(numClasses * N, [&](int){ return randDouble(-0.1, 0.1); });
        //Vec w0 = Vec::Constant(numClasses * N, 0.0);

        w0 = optimizer(func, grad, w0);
        

        W = Mat::Map(&w0(0), numClasses, N);

        intercept = W.col(N-1);

        W.conservativeResize(Eigen::NoChange, N-1);
    }



    int predict_ (const Vec& x)
    {
        Vec vals = W * x + intercept;

        double b = -1e20;
        int p = 0;

        for(int i = 0; i < numClasses; ++i)
            if(vals(i) > b)
                b = vals(i), p = i;

        return p;
    }

    Veci predict_ (const Mat& X)
    {
        Veci vals(X.rows());

        for(int i = 0; i < X.rows(); ++i)
            vals(i) = predict_(Vec(X.row(i)));

        return vals;
    }



    // double predictProb (const Vec& x)
    // {
    // }

    // Vec predictProb (const Mat& X)
    // {
    // }


    // double predictMargin (const Vec& x)
    // {
    // }

    // Vec predictMargin (const Mat& X)
    // {
    // }



    Mat W;
    
    Vec intercept;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, EncodeLabels, false>;


namespace poly
{
    template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
    using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, EncodeLabels, true>;
}




#endif // CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H