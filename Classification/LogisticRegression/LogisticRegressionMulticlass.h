#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H

#include "../../Modelo.h"

#include "../Classifier.h"

#include "LogisticRegressionBase.h"



namespace impl
{

template <class Regularizer, class Optimizer, bool Polymorphic = false>
struct LogisticRegressionMultiClass : public LogisticRegressionBase<Regularizer, Optimizer>,
                                    PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer, Polymorphic>,
                                                                                                          Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer>);
    USING_CLASSIFIER(PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer, Polymorphic>, Polymorphic>);


    void fit_ (const Mat& X_, const Veci& y)
    {
        Mat X = X_;
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;

        M = X.rows(), N = X.cols();


        auto func = [&](const Vec& w) -> double
        {
            Mat 

            ArrayXd sig = sigmoid((X * w).array());

            return -(y.array().cast<double>() * log(sig) + (1.0 - y.array().cast<double>()) * log(1.0 - sig)).sum()
                    + 0.5 * alpha * regularizer(w);
                    //+ 0.5 * alpha * regularizer(Vec(w.head(N-1)));
        };

        auto grad = [&](const Vec& w) -> Vec
        {
            return X.transpose() * (sigmoid((X * w).array()).matrix() - y.cast<double>()) +
                    0.5 * alpha * regularizer.gradient(w);
        };


        w = Vec::Constant(N, 0.0);
        
        w = optimizer(func, grad, w);


        intercept = w(N-1);

        w.conservativeResize(N-1);
    }



    int predict_ (const Vec& x)
    {
        return predictMargin(x) > 0.0;
    }

    Veci predict_ (const Mat& X)
    {
        return (ArrayXd(predictMargin(X)) > 0.0).cast<int>();
    }



    double predictProb (const Vec& x)
    {
        return sigmoid(predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return sigmoid(predictMargin(X).array());
    }


    double predictMargin (const Vec& x)
    {
        return w.dot(x) + intercept;
    }

    Vec predictMargin (const Mat& X)
    {
        return (X * w).array() + intercept;
    }



    Vec w;
    
    double intercept;
};

} // namespace impl



template <class Regularizer, class Optimizer>
using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, false>;


namespace poly
{
    template <class Regularizer, class Optimizer>
    using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, true>;
}




#endif // CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H