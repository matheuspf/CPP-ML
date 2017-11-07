#ifndef CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS
#define CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS

#include "../../Modelo.h"

#include "../Classifier.h"

#include "LogisticRegressionBase.h"



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>,
          bool EncodeLabels = true, bool Polymorphic = false>
struct LogisticRegressionTwoClass : public LogisticRegressionBase<Regularizer, Optimizer>,
                                    PickClassifierBase<LogisticRegressionTwoClass<Regularizer, Optimizer,
                                                       EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer>);
    USING_CLASSIFIER(PickClassifierBase<LogisticRegressionTwoClass<Regularizer, Optimizer,
                                        EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();


        auto func = [&](const Vec& w) -> double
        {
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



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegressionTwoClass = impl::LogisticRegressionTwoClass<Regularizer, Optimizer, EncodeLabels, false>;


namespace poly
{
    template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
    using LogisticRegressionTwoClass = impl::LogisticRegressionTwoClass<Regularizer, Optimizer, EncodeLabels, true>;
}
              



#endif // CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS