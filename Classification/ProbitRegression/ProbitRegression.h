#ifndef CPP_ML_PROBIT_REGRESSION_H
#define CPP_ML_PROBIT_REGRESSION_H

#include "../LogisticRegression/LogisticRegressionTwoClass.h"


namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool Polymorphic = false>
struct ProbitRegression : public impl::LogisticRegressionTwoClass<Regularizer, Optimizer, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(impl::LogisticRegressionTwoClass<Regularizer, Optimizer, Polymorphic>);
    
    using BaseLogisticRegression::predictMargin, BaseLogisticRegression::predict, BaseLogisticRegression::w,
          BaseLogisticRegression::intercept;


    void fit (const Mat& X, const Veci& y)
    {
        optimize(X, y);

        intercept = w(N-1);

        w.conservativeResize(N-1);
    }


    double predictProb (const Vec& x)
    {
        return std::erf(predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return this->erf(predictMargin(X));
    }


    Vec erf (Vec x)
    {
        std::transform(std::begin(x), std::end(x), std::begin(x), [](double x)
        {
            return 0.5 * (1.0 + (1.0 / std::sqrt(2)) * std::exp(0.5) * std::erf(x));
        });

        return x;
    }


    void optimize (const Mat& X, const Veci& y)
    {
        auto func = [&](const Vec& w) -> double
        {
            ArrayXd act = this->erf(X * w).array();

            return -(y.array().cast<double>() * log(act) + (1.0 - y.array().cast<double>()) * log(1.0 - act)).sum()
                    + 0.5 * alpha * regularizer(w);
        };

        auto grad = [&](const Vec& w) -> Vec
        {
            ArrayXd act = this->erf(X * w);

            ArrayXd der = (2.0 / std::sqrt(pi())) * (std::exp(0.5) / (2 * std::sqrt(2))) * Eigen::exp(-Eigen::pow((X * w).array(), 2));

            ArrayXd tt = (y.array().cast<double>() / act) - ((1.0 - y.array().cast<double>()) / (1.0 - act));

            Vec g = -(X.transpose() * (tt * der).matrix()) + 0.5 * alpha * regularizer.gradient(w);

            return g;
        };


        w = Vec::Constant(N, 0.0);

        w = optimizer(func, grad, w);
    }
};

} // namespace impl




template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using ProbitRegression = impl::Classifier<impl::ProbitRegression<L2, Optimizer, false>, EncodeLabels>;


namespace poly
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using ProbitRegression = impl::Classifier<impl::ProbitRegression<L2, Optimizer, true>, EncodeLabels>;
} // namespace poly



#endif // CPP_ML_PROBIT_REGRESSION_H