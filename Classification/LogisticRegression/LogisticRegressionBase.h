#ifndef CPP_ML_LOGISTIC_REGRESSION_BASE_H
#define CPP_ML_LOGISTIC_REGRESSION_BASE_H

#include "../Classifier.h"

#include "../../Regularizers.h"

#include "../../Optimization/Newton/Newton.h"


#define USING_LOGISTIC_REGRESSION(...) USING_CLASSIFIER_BASE(__VA_ARGS__);                      \
                                       using BaseLogisticRegression = __VA_ARGS__;              \
                                       using BaseLogisticRegression::BaseLogisticRegression,    \
                                             BaseLogisticRegression::alpha,                     \
                                             BaseLogisticRegression::regularizer,               \
                                             BaseLogisticRegression::optimizer,                 \
                                             BaseLogisticRegression::sigmoid,                   \
                                             BaseLogisticRegression::softmax,                   \
                                             BaseLogisticRegression::logSoftmax;



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<>, bool Polymorphic = false>
struct LogisticRegressionBase : public PickClassifier<Polymorphic>
{
    USING_CLASSIFIER_BASE(PickClassifier<Polymorphic>);
    using BaseClassifier::BaseClassifier;
    

    LogisticRegressionBase (double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                            alpha(alpha), optimizer(optimizer) {}

    LogisticRegressionBase (const Optimizer& optimizer, double alpha = 1e-8) :
                            alpha(alpha), optimizer(optimizer) {}
    

    template <class T>
    static auto sigmoid (const T& x)
    {
        using std::exp;
        using Eigen::exp;

        return 1.0 / (1.0 + exp(-x));
    }


    static Vec softmax (Vec x)
    {
        double maxVal = x.maxCoeff();

        double sum = std::log(Eigen::exp(x.array() - maxVal).sum()) + maxVal;

        return Eigen::exp(x.array() - sum);
    }


    static Mat softmax (Mat X)
    {
        Vec maxVal = X.rowwise().maxCoeff();

        Vec logExpSum = Eigen::log(Eigen::exp((X.colwise() - maxVal).array()).rowwise().sum()).matrix().colwise() + maxVal;

        for(int i = 0; i < X.rows(); ++i)
            if(std::isnan(logExpSum(i)) || std::isinf(logExpSum(i)))
                logExpSum(i) = maxVal(i);

        X = Eigen::exp((X.colwise() - logExpSum).array());
        
        return X;
    }


    static double logSoftmax (const Vec& vx, int k)
    {
        return vx(k) - std::log(Eigen::exp(vx.array()).sum());
    }

    static Vec logSoftmax (const Vec& vx)
    {
        return vx.array() - std::log(Eigen::exp(vx.array()).sum());
    }


    
    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<>>
using LogisticRegressionBase = impl::LogisticRegressionBase<Regularizer, Optimizer, false>;

namespace poly
{
template <class Regularizer = L2, class Optimizer = Newton<>>
using LogisticRegressionBase = impl::LogisticRegressionBase<Regularizer, Optimizer, true>;
}



#endif // CPP_ML_LOGISTIC_REGRESSION_BASE_H