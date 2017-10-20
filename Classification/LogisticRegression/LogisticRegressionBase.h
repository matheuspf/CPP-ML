#ifndef CPP_ML_LOGISTIC_REGRESSION_BASE_H
#define CPP_ML_LOGISTIC_REGRESSION_BASE_H

#include "../../Modelo.h"


template <class Regularizer, class Optimizer>
struct LogisticRegressionBase
{
    LogisticRegressionBase (double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                                alpha(alpha), optimizer(optimizer) {}

    LogisticRegressionBase (const Optimizer& optimizer, double alpha = 1e-8) :
                                alpha(alpha), optimizer(optimizer) {}
    
    virtual ~LogisticRegressionBase() {}
    
    LogisticRegressionBase (const LogisticRegressionBase&) = default;
    LogisticRegressionBase (LogisticRegressionBase&&)      = default;
    
    LogisticRegressionBase& operator= (const LogisticRegressionBase&) = default;
    LogisticRegressionBase& operator= (LogisticRegressionBase&&)      = default;


    virtual void fit_ (const Mat&, const Veci&) = 0;

    virtual int  predict (const Vec&) = 0;

    virtual Veci predict (const Mat&) = 0;



    template <class T>
    static auto sigmoid (const T& x)
    {
        using std::exp;
        using Eigen::exp;

        return 1.0 / (1.0 + exp(-x));
    }

    
    int M, N;
    
    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;
};





#endif // CPP_ML_LOGISTIC_REGRESSION_BASE_H