#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H

#include "../../Modelo.h"

#include "../ClassEncoder.h"

#include "LogisticRegressionTwoClass.h"


/// To make class a base of LogisticRegressionBaseVirtual
template <class Regularizer, class Optimizer, template <class> class OV>
struct LogisticRegressionOV : public OV<LogisticRegressionTwoClass<class Regularizer, class Optimizer>>,
                                        LogisticRegressionTwoClass<class Regularizer, class Optimizer>
{
    using Base = OV<LogisticRegressionTwoClass<class Regularizer, class Optimizer>>;


    template <typename... Args>
    LogisticRegressionOV (Args&&... args) : Base(std::forward<Args>(args)...) {}


    void fit (const Mat& X, const Veci& y, int K)
    {
        Base::fit(X, y, K);
    }


    int predict (const Vec& x)
    {
        return Base::predict(x);
    }

    Veci predict (const Mat& X)
    {
        return Base::predict(X);
    }


    Vec predictMargin (const Mat& X)
    {
        return Base::predictMargin(X);
    }

    Vec predictProb (const Mat& X)
    {
        return Base::predictProb(X);
    }
};




#endif // CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H