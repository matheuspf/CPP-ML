#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H

#include "../../Modelo.h"

#include "../ClassEncoder.h"

#include "LogisticRegressionBase.h"


/// To make class a base of LogisticRegressionBaseVirtual
template <class Regularizer, class Optimizer, template <class, bool> class OV, bool Encode = true>
struct LogisticRegressionOV : public OV<LogisticRegressionTwoClass<Regularizer, Optimizer, false>, Encode>,
                                        LogisticRegressionBase<Regularizer, Optimizer>
{
    using Base = OV<LogisticRegressionTwoClass<Regularizer, Optimizer, false>, Encode>;
    using Base::Base;


    void fit (const Mat& X, const Veci& y, int numClasses_ = 0)
    {
        Base::fit(X, y, numClasses_);
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