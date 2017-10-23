#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Optimization/Newton/Newton.h"

#include "../../Regularizers.h"

#include "../OVA/OVA.h"

#include "../OVO/OVO.h"

#include "LogisticRegressionTwoClass.h"

#include "LogisticRegressionMulticlass.h"

#include "../ClassEncoder.h"



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
struct LogisticRegression : public ClassEncoder<LogisticRegression<Regularizer, Optimizer>>
{
    using Base = ClassEncoder<LogisticRegression<Regularizer, Optimizer>>;
    using Base::fit, Base::predict, Base::numClasses;


    LogisticRegression (double alpha = 1e-8, const Optimizer& optimizer = Optimizer(), std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8, std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (std::string multiClassType = "OVA", double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = new LogisticRegressionTwoClass<Regularizer, Optimizer>(alpha, optimizer);
        
        else
        {
            if(multiClassType == "OVA")
                impl = new OVA<LogisticRegressionTwoClass<Regularizer, Optimizer>>(alpha, optimizer);

            else if(multiClassType == "OVO")
                impl = new OVO<LogisticRegressionTwoClass<Regularizer, Optimizer>>(alpha, optimizer);
        }
        

        impl->fit(X, y, numClasses);
    }



    int predict_ (const Vec& x)
    {
        return impl->predict(x);
    }

    Veci predict_ (const Mat& X)
    {
        return impl->predict(X);
    }



    // double predictProb (const Vec& x)
    // {
    //     return impl->predictProb(x);
    // }

    Vec predictProb (const Mat& X)
    {
        return impl->predictProb(X);
    }


    // double predictMargin (const Vec& x)
    // {
    //     return return impl->predictMargin(x);
    // }

    Vec predictMargin (const Mat& X)
    {
        return impl->predictMargin(X);
    }
    

    
    //std::unique_ptr<LogisticRegressionBase<Regularizer, Optimizer>> impl;
    LogisticRegressionBase<Regularizer, Optimizer>* impl;

    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;

    std::string multiClassType;
};






#endif // CPP_ML_LOGISTIC_REGRESSION_H