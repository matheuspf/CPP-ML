#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Optimization/Newton/Newton.h"

#include "../../Regularizers.h"

#include "../OVA/OVA.h"

#include "../OVO/OVO.h"

#include "LogisticRegressionTwoClass.h"

//#include "LogisticRegressionMulticlass.h"

#include "../Classifier.h"


namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool Polymorphic = false>
struct LogisticRegression : PickClassifierBase<LogisticRegression<Regularizer, Optimizer, Polymorphic>, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<LogisticRegression<Regularizer, Optimizer, Polymorphic>, Polymorphic>);


    LogisticRegression (double alpha, const Optimizer& optimizer = Optimizer(), std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8, std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (std::string multiClassType = "OVA", double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = std::make_unique<poly::LogisticRegressionTwoClass<Regularizer, Optimizer>>(alpha, optimizer);
        
        else
        {
            if(multiClassType == "OVA")
                impl = std::make_unique<OVA<poly::LogisticRegressionTwoClass<Regularizer, Optimizer>>>(alpha, optimizer);

            else if(multiClassType == "OVO")
                impl = std::make_unique<OVO<poly::LogisticRegressionTwoClass<Regularizer, Optimizer>>>(alpha, optimizer);
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

    // Vec predictProb (const Mat& X)
    // {
    //     return impl->predictProb(X);
    // }


    // double predictMargin (const Vec& x)
    // {
    //     return return impl->predictMargin(x);
    // }

    // Vec predictMargin (const Mat& X)
    // {
    //     return impl->predictMargin(X);
    // }
    

    
    std::unique_ptr<poly::Classifier> impl;


    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;

    std::string multiClassType;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
using LogisticRegression = impl::LogisticRegression<Regularizer, Optimizer, false>;

namespace poly
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
using LogisticRegression = impl::LogisticRegression<Regularizer, Optimizer, true>;

} // namespace poly





#endif // CPP_ML_LOGISTIC_REGRESSION_H