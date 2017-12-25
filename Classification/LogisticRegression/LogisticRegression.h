#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H


#include "LogisticRegressionTwoClass.h"

#include "LogisticRegressionMultiClass.h"

#include "../OVA/OVA.h"

#include "../OVO/OVO.h"



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<>, bool Polymorphic = false>
struct LogisticRegression : public LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>);


    LogisticRegression (double alpha, const Optimizer& optimizer = Optimizer(), std::string multiClassType = "Multi") :
                        impl(nullptr), BaseLogisticRegression(alpha, optimizer), multiClassType(multiClassType) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8, std::string multiClassType = "Multi") :
                        impl(nullptr), BaseLogisticRegression(alpha, optimizer), multiClassType(multiClassType) {}

    LogisticRegression (std::string multiClassType = "Multi", double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        impl(nullptr), BaseLogisticRegression(alpha, optimizer), multiClassType(multiClassType) {}


    LogisticRegression(const LogisticRegression& lr) : impl(lr.impl ? lr.impl->clone() : nullptr), 
                                                       BaseLogisticRegression(lr), 
                                                       multiClassType(lr.multiClassType) {}

    LogisticRegression& operator= (const LogisticRegression& lr)
    {
        BaseLogisticRegression::operator=(lr);

        if(lr.impl)
            impl = std::unique_ptr<poly::Classifier>(lr.impl->clone());

        alpha = lr.alpha;
        regularizer = lr.regularizer;
        optimizer = lr.optimizer;
        multiClassType = lr.multiClassType;
    }


    void fit (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = std::make_unique<poly::LogisticRegressionTwoClass<Regularizer, Optimizer, false>>(alpha, optimizer);
        
        else
        {
            if(multiClassType == "OVA")
                impl = std::make_unique<poly::OVA<::LogisticRegressionTwoClass<Regularizer, Optimizer, false>, false>>(alpha, optimizer);

            else if(multiClassType == "OVO")
                impl = std::make_unique<poly::OVO<::LogisticRegressionTwoClass<Regularizer, Optimizer, false>, false>>(alpha, optimizer);

            else if(multiClassType == "Multi")
                impl = std::make_unique<poly::LogisticRegressionMultiClass<Regularizer, Optimizer, false>>(alpha, optimizer);

            else
            {
                db((string("Multi-class type is invalid:  ") + multiClassType));
                exit(0);
            }
        }
        
        
        impl->numClasses = numClasses;

        impl->fit(X, y, false);
    }



    int predict (const Vec& x)
    {
        return impl->predict(x);
    }

    Veci predict (const Mat& X)
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

    std::string multiClassType;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegression = impl::Classifier<impl::LogisticRegression<Regularizer, Optimizer, false>, EncodeLabels>;


namespace poly
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegression = impl::Classifier<impl::LogisticRegression<Regularizer, Optimizer, true>, EncodeLabels>;

} // namespace poly





#endif // CPP_ML_LOGISTIC_REGRESSION_H