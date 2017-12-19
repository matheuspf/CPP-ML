#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "BayesianLogisticRegressionTwoClass.h"

#include "BayesianLogisticRegressionMultiClass.h"

#include "../OVA/OVA.h"

#include "../OVO/OVO.h"




namespace impl
{

template <bool EncodeLabels = true, bool Polymorphic = false>
struct BayesianLogisticRegression : public PickClassifierBase<BayesianLogisticRegression<EncodeLabels, Polymorphic>, 
                                                              EncodeLabels, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<BayesianLogisticRegression<EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);


    BayesianLogisticRegression (std::string multiClassType = "Multi") : impl(nullptr), multiClassType(multiClassType) {}

    BayesianLogisticRegression(const BayesianLogisticRegression& lr) : impl(lr.impl ? lr.impl->clone() : nullptr), 
                                                                       alpha(lr.alpha), multiClassType(lr.multiClassType) {}

    BayesianLogisticRegression& operator= (const BayesianLogisticRegression& lr)
    {
        if(lr.impl)
            impl = std::unique_ptr<poly::Classifier<false>>(lr.impl->clone());

        alpha = lr.alpha;
        multiClassType = lr.multiClassType;
    }


    void fit_ (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = std::make_unique<poly::BayesianLogisticRegressionTwoClass<false>>();
        
        else
        {
            if(multiClassType == "OVA")
                impl = std::make_unique<OVA<poly::BayesianLogisticRegressionTwoClass<false>, false>>();

            else if(multiClassType == "OVO")
                impl = std::make_unique<OVO<poly::BayesianLogisticRegressionTwoClass<false>, false>>();

            else if(multiClassType == "Multi")
                impl = std::make_unique<poly::BayesianLogisticRegressionMultiClass<false>>();

            else
                assert(0 && (string("Multi-class type is invalid:  ") + multiClassType).c_str());
        }
        
        impl->numClasses = numClasses;

        impl->fit_(X, y);
    }



    int predict_ (const Vec& x)
    {
        return impl->predict_(x);
    }

    Veci predict_ (const Mat& X)
    {
        return impl->predict_(X);
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
    

    
    std::unique_ptr<poly::Classifier<false>> impl;


    double alpha;

    std::string multiClassType;
};

} // namespace impl



template <bool EncodeLabels = true>
using BayesianLogisticRegression = impl::BayesianLogisticRegression<EncodeLabels, false>;

namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegression = impl::BayesianLogisticRegression<EncodeLabels, true>;

} // namespace poly





#endif // CPP_ML_LOGISTIC_REGRESSION_H