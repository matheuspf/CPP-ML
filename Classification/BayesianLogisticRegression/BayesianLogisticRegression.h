#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H


#include "../LogisticRegression/LogisticRegression.h"

#include "BayesianLogisticRegressionTwoClass.h"

#include "BayesianLogisticRegressionMultiClass.h"

#include "../OVA/OVA.h"

#include "../OVO/OVO.h"




namespace impl
{

template <bool Polymorphic = false>
struct BayesianLogisticRegression : public PickClassifier<Polymorphic>
{
    USING_CLASSIFIER(PickClassifier<Polymorphic>);
    using BaseClassifier::BaseClassifier;


    BayesianLogisticRegression (const std::string& multiClassType = "Multi") : multiClassType(multiClassType) {}

    BayesianLogisticRegression (const BayesianLogisticRegression& blr) : BaseClassifier(blr), multiClassType(multiClassType),
                                                                         impl(blr.impl ? blr.impl->clone() : nullptr) {}


    BayesianLogisticRegression& operator = (const BayesianLogisticRegression& blr)
    {
        BaseClassifier::operator=(blr);

        if(blr.impl)
            impl = std::unique_ptr<poly::Classifier>(blr.impl->clone());

        multiClassType = blr.multiClassType;
    }


    void fit (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = std::make_unique<poly::BayesianLogisticRegressionTwoClass<false>>();
        
        else
        {
            if(multiClassType == "OVA")
                impl = std::make_unique<poly::OVA<::BayesianLogisticRegressionTwoClass<false>>>();
            
            else if(multiClassType == "OVO")
                impl = std::make_unique<poly::OVO<::BayesianLogisticRegressionTwoClass<false>>>();

            else if(multiClassType == "Multi")
                impl = std::make_unique<poly::BayesianLogisticRegressionMultiClass<false>>();

            else
            {
                std::cerr << "Invalid multiClassType:  " << multiClassType << "\n";
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
    

    std::string multiClassType;

    std::unique_ptr<poly::Classifier> impl;
};

} // namespace impl



template <bool EncodeLabels = true>
using BayesianLogisticRegression =  impl::Classifier<impl::BayesianLogisticRegression<false>, EncodeLabels>;

namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegression =  impl::Classifier<impl::BayesianLogisticRegression<true>, EncodeLabels>;

} // namespace poly





#endif // CPP_ML_LOGISTIC_REGRESSION_H