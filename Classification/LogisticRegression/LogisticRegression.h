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

#include "../Classifier.h"


namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>,
          bool EncodeLabels = true, bool Polymorphic = false>
struct LogisticRegression : public PickClassifierBase<LogisticRegression<Regularizer, Optimizer,
                                                      EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<LogisticRegression<Regularizer, Optimizer,
                     EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);


    LogisticRegression (double alpha, const Optimizer& optimizer = Optimizer(), std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8, std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (std::string multiClassType = "OVA", double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        if(numClasses == 2)
            impl = std::make_unique<poly::LogisticRegressionTwoClass<Regularizer, Optimizer, false>>(alpha, optimizer);
        
        else
        {
            if(multiClassType == "OVA")
                impl = std::make_unique<OVA<poly::LogisticRegressionTwoClass<Regularizer, Optimizer, false>, false>>(alpha, optimizer);

            else if(multiClassType == "OVO")
                impl = std::make_unique<OVO<poly::LogisticRegressionTwoClass<Regularizer, Optimizer, false>, false>>(alpha, optimizer);

            else if(multiClassType == "Multi")
                impl = std::make_unique<poly::LogisticRegressionMultiClass<Regularizer, Optimizer, false>>(alpha, optimizer);
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

    Regularizer regularizer;

    Optimizer optimizer;

    std::string multiClassType;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegression = impl::LogisticRegression<Regularizer, Optimizer, EncodeLabels, false>;

namespace poly
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegression = impl::LogisticRegression<Regularizer, Optimizer, EncodeLabels, true>;

} // namespace poly





#endif // CPP_ML_LOGISTIC_REGRESSION_H