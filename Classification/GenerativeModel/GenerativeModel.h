#ifndef CPP_ML_GENERATIVE_MODEL_H
#define CPP_ML_GENERATIVE_MODEL_H

#include "../Classifier.h"

#include "../../Distributions/Gaussian/Gaussian.h"

#include "../../Distributions/Multinomial/Multinomial.h"


namespace impl
{

template <class ClassConditional, bool Polymorphic = false>
struct GenerativeModel : public PickClassifier<Polymorphic>
{
    USING_CLASSIFIER(PickClassifier<Polymorphic>);

    template <typename... Args>
    GenerativeModel (Args&&... args) : baseConditional(std::forward<Args>(args)...) {}


    void fit (const Mat& X, const Veci& y)
    {
        std::vector<int> classCount = lenc.countClasses(y);

        classPrior.params(classCount);

        classConditionals.resize(numClasses, baseConditional);
    
        
        for(int k = 0; k < numClasses; ++k)
        {
            Mat Xk(classCount[k], N);

            for(int i = 0, j = 0; i < M; ++i) if(y(i) == k)
                Xk.row(j++) = X.row(i);
            
            classConditionals[k].fit(Xk);
        }
    }


    int predict (const Vec& x)
    {
        int label = 0;
        double bestPosterior = std::numeric_limits<double>::min();

        for(int k = 0; k < numClasses; ++k)
        {
            double posterior = classPrior(k) * classConditionals[k](x);

            if(posterior > bestPosterior)
            {
                bestPosterior = posterior;
                label = k;
            }
        }

        return label;
    }


    Veci predict (const Mat& X)
    {
        return Veci::NullaryExpr(X.rows(), [&](int i){ return predict(Vec(X.row(i))); });
    }



    ClassConditional baseConditional;

    std::vector<ClassConditional> classConditionals;

    Multinomial classPrior;
};

} // namespace impl



template <class ClassConditional, bool EncodeLabels = true>
using GenerativeModel = impl::Classifier<impl::GenerativeModel<ClassConditional, false>, EncodeLabels>;


namespace poly
{

template <class ClassConditional, bool EncodeLabels = true>
using GenerativeModel = impl::Classifier<impl::GenerativeModel<ClassConditional, false>, EncodeLabels>;

}


#endif // CPP_ML_GENERATIVE_MODEL_H