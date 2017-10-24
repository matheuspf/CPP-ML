#ifndef CPP_ML_CLASSIFIER_H
#define CPP_ML_CLASSIFIER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"


#define USING_CLASSIFIER_BASE(CLASSIFIER) using Base = CLASSIFIER;    \
                                          using Base::Base,           \
                                                Base::lenc,           \
                                                Base::numClasses,     \
                                                Base::positiveClass,  \
                                                Base::negativeClass,  \
                                                Base::encodeLabels,   \
                                                Base::fit,            \
                                                Base::predict;



template <class Impl>
struct ClassifierBase
{   
    void fit (const Mat& X, const Veci& y, int numClasses_ = 0)
    {
        numClasses = numClasses_;
        encodeLabels = numClasses <= 0;

        if(encodeLabels)
        {
            lenc.fit(y);

            numClasses = lenc.numClasses;

            Veci y_enc;
            
            if(numClasses == 2)
                y_enc = lenc.transform(y, positiveClass, negativeClass);

            else
                y_enc = lenc.transform(y);


            return static_cast<Impl&>(*this).impl().fit_(X, y_enc);
        }
            
        return static_cast<Impl&>(*this).impl().fit_(X, y);
    }



    auto predict (const Vec& x)
    {
        auto label = static_cast<Impl&>(*this).impl().predict_(x);

        if(encodeLabels)
            label = lenc.reverseMap[label];
        
        return label;
    }

    auto predict (const Mat& X)
    {
        auto labels = static_cast<Impl&>(*this).impl().predict_(X);

        if(encodeLabels)
        {
            std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](const auto& label)
            {
                return lenc.reverseMap[label];
            });
        }

        return labels;
    }
    
    
    LabelEncoder<int> lenc;
    
    int numClasses;

    int positiveClass;
    int negativeClass;

    bool encodeLabels;


//private:

    ClassifierBase (int positiveClass = 1, int negativeClass = 0) :
                    positiveClass(positiveClass), negativeClass(negativeClass) {}

    friend Impl;
};




template <class Impl>
struct Classifier : public ClassifierBase<Classifier<Impl>>
{
    USING_CLASSIFIER_BASE(ClassifierBase<Classifier<Impl>>);


    decltype(auto) impl ()
    {
        return static_cast<Impl&>(*this);
    }

    
    template <bool T = false>
    void fit_ (const Mat&, const Veci&)
    {
        static_assert(T, "fit_ method not defined");
    }

    template <bool T = false>
    int predict_ (const Vec&)
    {
        static_assert(T, "predict_ (batch observation) method not defined");
    }

    template <bool T = false>
    Veci predict_ (const Mat&)
    {
        static_assert(T, "predict_ (batch observation) method not defined");
    }


private:

    friend Impl;
};




namespace poly
{

struct Classifier : public ClassifierBase<Classifier>
{
    USING_CLASSIFIER_BASE(ClassifierBase<Classifier>);


    decltype(auto) impl ()
    {
        return *this;
    }


    virtual void fit_ (const Mat&, const Veci&) = 0;

    virtual int predict_ (const Vec&) = 0;

    virtual Veci predict_ (const Mat&) = 0;
};

}




#undef USING_CLASSIFIER_BASE




#endif // CPP_ML_CLASSIFIER_H