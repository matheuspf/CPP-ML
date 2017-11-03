#ifndef CPP_ML_CLASSIFIER_H
#define CPP_ML_CLASSIFIER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"


#define USING_CLASSIFIER(...) using BaseClassifier = __VA_ARGS__;   \
                              using BaseClassifier::lenc,           \
                                    BaseClassifier::numClasses,     \
                                    BaseClassifier::positiveClass,  \
                                    BaseClassifier::negativeClass,  \
                                    BaseClassifier::encodeLabels,   \
                                    BaseClassifier::fit,            \
                                    BaseClassifier::predict;



template <class Impl>
struct ClassifierBase
{
    template <typename... Args>
    void fit (const Mat& X, const Veci& y, Args&&... args)
    {
        fit(X, y, 0, std::forward<Args>(args)...);
    }


    template <typename... Args>
    void fit (const Mat& X, const Veci& y, int numClasses_, Args&&... args)
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


            return static_cast<Impl&>(*this).impl().fit_(X, y_enc, std::forward<Args>(args)...);
        }
            
        return static_cast<Impl&>(*this).impl().fit_(X, y, std::forward<Args>(args)...);
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
    USING_CLASSIFIER(ClassifierBase<Classifier<Impl>>);
    using BaseClassifier::BaseClassifier;


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
    USING_CLASSIFIER(ClassifierBase<Classifier>);
    using BaseClassifier::BaseClassifier;


    decltype(auto) impl ()
    {
        return *this;
    }


    virtual void fit_ (const Mat&, const Veci&) = 0;

    virtual int predict_ (const Vec&) = 0;

    virtual Veci predict_ (const Mat&) = 0;
};

}



template <class T, bool Polymorphic = false>
using PickClassifierBase = std::conditional_t<Polymorphic, poly::Classifier, Classifier<T>>;






#endif // CPP_ML_CLASSIFIER_H