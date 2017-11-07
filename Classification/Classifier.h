#ifndef CPP_ML_CLASSIFIER_H
#define CPP_ML_CLASSIFIER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"


#define USING_CLASSIFIER(...) using BaseClassifier = __VA_ARGS__;   \
                              using BaseClassifier::lenc,           \
                                    BaseClassifier::numClasses,     \
                                    BaseClassifier::positiveClass,  \
                                    BaseClassifier::negativeClass,  \
                                    BaseClassifier::fit,            \
                                    BaseClassifier::predict,        \
                                    BaseClassifier::needsIntercept, \
                                    BaseClassifier::M,              \
                                    BaseClassifier::N;




namespace impl
{

template <class Impl>
struct ClassifierBase
{
    ClassifierBase(bool needsIntercept_ = true) : ClassifierBase(1, 0, needsIntercept_) {}

    ClassifierBase(int positiveClass_, int negativeClass_) : ClassifierBase(positiveClass_, negativeClass_, true) {}

    ClassifierBase (int positiveClass_, int negativeClass_, bool needsIntercept_) :
                    positiveClass(positiveClass_), negativeClass(negativeClass_), needsIntercept(needsIntercept_), numClasses(0) {}



    template <typename... Args>
    void fit (const Mat& X, const Veci& y, Args&&... args)
    {
        return fit(X, y, true, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void fit (const Mat& X, const Veci& y, bool addIntercept, Args&&... args)
    {
        addIntercept = addIntercept & needsIntercept;

        M = X.rows();
        N = addIntercept ? X.cols() + 1 : X.cols();

        return static_cast<Impl&>(*this).impl().fit_(addIntercept & needsIntercept ? addInterceptColumn(X) : X,
                                                     y, std::forward<Args>(args)...);        
    }


    auto predict (const Vec& x)
    {
        return static_cast<Impl&>(*this).impl().predict_(x);
    }

    auto predict (const Mat& X)
    {
        return static_cast<Impl&>(*this).impl().predict_(X);
    }
    



    Mat addInterceptColumn (Mat X)
    {
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;

        return X;
    }



    friend Impl;


    LabelEncoder<int> lenc;
    
    int numClasses;

    int positiveClass;
    int negativeClass;

    bool needsIntercept;

    int M;
    int N;
};





template <class Impl, bool EncodeLabels>
struct Classifier : public ClassifierBase<Impl>
{
    USING_CLASSIFIER(ClassifierBase<Impl>)
    using BaseClassifier::BaseClassifier;


    template <typename... Args>
    void fit (const Mat& X, const Veci& y, Args&&... args)
    {
        return fit(X, y, true, std::forward<Args>(args)...);
    }


    template <typename... Args>
    void fit (const Mat& X, const Veci& y, bool addIntercept, Args&&... args)
    {
        lenc.fit(y);

        numClasses = lenc.numClasses;

        Veci y_enc;
        
        if(numClasses == 2)
            y_enc = lenc.transform(y, positiveClass, negativeClass);

        else
            y_enc = lenc.transform(y);
            

        return BaseClassifier::fit(X, y_enc, addIntercept, std::forward<Args>(args)...);
    }



    auto predict (const Vec& x)
    {
        auto label = BaseClassifier::predict(x);

        return lenc.reverseMap[label];
    }

    auto predict (const Mat& X)
    {
        auto labels = BaseClassifier::predict(X);

        std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](const auto& label)
        {
            return this->lenc.reverseMap[label];
        });

        return labels;
    }
    

    friend Impl;
};





template <class Impl>
struct Classifier<Impl, false> : public ClassifierBase<Impl>
{
    USING_CLASSIFIER(ClassifierBase<Impl>)
    using BaseClassifier::BaseClassifier;

    friend Impl;
};

} // namespace impl









template <class Impl, bool EncodeLabels = true>
struct Classifier : public impl::Classifier<Classifier<Impl, EncodeLabels>, EncodeLabels>
{
    USING_CLASSIFIER(impl::Classifier<Classifier<Impl, EncodeLabels>, EncodeLabels>);
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

template <bool EncodeLabels = true>
struct Classifier : public ::impl::Classifier<Classifier<EncodeLabels>, EncodeLabels>
{
    USING_CLASSIFIER(::impl::Classifier<Classifier<EncodeLabels>, EncodeLabels>);
    using BaseClassifier::BaseClassifier;


    decltype(auto) impl ()
    {
        return *this;
    }


    virtual void fit_ (const Mat&, const Veci&) = 0;

    virtual void fit_ (const Mat&, const Veci&, const Vec&) {}

    virtual int predict_ (const Vec&) = 0;

    virtual Veci predict_ (const Mat&) = 0;
};

}



template <class T, bool EncodeLabels = true, bool Polymorphic = false>
using PickClassifierBase = std::conditional_t<Polymorphic, poly::Classifier<EncodeLabels>, Classifier<T, EncodeLabels>>;






#endif // CPP_ML_CLASSIFIER_H