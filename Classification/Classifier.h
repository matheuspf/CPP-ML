#ifndef CPP_ML_CLASSIFIER_H
#define CPP_ML_CLASSIFIER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"



#define USING_CLASSIFIER_BASE(...) using BaseClassifier = __VA_ARGS__;   \
                                    using BaseClassifier::lenc,           \
                                            BaseClassifier::numClasses,     \
                                            BaseClassifier::positiveClass,  \
                                            BaseClassifier::negativeClass,  \
                                            BaseClassifier::needsIntercept, \
                                            BaseClassifier::addInterceptColumn, \
                                            BaseClassifier::M,              \
                                            BaseClassifier::N;

#define USING_CLASSIFIER(...) USING_CLASSIFIER_BASE(__VA_ARGS__);   \
                              using BaseClassifier::fit,            \
                                    BaseClassifier::predict;



namespace impl
{

struct ClassifierBase
{
    ClassifierBase(bool needsIntercept_ = true) : ClassifierBase(1, 0, needsIntercept_) {}

    ClassifierBase(int positiveClass_, int negativeClass_) : ClassifierBase(positiveClass_, negativeClass_, true) {}

    ClassifierBase (int positiveClass_, int negativeClass_, bool needsIntercept_) :
                    positiveClass(positiveClass_), negativeClass(negativeClass_), needsIntercept(needsIntercept_), numClasses(0) {}



    Mat addInterceptColumn (Mat X)
    {
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;

        return X;
    }



    LabelEncoder<int> lenc;
    
    int numClasses;

    int positiveClass;
    int negativeClass;

    bool needsIntercept;

    int M;
    int N;
};



template <class Impl>
struct ClassifierImpl : public Impl
{
    USING_CLASSIFIER_BASE(Impl);
    using Impl::Impl;


    void fit (const Mat& X, const Veci& y, bool addIntercept = true)
    {
        fitImpl(X, y, addIntercept);
    }

    template <typename U, typename... Args, std::enable_if_t<!std::is_same<std::decay_t<U>, bool>::value>* = nullptr>
    void fit (const Mat& X, const Veci& y, U&& u, Args&&... args)
    {
        fitImpl(X, y, true, std::forward<U>(u), std::forward<Args>(args)...);
    }

    template <typename... Args>
    void fitImpl (const Mat& X, const Veci& y, bool addIntercept, Args&&... args)
    {
        addIntercept = addIntercept & needsIntercept;

        M = X.rows();
        N = addIntercept ? X.cols() + 1 : X.cols();

        Impl::fit(addIntercept & needsIntercept ? addInterceptColumn(X) : X, y, std::forward<Args>(args)...);        
    }


    int predict (const Vec& x)
    {
        return Impl::predict(x);
    }

    Veci predict (const Mat& X)
    {
        return Impl::predict(X);
    }


    ClassifierImpl* clone () const
    {
        return new ClassifierImpl(*this);
    }
};





template <class Impl, bool EncodeLabels>
struct Classifier : public ClassifierImpl<Impl>
{
    USING_CLASSIFIER_BASE(ClassifierImpl<Impl>);
    using ClassifierImpl<Impl>::ClassifierImpl;


    void fit (const Mat& X, const Veci& y, bool addIntercept = true)
    {
        fitImpl(X, y, addIntercept);
    }

    template <typename U, typename... Args, std::enable_if_t<!std::is_same<std::decay_t<U>, bool>::value>* = nullptr>
    void fit (const Mat& X, const Veci& y, U&& u, Args&&... args)
    {
        fitImpl(X, y, true, std::forward<Args>(args)...);
    }


    template <typename... Args>
    void fitImpl (const Mat& X, const Veci& y, bool addIntercept, Args&&... args)
    {
        lenc.fit(y);

        numClasses = lenc.numClasses;

        Veci y_enc;
        
        if(numClasses == 2)
            y_enc = lenc.transform(y, positiveClass, negativeClass);

        else
            y_enc = lenc.transform(y);
            

        return ClassifierImpl<Impl>::fit(X, y_enc, addIntercept, std::forward<Args>(args)...);
    }



    int predict (const Vec& x)
    {
        auto label = ClassifierImpl<Impl>::predict(x);

        return lenc.reverseMap[label];
    }

    Veci predict (const Mat& X)
    {
        auto labels = ClassifierImpl<Impl>::predict(X);

        std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](const auto& label)
        {
            return this->lenc.reverseMap[label];
        });

        return labels;
    }
    

    friend Impl;
};





template <class Impl>
struct Classifier<Impl, false> : public ClassifierImpl<Impl>
{
    USING_CLASSIFIER_BASE(ClassifierImpl<Impl>);
    using ClassifierImpl<Impl>::ClassifierImpl;
};

} // namespace impl








struct Classifier : public impl::ClassifierBase
{
    USING_CLASSIFIER_BASE(impl::ClassifierBase);
    using impl::ClassifierBase::ClassifierBase;


    
    template <bool T = false>
    void fit (const Mat&, const Veci&, bool = true)
    {
        static_assert(T, "fit method not defined");
    }

    template <bool T = false>
    int predict (const Vec&)
    {
        static_assert(T, "predict method not defined");
    }

    template <bool T = false>
    Veci predict (const Mat&)
    {
        static_assert(T, "predict (batch) method not defined");
    }
};




namespace poly
{


struct Classifier : public impl::ClassifierBase
{
    USING_CLASSIFIER_BASE(impl::ClassifierBase);
    using impl::ClassifierBase::ClassifierBase;


    virtual void fit (const Mat&, const Veci&, bool = true) = 0;

    virtual void fit (const Mat&, const Veci&, const Vec&) {}

    virtual int predict (const Vec&) = 0;

    virtual Veci predict (const Mat&) = 0;


    virtual Classifier* clone () const = 0;
};



// template <class Impl, bool EncodeLabels = true>
// struct ClassifierClone : public Classifier<EncodeLabels>
// {
    // virtual Classifier<EncodeLabels>* clone () const
    // {
    //     return new Impl(static_cast<const Impl&>(*this));
    // }
// };


}



template <bool Polymorphic = false>
using PickClassifier = std::conditional_t<Polymorphic, poly::Classifier, Classifier>;






#endif // CPP_ML_CLASSIFIER_H