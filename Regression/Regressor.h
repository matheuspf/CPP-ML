#ifndef CPP_ML_REGRESSOR_H
#define CPP_ML_REGRESSOR_H

#include "../Modelo.h"


#define USING_REGRESSOR_BASE(...) using BaseRegressor = __VA_ARGS__;      \
                                  using BaseRegressor::BaseRegressor,     \
                                        BaseRegressor::M,                 \
                                        BaseRegressor::N,                 \
                                        BaseRegressor::needsIntercept,    \
                                        BaseRegressor::addInterceptColumn;

#define USING_REGRESSOR(...) USING_REGRESSOR_BASE(__VA_ARGS__); \
                             using BaseRegressor::fit,          \
                                   BaseRegressor::predict,      \
                                   BaseRegressor::polymorphic;


namespace impl
{

template <class Impl>
struct Regressor
{
    Regressor(bool needsIntercept = true) : needsIntercept(needsIntercept) {}


    Mat addInterceptColumn (Mat X)
    {
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;

        return X;
    }


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



    // ClassifierImpl* clone () const
    // {
    //     return new ClassifierImpl(*this);
    // }




    bool needsIntercept;

    int M;
    int N;
};

    
} // namespace impl


template <class Impl>
struct Regressor : public impl::Regressor<Regressor<Impl>>
{
    USING_REGRESSOR_BASE(impl::Regressor<Regressor<Impl>>);

    enum { polymorphic = false };

    
    template <bool T = false>
    void fit (const Mat&, const Vec&, bool = true)
    {
        static_assert(T, "fit method not defined");
    }

    template <bool T = false>
    double predict (const Vec&) const
    {
        static_assert(T, "predict method not defined");
    }

    template <bool T = false>
    Vec predict (const Mat&) const
    {
        static_assert(T, "predict (batch) method not defined");
    }
};



namespace poly
{

struct Regressor : public impl::Regressor<Regressor>
{
    USING_REGRESSOR_BASE(impl::Regressor<Regressor>);

    enum { polymorphic = false };


    virtual void fit (const Mat&, const Vec&, bool = true) = 0;

    // virtual void fit (const Mat&, const Vec&, const Vec&) {}

    virtual double predict (const Vec&) const = 0;

    virtual Vec predict (const Mat&) const = 0;


    //virtual Regressor* clone () const = 0;
};

} // namespace poly




template <class Impl, bool Polymorphic>
using PickRegressor = std::conditional_t<Polymorphic, poly::Regressor, Regressor<Impl>>;





#endif // CPP_ML_REGRESSOR_H