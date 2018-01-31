#ifndef CPP_ML_GAUSSIAN_PROCESS_H
#define CPP_ML_GAUSSIAN_PROCESS_H

#include "../Regressor.h"

#include "Kernel.h"
#include "Solver.h"


namespace impl
{

template <class Kernel = ExponentialKernel, class Solver = Invert, bool Polymorphic = false>
struct GaussianProcess : public PickRegressor<GaussianProcess<Kernel, Solver, Polymorphic>, Polymorphic>
{
    USING_REGRESSOR(PickRegressor<GaussianProcess<Kernel, Solver, Polymorphic>, Polymorphic>);

    GaussianProcess (const Kernel& kernel = Kernel(), const Solver solver = Solver(), double beta = 1.0) : 
                     BaseRegressor(false), beta(beta), kernel(kernel), solver(solver) {}

    GaussianProcess (double beta, const Kernel& kernel = Kernel(), const Solver solver = Solver()) : 
                     BaseRegressor(false), beta(beta), kernel(kernel), solver(solver) {}
    


    void fit (Mat&& X_, Vec&& y_)
    {
        X = std::move(X_);
        y = std::move(y_);

        fitImpl();   
    }

    void fit (const Mat& X_, const Vec& y_)
    {
        X = X_;
        y = y_;

        fitImpl();
    }
    

    void fitImpl ()
    {
        //optimize(X, y);

        C = kernel(X, X);
        
        C.diagonal().array() += (1.0 / beta);

        C = solver(C);
    }



    double predict (const Vec& x)
    {
        return (kernel(X, x).transpose() * C).dot(y);
    }

    Vec predict (const Mat& X_)
    {
        return (kernel(X_, X) * C) * y;
    }


    double variance (const Vec& x)
    {

    }

    Vec variance (const Mat& X)
    {

    }



    Mat X;
    Vec y;

    Mat C;
    

    double alpha;
    double beta;


    Kernel kernel;

    Solver solver;
};



}






#endif // CPP_ML_GAUSSIAN_PROCESS_H