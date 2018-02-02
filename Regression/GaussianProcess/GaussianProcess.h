#ifndef CPP_ML_GAUSSIAN_PROCESS_H
#define CPP_ML_GAUSSIAN_PROCESS_H

#include "../Regressor.h"

#include "../../Optimization/CG/CG.h"
#include "../../Optimization/BFGS/BFGS.h"
#include "../../Optimization/Newton/Newton.h"

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
        update();


        Newton<Goldstein> opt;
        //BFGS<> opt;
        //CG<> opt;

        opt.maxIterations = 10;

        Vec x0 = Vec::Constant(5, -10.0);

        x0 = opt([&](const Vec& z)
        {
            set(z);

            return -logLikelihood();
        }, x0);
        
        set(x0);
    }



    void set (const Vec& z)
    {
        beta = exp(z(0));
        //kernel.phi(1) = exp(z(1));
        kernel.set(exp(z.tail(z.rows() - 1).array()));

        update();
    }


    void update ()
    {
        C = kernel(X, X);
        C.diagonal().array() += beta;
        
        w = solver(C, y);
    }



    double predict (const Vec& x) const
    {
        return kernel(X, x).dot(w);
    }

    Vec predict (const Mat& X_) const
    {
        return kernel(X_, X) * w;
    }


    double variance (const Vec& x) const
    {
    }

    Vec variance (const Mat& X) const
    {
    }


    double logLikelihood () const
    {
        return -0.5 * (log(C.diagonal().array()).sum() + w.dot(y) + M * log(2*pi()));
    }




    Mat X;
    Vec y;

    Mat C;
    Vec w;
    
    double beta;

    Kernel kernel;
    Solver solver;
};

} // namespace impl


template <class Kernel = ExponentialKernel, class Solver = Invert>
using GaussianProcess = impl::GaussianProcess<Kernel, Solver, false>;


namespace poly
{

template <class Kernel = ExponentialKernel, class Solver = Invert>
using GaussianProcess = impl::GaussianProcess<Kernel, Solver, true>;

}



#endif // CPP_ML_GAUSSIAN_PROCESS_H