#ifndef CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_H
#define CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_H

#include "../Classifier.h"

#include "../../Optimization/LineSearch/Brents/Brents.h"



namespace impl
{

template <bool EncodeLabels = true, bool Polymorphic = false>
struct BayesianLogisticRegression : PickClassifierBase<BayesianLogisticRegression<EncodeLabels, Polymorphic>,
                                                        EncodeLabels, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<BayesianLogisticRegression<EncodeLabels, Polymorphic>,
                                                        EncodeLabels, Polymorphic>);


    void fit_ (const Mat& X, const Veci& y)
    {
        fit_(X, y, 1e-4, 1e-4, 50);
    }


    void fit_ (const Mat& X, const Veci& y, double gTol, double aTol = 1e-4, int maxIter = 50)
    {
        optimize(X, y, gTol, aTol, maxIter);

        intercept = w(N-1);
        
        w.conservativeResize(N-1);
    }



    void optimize (const Mat& X, const Veci& y, double gTol, double aTol, int maxIter)
    {
        w = Vec::Constant(N, 0.0);

        alpha = 0.0;

        double oldAlpha = alpha;

        Vec a, s, R, g;

        
        for(int iter = 0; iter < maxIter; ++iter)
        {
            a = X * w;

            s = sigmoid(a.array());

            R = s.array() * (1.0 - s.array());

            g = X.transpose() * (s - y.cast<double>()) + alpha * w;

            Sn = X.transpose() * R.asDiagonal() * X;

            Eigen::EigenSolver<Mat> eigSolver(Sn);

            Sn.diagonal().array() += alpha;

            w = w - solveMat(Sn, g);


            const ArrayXd& eigVals = eigSolver.eigenvalues().real().array();

            double gamma = (eigVals / (alpha + eigVals)).sum();

            oldAlpha = alpha;

            alpha = gamma / w.dot(w);

            
            if(g.norm() < gTol && std::abs(alpha - oldAlpha) < aTol)
                break;
        }
    }





    int predict_ (const Vec& x)
    {
        return predictMargin(x) > 0.0;
    }

    Veci predict_ (const Mat& X)
    {
        return (ArrayXd(predictMargin(X)) > 0.0).cast<int>();
    }


    double predictProb (const Vec& x)
    {
        return sigmoid(kappa(x.dot(Sn * x)) * predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return Vec::NullaryExpr(X.rows(), [&](int i){ return predictProb(Vec(X.row(i))); });
    }


    double predictMargin (const Vec& x)
    {
        return w.dot(x) + intercept;
    }

    Vec predictMargin (const Mat& X)
    {
        return (X * w).array() + intercept;
    }




    template <class T>
    static auto sigmoid (const T& x)
    {
        return 1.0 / (1.0 + exp(-x));
    }


    template <typename T>
    auto kappa (const T& sigma)
    {
        return sqrt(1.0 + pi() * (sigma / 8.0));
    }




    Vec w;

    double intercept;

    double alpha;

    Mat Sn;

};


}



template <bool EncodeLabels = true>
using BayesianLogisticRegression = impl::BayesianLogisticRegression<EncodeLabels, false>;


namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegression = impl::BayesianLogisticRegression<EncodeLabels, true>;
    
}







#endif // CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_H