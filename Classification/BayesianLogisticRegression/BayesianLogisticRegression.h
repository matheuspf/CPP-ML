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
        fit_(X, y, 1e-4, 1e-4, 20, 10);
    }


    void fit_ (const Mat& X, const Veci& y, double gTol, double aTol = 1e-4, int outerIter = 20, int innerIter = 10)
    {
        optimize(X, y, gTol, aTol, outerIter, innerIter);

        intercept = w(N-1);
        
        w.conservativeResize(N-1);
    }



    void optimize (const Mat& X, const Veci& y, double gTol, double aTol, int outerIter, int innerIter)
    {
        //w = Vec::Constant(N, 0.0);

        RandDouble rd(0);
        w = Vec::NullaryExpr(N, [&](int){ return rd(-1.0, 1.0); });        

        alpha = 1e-7;

        double oldAlpha = alpha;

        Vec a, s, R, g;

        
        for(int i = 0; i < outerIter; ++i)
        {
            for(int j = 0; j < innerIter; ++j)
            {
                a = X * w;

                s = sigmoid(a.array());

                R = s.array() * (1.0 - s.array());

                g = X.transpose() * (s - y.cast<double>()) + alpha * w;

                Sn = X.transpose() * R.asDiagonal() * X;
                Sn.diagonal().array() += alpha;

                w = w - inverseMat(Sn) * g;


                if(g.norm() < gTol)
                    break;
            }



            // ArrayXd eigVals = Sn.eigenvalues().real();

            // double gamma = (eigVals / (alpha + eigVals)).sum();

            // oldAlpha = alpha;

            // alpha = gamma / w.squaredNorm();

            
            double traceSn = Sn.trace(), ww = w.dot(w);
            double logll = 0.0;

            for(int i = 0; i < M; ++i)
                logll += (y(i) == 1 ? std::log(s(i)) : std::log(1.0 - s(i)));

            auto alphaFunc = [&](double a)
            {
                return -logll + 0.5 * (a * ww + traceSn - N * std::log(a));
            };

            Brents brents;

            alpha = brents(alphaFunc, 0.0, 1e3);
            

            if(std::abs(alpha - oldAlpha) < aTol)
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