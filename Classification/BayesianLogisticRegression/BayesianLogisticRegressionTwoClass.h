#ifndef CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H
#define CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H

#include "../Classifier.h"

#include "../LogisticRegression/LogisticRegressionTwoClass.h"



namespace impl
{

template <bool EncodeLabels = true, bool Polymorphic = false>
struct BayesianLogisticRegressionTwoClass : virtual public PickClassifierBase<BayesianLogisticRegressionTwoClass<EncodeLabels, Polymorphic>, 
                                                                      EncodeLabels, Polymorphic>,
                                            public LogisticRegressionTwoClass<L2, Newton<>, EncodeLabels, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<BayesianLogisticRegressionTwoClass<EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);
    USING_LOGISTIC_REGRESSION(LogisticRegressionTwoClass<L2, Newton<>, EncodeLabels, Polymorphic>);

    using BaseLogisticRegression::w, BaseLogisticRegression::intercept, BaseLogisticRegression::predict_, 
          BaseLogisticRegression::predictMargin, BaseLogisticRegression::optimizeFunc;


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

        Vec g;        

        auto func = optimizeFunc(X, y);

        
        for(int iter = 0; iter < maxIter; ++iter)
        {
            std::tie(g, Sn) = func(w);

            w = w - solveMat(Sn, g);
            

            Sn.diagonal().array() -= alpha;

            Eigen::EigenSolver<Mat> eigSolver(Sn);

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


    template <typename T>
    auto kappa (const T& sigma)
    {
        return sqrt(1.0 + pi() * (sigma / 8.0));
    }



    Mat Sn;

};


}



template <bool EncodeLabels = true>
using BayesianLogisticRegressionTwoClass = impl::BayesianLogisticRegressionTwoClass<EncodeLabels, false>;


namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegressionTwoClass = impl::BayesianLogisticRegressionTwoClass<EncodeLabels, true>;
    
}







#endif // CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H