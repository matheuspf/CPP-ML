#ifndef CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS
#define CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS


#include "LogisticRegressionBase.h"



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<>, bool Polymorphic = false>
struct LogisticRegressionTwoClass : public LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>);

        
    void fit (const Mat& X, const Veci& y)
    {
        optimize(X, y, 1e-8);

        intercept = w(N-1);

        w.conservativeResize(N-1);
    }



    int predict (const Vec& x)
    {
        return predictMargin(x) > 0.0;
    }

    Veci predict (const Mat& X)
    {
        return (ArrayXd(predictMargin(X)) > 0.0).cast<int>();
    }



    double predictProb (const Vec& x)
    {
        return sigmoid(predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return sigmoid(predictMargin(X).array());
    }


    double predictMargin (const Vec& x)
    {
        return w.dot(x) + intercept;
    }

    Vec predictMargin (const Mat& X)
    {
        return (X * w).array() + intercept;
    }



    auto optimizeFunc (const Mat& X, const Veci& y)
    {
        static Vec a, s, R, g;

        static Mat H;
        

        return [&](const Vec& w)
        {
            a = X * w;

            s = sigmoid(a.array());

            R = s.array() * (1.0 - s.array());

            g = X.transpose() * (s - y.cast<double>()) + alpha * w;

            H = X.transpose() * R.asDiagonal() * X;

            H.diagonal().array() += alpha;


            return std::tie(g, H);
        };
    }



    void optimize (const Mat& X, const Veci& y, double gTol)
    {
        w = Vec::Constant(N, 0.0);

        auto func = optimizeFunc(X, y);

        Vec g;

        Mat H;


        for(int i = 0; i < 50; ++i)
        {
            std::tie(g, H) = func(w);

            w = w - solveMat(H, g);

            if(g.norm() < gTol)
                break;
        }
    }



    Vec w;
    
    double intercept;
};

} // namespace impl





template <class Regularizer = L2, class Optimizer = Newton<>, bool EncodeLabels = true>
using LogisticRegressionTwoClass = impl::Classifier<impl::LogisticRegressionTwoClass<Regularizer, Optimizer, false>, 
                                                    EncodeLabels>;


namespace poly
{
template <class Regularizer = L2, class Optimizer = Newton<>, bool EncodeLabels = true>
using LogisticRegressionTwoClass = impl::Classifier<impl::LogisticRegressionTwoClass<Regularizer, Optimizer, true>, 
                                                    EncodeLabels>;
}
              



#endif // CPP_ML_LOGISTIC_REGRESSION_TWO_CLASS