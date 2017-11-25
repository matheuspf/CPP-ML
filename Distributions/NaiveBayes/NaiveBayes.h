#ifndef CPP_ML_DISTRIBUTIONS_NAIVE_BAYES_H
#define CPP_ML_DISTRIBUTIONS_NAIVE_BAYES_H

#include "../../Modelo.h"



template <class Distribution>
struct NaiveBayes
{
    NaiveBayes () {}

    NaiveBayes (const std::vector<Distribution>& distributions) :
                distributions(distributions), dimensions(distributions.size()) {}

    
    template <typename T>
    void fit (const MatX<T>& X)
    {
        int M = X.rows(), N = X.cols();

        dimensions = N;

        distributions.resize(dimensions);

        for(int i = 0; i < dimensions; ++i)
            distributions[i].fit(X.col(i));
    }



    auto params ()
    {
        return distributions;
    }


    template <typename T>
    double operator () (const VecX<T>& x)
    {
        return std::inner_product(distributions.begin(), distributions.end(), std::begin(x), 1.0,
                                  std::multiplies<double>(), [&](auto& dist, const auto& val)
                                  {
                                      return dist(val);
                                  });
    }

    template <typename T>
    Vec operator () (const MatX<T>& X)
    {
        Vec probs(X.rows());

        for(int i = 0; i < probs.size(); ++i)
            probs(i) = operator()(VecX<T>(X.row(i)));

        return probs;
    }

    
    auto operator () ()
    {
        VecX<std::decay_t<decltype(distributions[0]())>> x(dimensions);

        std::transform(distributions.begin(), distributions.end(), std::begin(x), [](auto& dist)
        {
            return dist();
        });

        return x;
    }




    std::vector<Distribution> distributions;

    int dimensions = 0;
};




#endif // CPP_ML_DISTRIBUTIONS_NAIVE_BAYES_H