#ifndef CPP_ML_STUMP_H
#define CPP_ML_STUMP_H

#include "../Classifier.h"

#include "../../ZipIter/ZipIter.h"


namespace impl
{

template <bool EncodeLabels = true, bool Polymorphic = false>
struct Stump : public PickClassifierBase<Stump<EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>
{
    USING_CLASSIFIER(PickClassifierBase<Stump<EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>)

    Stump () : BaseClassifier(1, -1, false) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        fit_(X, y, Vec::Constant(M, 1.0 / M));
    }

    void fit_ (Mat X, const Veci& y, const Vec& prob)
    {
        X.transposeInPlace();


        double bestCost = 1e20;

        std::vector<int> ids(M);


        for(int i = 0; i < N; ++i)
        {
            std::iota(ids.begin(), ids.end(), 0);

            std::sort(it::zipIter(&X.row(i)(0), ids.begin()), it::zipIter(&X.row(i)(0) + M, ids.end()));

            for(int dir = 1; dir <= 1; dir += 2)
            {
                double cost = std::accumulate(ids.begin(), ids.end(), 0.0, [&](double sum, int id)
                {
                    return sum + (y(id) == -dir) * prob(id);
                });


                if(cost < bestCost)
                {
                    bestCost = cost;
                    index = i;
                    stump = X(i, 0) - 1e-8;
                    direction = dir;
                }

                
                for(int j = 0; j < M; ++j)
                {
                    cost += (y(ids[j]) == dir ? 1.0 : -1.0) * prob(ids[j]);

                    if(cost < bestCost)
                    {
                        bestCost = cost;
                        index = i;
                        stump = X(i, j) + 1e-8;
                        //stump = (j == M-1) ? X(i, j) + 1e-8 : X(i, j+1) - 1e-8;
                        direction = dir;
                    }
                }
            }
        }
    }


    int predict_ (const Vec& x)
    {
        return direction * (x(index) - stump >= 0.0 ? positiveClass : negativeClass);
    }


    Veci predict_ (const Mat& X)
    {
        Veci pred(X.rows());

        for(int i = 0; i < X.rows(); ++i)
            pred(i) = direction * (X(i, index) - stump >= 0.0 ? positiveClass : negativeClass);

        return pred;
    }



    int index;

    double stump;

    int direction;
};

} // namespace impl



template <bool EncodeLabels = true>
using Stump = impl::Stump<EncodeLabels, false>;


namespace poly
{

template <bool EncodeLabels = true>
using Stump = impl::Stump<EncodeLabels, true>;

}


#endif // CPP_ML_STUMP_H