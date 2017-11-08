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

    void fit_ (const Mat& X_, const Veci& y_, const Vec& prob_)
    {
        Mat X = X_;
        Veci y = y_;
        Vec prob = prob_;

        X.transposeInPlace();


        double bestCost = 1e20;

        std::vector<int> ids(M);

        Vec x;


        auto att = std::tie(bestCost, index, stump, direction);


        for(int i = 0; i < N; ++i)
        {
            x = X.row(i);

            std::iota(ids.begin(), ids.end(), 0);

            std::sort(it::zipIter(std::begin(x), ids.begin()), it::zipIter(std::end(x), ids.end()));


            double posCost = std::accumulate(ids.begin(), ids.end(), 0.0, [&](double sum, int id)
            {
                return sum + (y(id) == -1) * prob(id);
            });

            double negCost = 1.0 - posCost;
            

            if(posCost < bestCost)
                att = std::make_tuple(posCost, i, x(0) - 1e-8, 1);

            else if(negCost < bestCost)
                att = std::make_tuple(negCost, i, x(0) - 1e-8, -1);
            

            for(int j = 0; j < M; ++j)
            {
                posCost += (y(ids[j]) == 1 ? 1.0 : -1.0) * prob(ids[j]);

                negCost = 1.0 - posCost;

                if(posCost < bestCost)
                    att = std::make_tuple(posCost, i, x(j) + 1e-8, 1);
                
                else if(negCost < bestCost)
                    att = std::make_tuple(negCost, i, x(j) + 1e-8, -1);
            }

        }

        db(bestCost, "\n");
    }


    int predict_ (const Vec& x)
    {
        return direction * (x(index) > stump ? positiveClass : negativeClass);
    }


    Veci predict_ (const Mat& X)
    {
        Veci pred(X.rows());

        for(int i = 0; i < X.rows(); ++i)
            pred(i) = direction * (X(i, index) > stump ? positiveClass : negativeClass);

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