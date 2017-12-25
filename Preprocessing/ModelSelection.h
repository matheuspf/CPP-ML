#ifndef CPP_ML_MODEL_SELECTION
#define CPP_ML_MODEL_SELECTION

#include "../Modelo.h"


struct SquaredError
{
    inline double operator () (double x, double y) const
    {
        return std::pow(x - y, 2);
    }

    template <class T>
    inline double operator () (const T& x, const T& y) const
    {
        return (x - y).squaredNorm();
    }
};

struct Accuracy
{
    //template <class T>
    inline double operator () (const Veci& x, const Veci& y) const
    {
        return 1.0 - ((x.array() == y.array()).cast<double>().sum() / (x.rows() * x.cols()));
    }
};




struct KFold
{
    KFold (int M, int K = 2, int rngState = 0, bool shuffle = false) :
           M(M), K(K), shuffle(shuffle), gen(rngState), pos(0), step(llround(double(M) / K)), 
           indices(M), trainSet(M - step), testSet(step)
            {
                assert(M >= K && "Number of observations is smaller than number of K-Folds");
                assert(K >= 2 && "Number of K-Folds must be at least 2");

                std::iota(indices.begin(), indices.end(), 0);

                if(shuffle)
                    std::shuffle(indices.begin(), indices.end(), gen);
            }


    auto operator () ()
    {
        std::copy(indices.begin() + pos, indices.begin() + pos + step, testSet.begin());
        std::copy(indices.begin(), indices.begin() + pos, trainSet.begin());
        std::copy(indices.begin() + pos + step, indices.end(), trainSet.begin() + pos);

        if(pos + step == M)
            pos = 0;
        
        else
        {
            pos += step;

            if(pos + step > M)
                pos = M - step;
        }

        return pair<const vector<int>&, const vector<int>&>(trainSet, testSet);
    }


    int M;
    int K;
    bool shuffle;
    std::mt19937 gen;

    vector<int> indices;
    int pos;
    int step;

    vector<int> trainSet;
    vector<int> testSet;
};



template <class Estimator, class Data, class Labels, class Score = SquaredError>
Vec crossValScore (Estimator&& estimator, const Data& X, const Labels& y, int K = 2,
                   Score score = Score(), int rngState = 0, bool shuffle = false)
{
    assert(X.rows() == y.rows() && "Data and Labels have different number of observations.");

    KFold kFold(X.rows(), K, rngState, shuffle);


    Vec scores(K);

    vector<int> trainSet, testSet;

    for(int i = 0; i < K; ++i)
    {
        tie(trainSet, testSet) = kFold();

        auto XTrain = index(X, trainSet);
        auto yTrain = index(y, trainSet);
        auto XTest  = index(X, testSet);
        auto yTest  = index(y, testSet);

        estimator.fit(XTrain, yTrain);

        scores(i) = score(estimator.predict(XTest), yTest);
    }

    return scores;
}




template <class Estimator, class ParamGrid, class Scorer = SquaredError>
struct GridSearchCV
{
    GridSearchCV (const Estimator& estimator, ParamGrid paramGrid, int K = 2, Scorer scorer = Scorer(), int rngState = 0) :
                  estimator(estimator), paramGrid(paramGrid), K(K), scorer(scorer), rngState(rngState) {}


    template <class Data, class Labels>
    void fit (const Data& X, const Labels& y)
    {
        double bestScore = std::numeric_limits<double>::max();

        loopGrid<0>([&](Estimator& est)
                    {
                        double score = crossValScore(est, X, y, K, scorer, rngState).mean();

                        if(score < bestScore)
                        {
                            bestScore = score;
                            bestEstimator = est;
                        }
                    });
    }


    template <int I, class F, std::enable_if_t<(I < std::tuple_size_v<ParamGrid>), int> = 0>
    void loopGrid (F f)
    {
        for(const auto& x : std::get<I>(paramGrid).second)
        {
            std::get<I>(paramGrid).first(estimator, x);

            loopGrid<I+1>(f);
        }
    }

    template <int I, class F, std::enable_if_t<(I == std::tuple_size_v<ParamGrid>), int> = 0>
    void loopGrid (F f)
    {
        Estimator est = estimator;

        f(est);
    }


    Estimator estimator;

    Estimator bestEstimator;

    ParamGrid paramGrid;

    int K;

    Scorer scorer;

    int rngState;
};


template <class Estimator, class ParamGrid, class Scorer = SquaredError,
          enable_if_t<IsSpecialization<ParamGrid, std::tuple>::value, int> = 0>
auto makeGridsearchCV (const Estimator& estimator, ParamGrid paramGrid, int K = 2, Scorer scorer = Scorer())
{
    return GridSearchCV<Estimator, ParamGrid, Scorer>(estimator, paramGrid, K, scorer);
}

template <class Estimator, class ParamGrid, class Scorer = SquaredError,
          enable_if_t<!IsSpecialization<ParamGrid, std::tuple>::value, int> = 0>
auto makeGridsearchCV (const Estimator& estimator, ParamGrid paramGrid, int K = 2, Scorer scorer = Scorer())
{
return GridSearchCV<Estimator, std::tuple<ParamGrid>, Scorer>(estimator, std::make_tuple(paramGrid), K, scorer);
}



#endif //CPP_ML_MODEL_SELECTION