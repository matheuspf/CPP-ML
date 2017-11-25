#ifndef CPP_ML_GENERATIVE_MODEL_H
#define CPP_ML_GENERATIVE_MODEL_H

#include "../../Modelo.h"

#include "../../Distributions/Gaussian/Gaussian.h"

#include "../../Distributions/Multinomial/Multinomial.h"




template <class ClassConditional>
struct GenerativeModel
{
    template <typename T>
    GenerativeModel& fit (const MatX<T>& X, Veci y, bool sharedVariance = false)
    {
        M = X.rows(), N = X.cols();

        classMapping(y);

        classPrior.params(classCount);

        classConditionals.resize(numClasses);

        
        for(int k = 0; k < numClasses; ++k)
        {
            MatX<T> Xk(classCount[k], N);

            for(int i = 0, j = 0; i < M; ++i) if(y(i) == k)
                Xk.row(j++) = X.row(i);

            classConditionals[k].fit(Xk);
        }


        return *this;
    }


    template <typename T>
    int predict (const VecX<T>& x)
    {
        int label = 0;
        double bestPosterior = std::numeric_limits<double>::min();

        //cout << x.transpose() << "        ";

        for(int k = 0; k < numClasses; ++k)
        {
            double posterior = classPrior(k) * classConditionals[k](x);

            //cout << classPrior(k) << "   " << classConditionals[k](x) << "              ";

            if(posterior > bestPosterior)
            {
                bestPosterior = posterior;
                label = k;
            }
        }//db("\n");

        return classReverseMap[label];
    }


    template <typename T>
    Veci predict (const MatX<T>& X)
    {
        Veci res(X.rows());

        for(int i = 0; i < res.rows(); ++i)
            res(i) = predict(VecX<T>(X.row(i)));
        
        return res;
    }


    void classMapping (Veci& y)
    {
        classMap.clear();
        classReverseMap.clear();
        classCount.clear();
        numClasses = 0;

        for(int i = 0; i < y.rows(); ++i)
        {
            if(classMap.find(y(i)) == classMap.end())
            {
                classMap[y(i)] = numClasses++;

                classCount.push_back(0);
            }

            int label = classMap[y(i)];

            classReverseMap[label] = y(i);
            
            y(i) = label;
            
            classCount[y(i)]++;
        }
    }




    int M, N;

    std::vector<ClassConditional> classConditionals;

    Multinomial classPrior;


    std::unordered_map<int, int> classMap;
    std::unordered_map<int, int> classReverseMap;

    std::vector<int> classCount;

    int numClasses;
};



#endif // CPP_ML_GENERATIVE_MODEL_H