/** @file
    
    @brief Principal Component Analisys

    @details Algorithm described in section 12.1 of Bishop (2006)
*/

#ifndef CPP_ML_UNSUPERVISED_PCA_H
#define CPP_ML_UNSUPERVISED_PCA_H

#include "../../Modelo.h"
#include "../../Preprocessing/Preprocess.h"
#include "../../SpectraHelpers.h"



/** @brief Reduces the dimensionality of the data by applying a linear projection
    
    @details The PCA (Principal Component Analysis) finds, for some data @c X defined in a @c N dimensional
             space, the @c D < @c N subspace that retains the largest variance of @c X.

             The process is described in details in section 12.1 of Bishop (2006)
*/
struct PCA
{
    /// Receives a single argument in constructor @p D, the number of dimensions to project the data.
    PCA (int D = 2) : D(D)
    {
        assert(D >= 1 && "The dimension to project must have dimension of at least one");
    }


    /** @brief Fits the principal components of the data matrix @p X

        @details The data matrix @p X is taken as value, because we will center it.

                 We also use Spectra to calculate the min{D, M-1, N-1} eigenvectors, because at most
                 min{M-1, N-1} of the eigenvalues will not be zero.

                 If N > M, a special formula is used, described in section 12.1.4 of Bishop (2006)
    */
    PCA& fit (Mat X)
    {
        int M = X.rows(), N = X.cols();

        xMean = X.colwise().mean();

        X = X.rowwise() - xMean.transpose();

        Mat S;


        if(N > M)
            S = (1.0 / N) * X * X.transpose();
    
        else
            S = (1.0 / N) * X.transpose() * X;

            
        TopEigen<> largestEigen(S, std::min(D, std::min(N-1, M-1)));

        U = largestEigen.eigenvectors();

        L = largestEigen.eigenvalues();


        if(N > M)
            U = X.transpose() * U;


        U = U.array().rowwise() / U.colwise().norm().array();


        return *this;
    }


    /// Projects the data matrix @p X (with @c N features) into the #D dimensional subspace spanned by #U
    Mat transform (const Mat& X)
    {
        return X * U;
    }

    /// Projects the @c N dimensional vector @p x into the #D dimensional subspace spanned by #U
    Vec transform (const Vec& x)
    {
        return x * U;
    }


    /** @brief Simply fit(Mat) and then transform(const Mat&)
        @note We can optimize this a bit
    */
    auto fitTransform (const Mat& X)
    {
        return fit(X).transform(X);
    }



    int D;  ///< Number of dimensions of the projected data

    Mat U;  ///< Principal components - largest eigenvectors

    Vec L;  ///< Eigenvalues corresponding to the eigenvectors #U

    Vec xMean;  ///< Mean of the fitted data

};



/** @brief Normalizes the data to have zero mean and unit covariance

    @details This algorithm basically does a PCA to a centered data and scales each projected dimension by
             the inverse square root of the corresponding eigenvalue.

             The algorithm is described in details in section 12.1.3 of Bishop (2006)
*/
struct Whitening : public PCA
{
    using PCA::PCA;

    
    /// We call PCA::fit(Mat) and precompute the inverse square root of the eigenvalues in the vecotr #LInv
    Whitening& fit (const Mat& X)
    {
        PCA::fit(X);

        LInv = Eigen::sqrt(1.0 / L.array());

        return *this;
    }


    /// Subtracts #xMean from the rows of the data @p X and multiplies by #U and #LInv
    Mat transform (const Mat& X)
    {
        return (X.rowwise() - xMean.transpose()) * U * LInv.asDiagonal();
    }

    /// @copybrief transform(const Mat&)
    Vec transform (const Vec& x)
    {
        return (x - xMean.transpose()) * U * LInv.asDiagonal();
    }


    /// Apply fit and then transform
    auto fitTransform (const Mat& X)
    {
        return fit(X).transform(X);
    }


    Vec LInv;   ///< Inverse square root of the eigenvalues #L
};




#endif // CPP_ML_UNSUPERVISED_PCA_H