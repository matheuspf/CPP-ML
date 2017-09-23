#ifndef FOREACHMAT_H
#define FOREACHMAT_H

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <thread>
#include <algorithm>
#include <assert.h>



namespace itmat
{

namespace impl
{

template <class F, class... Mats>
constexpr auto chooseFunc (decltype(std::declval<F>()(std::declval<Mats>()(int(), int())...), void())* = nullptr)
{
    return [&](auto g, int i, int j, auto&&... mats){ g(mats(i, j)...); };
}

template <class F, class... Mats>
constexpr auto chooseFunc (decltype(std::declval<F>()(int(), int(), std::declval<Mats>()(int(), int())...), void())* = nullptr)
{
    return [&](auto g, int i, int j, auto&&... mats){ g(i, j, mats(i, j)...); };
}




template <std::size_t N, typename... Args>
constexpr decltype(auto) choose (Args&&... args)
{
    static_assert(N < sizeof...(Args), "Position exceeds maximum number of arguments");

    return std::get<N>(std::tuple<Args...>(std::forward<Args>(args)...));
}


template <std::size_t N>
struct Choose
{
    template <typename T, typename... Args>
    constexpr decltype(auto) operator () (T&&, Args&&... args)
    {
        return Choose<N-1>()(std::forward<Args>(args)...);
    }
};

template <>
struct Choose<0>
{
    template <typename T, typename... Args>
    constexpr decltype(auto) operator () (T&& t, Args&&...)
    {
        return std::forward<T>(t);
    }
};



template <class F, class... Mats>
constexpr std::size_t hasPosition (decltype(std::declval<F>()(std::declval<Mats>()(int(), int())...), void())* = nullptr)
{
    return 0;
}

template <class F, class... Mats>
constexpr std::size_t hasPosition (decltype(std::declval<F>()(int(), int(), std::declval<Mats>()(int(), int())...), void())* = nullptr)
{
    return 1;
}


//------------------------------------------------------------------------------------------

template <bool...>
struct And;

template <bool B1, bool... Bs>
struct And<B1, Bs...> : public And<Bs...> {};

template <bool... Bs>
struct And<false, Bs...> : public std::false_type {};

template <>
struct And<true> : public std::true_type {};


template <bool...>
struct Or;

template <bool B1, bool... Bs>
struct Or<B1, Bs...> : public Or<Bs...> {};

template <bool... Bs>
struct Or<true, Bs...> : public std::true_type {};

template <>
struct Or<false> : public std::false_type {};


//------------------------------------------------------------------------------------


constexpr std::size_t maxThreads = 8;



template <class F, class... Mats>
void forEach (F, int, int, int, int, int, Mats&&...);



#ifdef USING_OPENCV_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


template <class>
struct IsCVMat_ : public std::false_type {};

template <class T>
struct IsCVMat_ <cv::Mat_<T>> : public std::true_type {};


#ifdef USING_MATRIX

template <class T>
struct IsCVMat_<wrp::Matrix<cv::Mat_<T>>> : public std::true_type {};

#endif   // USING_MATRIX


template <class T>
constexpr bool isCVMat_ = IsCVMat_<T>::value;


#endif     // USING_OPENCV_



}   // namespace impl



//===========================================================================================


template <class Mat, class F>
inline void forEach (Mat&& mat, F f, int numThreads = 1,
                     int startRow = -1, int startCol = -1, int lastRow = -1, int lastCol = -1)
{
    impl::forEach(f, numThreads, startRow, startCol, lastRow, lastCol, std::forward<Mat>(mat));
}

template <class Mat1, class Mat2, class F>
inline void forEach (Mat1&& mat1, Mat2&& mat2, F f, int numThreads = 1,
                     int startRow = -1, int startCol = -1, int lastRow = -1, int lastCol = -1)
{
    impl::forEach(f, numThreads, startRow, startCol, lastRow, lastCol, std::forward<Mat1>(mat1), std::forward<Mat2>(mat2));
}

template <class Mat1, class Mat2, class Mat3, class F>
inline void forEach (Mat1&& mat1, Mat2&& mat2, Mat3&& mat3, F f, int numThreads = 1,
                     int startRow = -1, int startCol = -1, int lastRow = -1, int lastCol = -1)
{
    impl::forEach(f, numThreads, startRow, startCol, lastRow, lastCol, std::forward<Mat1>(mat1), std::forward<Mat2>(mat2), std::forward<Mat3>(mat3));
}


template <class F, class... Mats>
inline void forEach (F f, int startRow = -1, int startCol = -1, int lastRow = -1, int lastCol = -1, Mats&&... mats)
{
    impl::forEach(f, 1, startRow, startCol, lastRow, lastCol, std::forward<Mats>(mats)...);
}

template <class F, class... Mats>
inline void forEach (F f, int numThreads, int startRow = -1, int startCol = -1, int lastRow = -1, int lastCol = -1, Mats&&... mats)
{
    impl::forEach(f, numThreads, startRow, startCol, lastRow, lastCol, std::forward<Mats>(mats)...);
}


//===========================================================================================




namespace impl
{


template <class F, class... Mats>
void forEach (F f, int numThreads, int startRow, int startCol, int lastRow, int lastCol, Mats&&... mats)
{
    assert((numThreads <= maxThreads) && "Maximum number of threads exceeded");


    const std::array<int, sizeof...(Mats)> matsRows = { mats.rows... };
    const std::array<int, sizeof...(Mats)> matsCols = { mats.cols... };

    assert(std::all_of(matsRows.begin(), matsRows.end(), [&](auto x){ return x == matsRows[0]; }) &&
           std::all_of(matsRows.begin(), matsRows.end(), [&](auto x){ return x == matsRows[0]; }) &&
           "All Mats_ must be equal in size");

    int rows = matsRows[0];
    int cols = matsCols[0];


    startRow = startRow == -1 ? 0    : startRow;
    lastRow  = lastRow  == -1 ? rows : lastRow;

    startCol = startCol == -1 ? 0    : startCol;
    lastCol  = lastCol  == -1 ? cols : lastCol;


    std::array<std::thread, impl::maxThreads> threads;


    int blockSize = (lastRow - startRow) / numThreads;


    auto threadFunction = [&](int start, int last, auto g)
    {
        for(int i = start; i < last; ++i)
            for(int j = startCol; j < lastCol; ++j)
                g(f, i, j);
    };


    //auto apply = chooseFunc<F, Mats...>();

    constexpr std::size_t hasPos = hasPosition<F, Mats...>();

    auto apply = Choose<hasPos>()([&](auto g, int i, int j){ g(std::forward<Mats>(mats)(i, j)...); }, [&](auto g, int i, int j){ g(i, j, mats(i, j)...); });

    //auto apply = choose(0, [&](auto g, int i, int j){ g(mats(i, j)...); }, [&](auto g, int i, int j){ g(i, j, mats(i, j)...); });


    int start = startRow, last = lastRow;

    for(int i = 0; i < numThreads - 1; ++i)
    {
        int aux = start + blockSize;

        threads[i] = std::thread(threadFunction, start, aux, apply);

        start = aux;
    }


    threadFunction(start, last, apply);

    for(int i = 0; i < numThreads - 1; ++i)
        threads[i].join();
}




//===================================================================================================


#ifdef USING_OPENCV_


template <class F, int... Is, class... Mats>
std::enable_if_t<impl::And<impl::isCVMat_<std::decay_t<Mats>>...>::value,
void> forEach (F f, int numThreads, sequence<Is...>, Mats&&... mats)
{
    assert((numThreads <= maxThreads) && "Maximum number of threads exceeded");


    const std::array<int, sizeof...(Mats)> matsRows = { mats.rows... };
    const std::array<int, sizeof...(Mats)> matsCols = { mats.cols... };

    assert(std::all_of(matsRows.begin(), matsRows.end(), [&](auto x){ return x == matsRows[0]; }) &&
           std::all_of(matsRows.begin(), matsRows.end(), [&](auto x){ return x == matsRows[0]; }) &&
           "All Mats_ must be equal in size");

    int rows = matsRows[0];
    int cols = matsCols[0];


    const std::array<bool, sizeof...(Mats)> matsContinuous = { mats.isContinuous()... };

    assert(std::all_of(matsContinuous.begin(), matsContinuous.end(), [](auto x){ return x; }) &&
           "All Mats_ must be continuous");



    std::array<std::thread, impl::maxThreads> threads;


    std::size_t totalSize = rows * cols;

    std::size_t blockSize = totalSize / numThreads;


    auto threadFunction = [&](int start, int last, auto f)
    {
        auto iterators = std::make_tuple((reinterpret_cast<typename std::decay_t<Mats>::value_type*>(mats.data) + start - 1)...);

        int r = start / cols;
        int c = start % cols;

        for(; start != last; ++start, c = (c + 1) % cols, r += !c)
            f(r, c, (*std::get<Is>(iterators)++)...);
    };


    int start = 0, last = rows * cols;

    for(int i = 0; i < numThreads - 1; ++i)
    {
        int aux = start + blockSize;

        threads[i] = std::thread(threadFunction, start, aux, f);

        start = aux;
    }


    threadFunction(start, last, f);

    for(int i = 0; i < numThreads - 1; ++i)
        threads[i].join();
}

#endif  // USING_OPENCV_


} // namespace impl


} // namespace itmat


#endif  // FOREACHMAT_H
