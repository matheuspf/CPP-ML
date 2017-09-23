#ifndef ML_LOAD_MNIST_H
#define ML_LOAD_MNIST_H

#include "../Modelo.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define READ(x, f) f.read((char*)&x, sizeof(x)), x = Reverse(x)


inline bool is_little_endian() {
    int x = 1;
    return *(char*) &x != 0;
}

template<typename T>
T* reverse_endian(T* p) {
    std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
    return p;
}


namespace detail {

struct mnist_header {
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
};

inline void parse_mnist_header(std::ifstream& ifs, mnist_header& header) {
    ifs.read((char*) &header.magic_number, 4);
    ifs.read((char*) &header.num_items, 4);
    ifs.read((char*) &header.num_rows, 4);
    ifs.read((char*) &header.num_cols, 4);

    if (is_little_endian()) {
        reverse_endian(&header.magic_number);
        reverse_endian(&header.num_items);
        reverse_endian(&header.num_rows);
        reverse_endian(&header.num_cols);
    }
}

inline void parse_mnist_image(std::ifstream& ifs,
    const mnist_header& header,
    float_t scale_min,
    float_t scale_max,
    int x_padding,
    int y_padding,
    vector<double>& dst) {
    const int width = header.num_cols + 2 * x_padding;
    const int height = header.num_rows + 2 * y_padding;

    std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);

    ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);

    dst.resize(width * height, scale_min);

    for (size_t y = 0; y < header.num_rows; y++)
      for (size_t x = 0; x < header.num_cols; x++)
        dst[width * (y + y_padding) + x + x_padding]
        = (image_vec[y * header.num_cols + x] / 255.0) * (scale_max - scale_min) + scale_min;
}

} // namespace detail

/**
 * parse MNIST database format labels with rescaling/resizing
 * http://yann.lecun.com/exdb/mnist/
 *
 * @param label_file [in]  filename of database (i.e.train-labels-idx1-ubyte)
 * @param labels     [out] parsed label data
 **/



inline void parse_mnist_labels(const std::string& label_file, std::vector<size_t> *labels) {
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

    uint32_t magic_number, num_items;

    ifs.read((char*) &magic_number, 4);
    ifs.read((char*) &num_items, 4);

    if (is_little_endian()) { // MNIST data is big-endian format
        reverse_endian(&magic_number);
        reverse_endian(&num_items);
    }

    for (size_t i = 0; i < num_items; i++) {
        uint8_t label;
        ifs.read((char*) &label, 1);
        labels->push_back((size_t) label);
    }
}

/**
 * parse MNIST database format images with rescaling/resizing
 * http://yann.lecun.com/exdb/mnist/
 * - if original image size is WxH, output size is (W+2*x_padding)x(H+2*y_padding)
 * - extra padding pixels are filled with scale_min
 *
 * @param image_file [in]  filename of database (i.e.train-images-idx3-ubyte)
 * @param images     [out] parsed image data
 * @param scale_min  [in]  min-value of output
 * @param scale_max  [in]  max-value of output
 * @param x_padding  [in]  adding border width (left,right)
 * @param y_padding  [in]  adding border width (top,bottom)
 *
 * [example]
 * scale_min=-1.0, scale_max=1.0, x_padding=1, y_padding=0
 *
 * [input]       [output]
 *  64  64  64   -1.0 -0.5 -0.5 -0.5 -1.0
 * 128 128 128   -1.0  0.0  0.0  0.0 -1.0
 * 255 255 255   -1.0  1.0  1.0  1.0 -1.0
 *
 **/
inline void parse_mnist_images(const std::string& image_file,
    std::vector<vector<double>> *images,
    float_t scale_min,
    float_t scale_max,
    int x_padding,
    int y_padding) {

    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    detail::mnist_header header;

    detail::parse_mnist_header(ifs, header);

    for (size_t i = 0; i < header.num_items; i++) {
        vector<double> image;
        detail::parse_mnist_image(ifs, header, scale_min, scale_max, x_padding, y_padding, image);
        images->push_back(image);
    }
}






inline int Reverse (int i);

vector<double> Entrada (const cv::Mat_<uchar>&);
cv::Mat_<uchar> Imagem (const vector<double>&);
vector<double> Objetivo (int);

vector<pair<vector<double>, vector<double>>> loadMnistTrain ();
vector<pair<vector<double>, int>> loadMnistTest ();



vector<pair<vector<double>, vector<double>>> loadMnistTrain ()
{
    vector<vector<double>> imgs;
    vector<size_t> labels;

    parse_mnist_labels("/home/matheus/Algoritmos/Vision/Data/mnist/train-labels.idx1-ubyte", &labels);
    parse_mnist_images("/home/matheus/Algoritmos/Vision/Data/mnist/train-images.idx3-ubyte", &imgs, 0.0, 1.0, 0, 0);

    vector<pair<vector<double>, vector<double>>> r;

    for(int i = 0; i < imgs.size(); ++i)
        r.push_back(make_pair(imgs[i], Objetivo(labels[i])));

    return r;
}


vector<pair<vector<double>, int>> loadMnistTest ()
{
    vector<vector<double>> imgs;
    vector<size_t> labels;

    parse_mnist_labels("/home/matheus/Algoritmos/Vision/Data/mnist/t10k-labels.idx1-ubyte", &labels);
    parse_mnist_images("/home/matheus/Algoritmos/Vision/Data/mnist/t10k-images.idx3-ubyte", &imgs, 0.0, 1.0, 0, 0);

    vector<pair<vector<double>, int>> r;

    for(int i = 0; i < imgs.size(); ++i)
        r.push_back(make_pair(imgs[i], labels[i]));


    return r;
}


cv::Mat_<uchar> Imagem (const vector<double>& v)
{
    cv::Mat_<uchar> img(28, 28);
    int k = 0;

    FOR(i, 28)
    FOR(j, 28)
        img[i][j] = uchar(v[k++] * 255);

    return img;
}


vector<double> Entrada (const cv::Mat_<uchar>& img)
{
    vector<double> v(img.rows * img.cols);
    int k = 0;

    FOR(i, img.rows)
        FOR(j, img.cols)
            v[k++] = double(img[i][j]) / 255.0;

    return v;
}


vector<double> Objetivo (int x)
{
    static vector<double> v[10];

    if(v[0].empty())
    {
        FOR(i, 10)
        {
           v[i] = vector<double>(10);
           v[i][i] = 1.0;
        }
    }

    return v[x];
}



inline int Reverse (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return i = (((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4);
}




#endif // ML_LOAD_MNIST_H