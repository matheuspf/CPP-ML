#ifndef ML_PREPROCESS_H
#define ML_PREPROCESS_H

#include "../Modelo.h"


Mat readMat (const string& fName, char delimiter = ' ')
{
	ifstream file(fName);


	int M = 0, N = 0;

	vector<double> buf;
	
	string str;


	while(getline(file, str))
	{
        stringstream ss(str);
        string value;
        
        int cnt = 0;

        while(getline(ss, value, delimiter))
            buf.push_back(stod(value)), cnt++;

		N = N == 0 ? cnt : N;
        ++M;
	}


	Mat X = Eigen::Map<Mat>(&buf[0], N, M);


	return X.transpose();
}



template <class MatType, class VecType>
auto trainTestSplit (const MatType& X, const VecType& y, double split = 0.3, int rng = 0)
{
	assert(X.rows() == y.rows() && split > 0 && split < 1.0);


	int M = X.rows(), N = X.cols();


	vector<int> perm(X.rows());

	iota(ALL(perm), 0);
	shuffle(ALL(perm), mt19937(rng));


	int trainM = round((1.0 - split) * M);
	int testM = M - trainM;

	MatType Xtrain(trainM, N), Xtest(testM, N);
	VecType ytrain(trainM), ytest(testM);


	for(int i = 0; i < trainM; ++i)
	{
		Xtrain.row(i) = X.row(perm[i]);
		ytrain(i) = y(perm[i]);
	}

	for(int i = 0; i < testM; ++i)
	{
		Xtest.row(i) = X.row(perm[i + trainM]);
		ytest(i) = y(perm[i + trainM]);
	}


	return make_tuple(Xtrain, ytrain, Xtest, ytest);
}



struct Normalize
{
	Normalize& fit (const Mat& X, ...)
	{
		l = X.colwise().minCoeff().transpose();
		u = X.colwise().maxCoeff();

		return *this;
	}

	Mat transform (const Mat& X)
	{
		return (X.rowwise() - l.transpose()).array().rowwise() / (u - l).transpose().array();
	}

	Mat fitTransform (const Mat& X, ...)
	{
		return fit(X).transform(X);
	}


	Vec l;
	Vec u;
};





struct Standardize
{
	Standardize& fit (const Mat& X, ...)
	{
		M = X.rows();

		mean = X.colwise().mean();
		dev = sqrt(pow((X.rowwise() - mean.transpose()).array(), 2).colwise().sum() / double(X.rows()));
		
		FOR(i, dev.rows()) dev(i) = max(1e-8, dev(i));

		return *this;
	}

	Mat transform (const Mat& X)
	{
		return (X.rowwise() - mean.transpose()).array().rowwise() / dev.transpose().array();
	}

	Mat fitTransform (const Mat& X, ...)
	{
		return fit(X).transform(X);
	}


	Vec mean;
	Vec dev;

	int M;
};



struct OneHotEncoding
{
	OneHotEncoding (vector<int> indices_ = vector<int>()) :
					indices(indices_), numClasses(indices_.size()), classMap(indices_.size())
	{
		sort(indices.begin(), indices.end());
	}


	OneHotEncoding& fit (const Mat& X)
	{
		int M = X.rows(), N = X.cols();

		for(int i = 0; i < X.rows(); ++i)
		{
			for(int j = 0; j < indices.size(); ++j)
			{
				int x = llround(X(i, indices[j]));

				if(!contains(classMap[j], x))
					classMap[j][x] = numClasses[j]++;
			}
		}

		P = accumulate(numClasses.begin(), numClasses.end(), 0) + N - indices.size();

		return *this;
	}

	Mat transform (const Mat& X)
	{
		int M = X.rows(), N = X.cols(), k = 0;

		Mat Z = Mat::Constant(M, P, 0.0);

		for(int j = 0; j < indices.size(); ++j)
		{
			for(int i = 0; i < X.rows(); ++i)
				Z(i, k + classMap[j][llround(X(i, indices[j]))]) = 1.0;

			k += numClasses[j];
		}

		for(int j = 0, l = 0; j < N; ++j) if(!binary_search(indices.begin(), indices.end(), j))
		{
			for(int i = 0; i < X.rows(); ++i)
				Z(i, k + l) = X(i, j);

			++l;
		}

		return Z;
	}

	Mat fitTransform (const Mat& X)
	{
		return fit(X).transform(X);
	}


	vector<int> indices;

	vector<int> numClasses;

	vector<unordered_map<int, int>> classMap;

	int P;

};




Mat polyExpansion (const Mat& X, int degree = 2)
{
	int M = X.rows(), N = X.cols(), P = N * degree;

	//Mat Z(M, P + degree * (N * (N - 1)) / 2);
	Mat Z(M, P + N * ((N - 1)) / 2);


	for(int i = 0; i < M; ++i)
		for(int j = 0; j < N; ++j)
			for(int k = 0; k < degree; ++k)
				Z(i, degree * j + k) = pow(X(i, j), k+1);

	// for(int i = 0; i < M; ++i)
	// 	for(int j = 0, l = 0; j < N; ++j)
	// 		for(int k = j+1; k < N; ++k, ++l)
	// 			Z(i, P + l) = X(i, j) * X(i, k);

	// for(int i = 0; i < M; ++i)
	// 	for(int a = 0, l = 0; a < degree; ++a)
	// 		for(int j = 0; j < N; ++j)
	// 			for(int k = j + 1; k < N; ++k, ++l)
	// 				Z(i, P + l) = pow(X(i, j), a+1) * pow(X(i, k), a+1);


	return Z;
}


template <class T = int>
struct LabelEncoder
{
	LabelEncoder& fit (const VecX<T>& x)
	{
		K = 0;
		labelSet.clear();
		reverseMap.clear();

		std::for_each(std::begin(x), std::end(x), [&](const T& t)
		{
			if(labelSet.find(t) == labelSet.end())
				labelSet[t] = K++;
			
			reverseMap[labelSet[t]] = t;
		});

		return *this;
	}


	VecX<T> transform (const VecX<T>& x, std::vector<int> labels = std::vector<int>())
	{
		if(labels.empty())
		{
			labels.resize(K);
			std::iota(labels.begin(), labels.end(), 0);
		}

		assert(K == labels.size() && "Number of labels given does not match the number of labels in the data.");


		Veci y(x.rows());

		for(int i = 0; i < x.rows(); ++i)
		{
			auto it = labelSet.find(x(i));

			assert(it != labelSet.end() && "Invalid label found in the data.");

			y(i) = labels[it->second];
		}

		return y;
	}


	VecX<T> fitTransform (const VecX<T>& x, const std::vector<int>& labels = std::vector<int>())
	{
		return fit(x).transform(x, labels);
	}


	int K;

	std::unordered_map<T, int> labelSet;
	std::unordered_map<int, T> reverseMap;
};






#endif //ML_PREPROCESS_H