#ifndef ML_PREPROCESS_H
#define ML_PREPROCESS_H

#include "../Modelo.h"


Mat readMat (const std::string& fName, char delimiter = ' ')
{
	std::ifstream file(fName);

	std::vector<double> buffer;

	std::vector<std::map<std::string, int>> stringMap;
	std::vector<int> classCount;

	std::string str;

	int M = 1, N = 0;


	std::getline(file, str);

    std::stringstream ss(str);
    std::string value;

	while(std::getline(ss, value, delimiter))
	{
		++N;
		stringMap.push_back(std::map<std::string, int>{});
		classCount.push_back(0);

		try
		{
			double dVal = std::stod(value);
			buffer.push_back(dVal);
		}
		catch(const std::exception& except)
		{
			stringMap[N-1][value] = 0;
			classCount[N-1] = 1;
			buffer.push_back(0.0);
		}
	}


	while(std::getline(file, str))
	{
        ss = std::stringstream(str);

		int i = 0;
        
        while(std::getline(ss, value, delimiter))
		{
			if(classCount[i] > 0)
			{
				auto it = stringMap[i].find(value);

				if(it == stringMap[i].end())
				{
					stringMap[i][value] = classCount[i];
					classCount[i]++;
				}

				buffer.push_back(stringMap[i][value]);
			}

			else
			{
            	buffer.push_back(std::stod(value));
			}
			
			++i;
		}

        ++M;
	}


	Mat X = Eigen::Map<Mat>(&buffer[0], N, M);


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
		std::sort(indices.begin(), indices.end());
	}


	OneHotEncoding& fit (const Mat& X)
	{
		int M = X.rows(), N = X.cols();

		if(indices.empty())
		{
			indices.resize(N);
			numClasses.resize(N);
			classMap.resize(N);
			std::iota(indices.begin(), indices.end(), 0);
		}

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




Mat polyExpansion (const Mat& X, int degree = 2, bool correlation = false)
{
	int M = X.rows(), N = X.cols(), P = N * degree;

	Mat Z;

	if(correlation)
		Z = Mat(M, N * degree + (N * (N - 1)) / 2);
	
	else
		Z = Mat(M, N * degree);


	for(int i = 0; i < M; ++i)
		for(int j = 0; j < N; ++j)
			for(int k = 0; k < degree; ++k)
				Z(i, degree * j + k) = pow(X(i, j), k+1);


	if(correlation)
	{
		for(int i = 0; i < M; ++i)
			for(int j = 0, l = 0; j < N; ++j)
				for(int k = j+1; k < N; ++k, ++l)
					Z(i, N * degree + l) = X(i, j) * X(i, k);
	}

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
	LabelEncoder& fit (const VecX<T>& x, bool countClasses = false)
	{
		numClasses = 0;
		labelSet.clear();
		labelMap.clear();

		std::for_each(std::begin(x), std::end(x), [&](const T& t)
		{
			labelSet.insert(t);
		});

		
		for(const auto& t : labelSet)
			labelMap[t] = numClasses++;



		return *this;
	}


	template <typename... Labels, std::enable_if_t<And_v<std::is_integral_v<std::decay_t<Labels>>...>, int> = 0>
	VecX<T> transform (const VecX<T>& x, Labels... labels)
	{
		return transform(x, std::vector<int>{labels...});
	}


	VecX<T> transform (const VecX<T>& x, std::vector<int> labels = std::vector<int>())
	{
		if(labels.empty())
		{
			labels.resize(numClasses);
			std::iota(labels.begin(), labels.end(), 0);
		}

		assert(numClasses == labels.size() && "Number of labels given does not match the number of labels in the data.");


		reverseMap.clear();

		for(auto it = labelSet.begin(), k = 0; k < numClasses; ++it, ++k)
			reverseMap.emplace(labels[k], *it);


		Veci y(x.rows());

		for(int i = 0; i < x.rows(); ++i)
		{
			auto it = labelMap.find(x(i));

			assert(it != labelMap.end() && "Invalid label found in the data.");

			y(i) = labels[it->second];
		}

		return y;
	}


	template <typename... Labels>
	VecX<T> fitTransform (const VecX<T>& x, Labels&&... labels)
	{
		return fit(x).transform(x, std::forward<Labels>(labels)...);
	}


	int decode (const T& t)
	{
		return reverseMap[t];
	}


	
	std::vector<int> countClasses (const Veci& x)
	{
		std::map<int, int> classMap;

		std::for_each(std::begin(x), std::end(x), [&](const T& t){ classMap[t]++; });

		std::vector<int> classCount(classMap.size());

        std::transform(classMap.begin(), classMap.end(), classCount.begin(), [](const auto& p){ return p.second; });

		return classCount;
	}



	int numClasses;

	std::set<T> labelSet;

	std::unordered_map<T, int> labelMap;

	std::unordered_map<int, T> reverseMap;
};





auto pickTarget (const Mat& X, int pos = 0)
{
	Mat Z;
	Veci y;

	if(pos == 0)
	{
		y = X.col(0).cast<int>();
    	Z = X.block(0, 1, X.rows(), X.cols()-1);
	}

	else
	{
		y = X.col(X.cols()-1).cast<int>();
		Z = X.block(0, 0, X.rows(), X.cols()-1);
	}

	return std::make_tuple(Z, y);
}




#endif //ML_PREPROCESS_H