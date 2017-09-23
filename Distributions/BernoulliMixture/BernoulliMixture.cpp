#include "BernoulliMixture.h"

#include "../../Preprocessing/LoadMnist.h"


int main ()
{
	BernoulliMixture dist;

	vector<pair<vector<double>, vector<double>>> mnistTrain = loadMnistTrain();


	int M = mnistTrain.size(), N = mnistTrain[0].F.size(), K = 10;

	
	Mati X(M, N);


	FOR(i, M) FOR(j, N)
		X(i, j) = mnistTrain[i].F[j] >= 0.5;


	dist.fit(X, K);




	// FOR(k, K)
	// {
	// 	vector<double> v(N);

	// 	FOR(i, N)
	// 		v[i] = dist.mu(k, i);


	// 	auto img = Imagem(v);


	// 	// cv::resize(img, img, cv::Size(), 2.0, 2.0);
	// 	// cv::imshow("w", img);
	// 	// cv::waitKey(0);

	// 	cv::imwrite((string("imgs/") + to_string(k) + string(".png")).c_str(), img);  
	// }


	while(1)
	{
		Vec x = dist();

		vector<double> v(x.data(), x.data() + N);

		auto img = Imagem(v);

		cv::imshow("w", img);
		cv::waitKey(0);
	}



	return 0;
}