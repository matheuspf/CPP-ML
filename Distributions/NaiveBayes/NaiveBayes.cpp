#include "NaiveBayes.h"

#include "../../Classification/GenerativeModel/GenerativeModel.h"

#include "../Normal/Normal.h"
#include "../BernoulliBayes/BernoulliBayes.h"

#include <gnuplot-iostream.h>



Mat genData (int M, const std::vector<double>& devs, int rng = 0)
{
    int N = devs.size();

    Mat X(M, N);

    std::mt19937 gen(rng);


    for(int i = 0; i < M; ++i)
        for(int j = 0; j < N; ++j)
            X(i, j) = std::normal_distribution<double>(0.0, devs[j])(gen);
    

    return X;
}



template <typename Dist>
void plot3d (Dist& dist, int l = -4, int u = 4, int p = 500)
{
    std::vector<std::tuple<double, double, double>> plot;

	double d = (u - l) / double(p);


	for(int i = 0; i < p; ++i) for(int j = 0; j < p; ++j)
	{
		Vec x(2);
        
        x << l + i * d, l + j * d;

		plot.emplace_back(l + i * d, l + j * d, dist(x));
	}
    

    Gnuplot gp;

	gp << "set pm3d\n";
	gp << "set dgrid3d 30,30\n";
	gp << "splot '-' u 1:2:3 with lines\n";

	gp.send1d(plot);


    cin.get();
}


void diagonalGaussian ()
{
    NaiveBayes<Normal> nb;

    Mat X = genData(1e3, {0.5, 0.5});

    nb.fit(X);

    plot3d(nb);
}


auto read (string str)
{
	ifstream in(str);

	int rows, cols;

	in >> rows >> cols;

	in.ignore();


	vector<string> tokens;

	getline(in, str);

	stringstream ss(str);

	while(ss >> str) tokens.pb(str);


    int i, j;
	double val;


	Veci labels(rows);

	FOR(i, rows)
    {
        in >> val;
        labels(i) = val;
    }

	in.ignore();


	// vector<Triplet<double>> trips;

	// SparseMatrix<double> mat(rows, cols);

	// while(in >> i >> j >> val) trips.pb(Triplet<double>(i-1, j-1, val));

	// mat.setFromTriplets(ALL(trips));


    Mati mat = Mati::Constant(rows, cols, 0);

    while(in >> i >> j >> val)
        mat(i-1, j-1) = val;


	return make_tuple(tokens, labels, mat);
}


int main ()
{
    auto [tokens1, y_train, X_train] = read("/home/matheus/Algoritmos/Machine Learning/spam_data/dataTrain50.txt");
    auto [tokens2, y_test, X_test] = read("/home/matheus/Algoritmos/Machine Learning/spam_data/dataTest.txt");


    GenerativeModel<NaiveBayes<BernoulliBayes>> nb;

    nb.fit(X_train, y_train);

    Veci y_pred = nb.predict(X_test);

    db("Accuracy:   ", (y_pred.array() == y_test.array()).cast<double>().sum() / y_test.rows());


	return 0;
}