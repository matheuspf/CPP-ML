#include "GaussianProcess.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Distributions/Normal/Normal.h"

#include <gnuplot-iostream.h>




void plot (const vector<pd>& v, pd x = pd(1e20, 1e20), pd y = pd(1e20, 1e20), bool lines = true)
{
	Gnuplot gp;

	// if(x.F != 1e20)
	// 	gp << "set xrange[" << x.F << ":" << x.S << "]\n";

	// if(y.F != 1e20)
	// 	gp << "set xrange[" << y.F << ":" << y.S << "]\n";


	//gp << "set xrange[-0.1:1.1]\nset yrange[-1.1:1.1]\n";




	if(lines)
		gp << "plot '-' with lines\n" << endl;
	else
		gp << "plot '-'\n";

	gp.send1d(v);

	cin.get();
}

void plot (const Vec& v, const Vec& z, pd x = pd(1e20, 1e20), pd y = pd(1e20, 1e20), bool lines = true)
{
	vector<pd> p;

	FOR(i, v.rows())
		p.pb(make_pair(v(i), z(i)));

	plot(p, x, y, lines);
}




double senoid (double x)
{
	return sin(2*pi()*x);
}


template <class F>
auto genPoints (F f, double l, double u, int pts, double noise = 0.0, int seed = 0)
{
	vector<pd> v(pts);

	RandDouble rd(seed);

	Normal norm(0.0, noise);


	FOR(i, pts)
	{
		double x = rd(l, u);

		v[i] = pd(x, f(x) + norm());
	}

	return v;
}


template <class F>
auto genLine (F f, double l, double u, int pts)
{
	vector<pd> v(pts);

	double d = (u - l) / pts;

	FOR(i, pts)
		v[i] = pd(l + i * d, f(l + i * d));

	return v;
}



void toyTest ()
{
	Mat A(3, 2);
	Vec b(3);

	A << 1, 2, 2, 4, 3, 7;
	b << 10, 2, 1;


	GaussianProcess<> gp;

	gp.fit(A, b);

	FOR(i, A.rows())
		DB(gp(Vec(A.row(i))));
}


void senoidTest ()
{
	int N = 5;

	Mat A(N, 1);

	Vec b(N);


	auto ab = genPoints(senoid, 0.0, 1.0, N, 0.01, 1);
	auto lins = genLine(senoid, 0.0, 1.0, 1000);


	FOR(i, ab.size())
		A(i, 0) = ab[i].F, b(i) = ab[i].S;


	int pts = 1000;
	double l = 0.0, u = 1.0, d = (u - l) / pts;

	vector<pd> v(pts);

	GaussianProcess<> lr;

	lr.fit(A, b);


	FOR(i, pts)
	{
		Vec vx(1); vx(0) = l + i * d;

		v[i] = pd(l + i * d, lr(vx));
	}


	Gnuplot gp;

	gp << "set xrange[-0.1:1.1]\nset yrange[-1.1:1.1]\n";

	gp << "plot '-' with lines, "
	   << "'-' with lines, "
	   << "'-'\n";

	gp.send1d(lins);
	gp.send1d(v);
	gp.send1d(ab);

	cin.get();
}


// void realTest ()
// {
// 	string path = "reg.txt";



// 	auto [X, y] = getData(path);

// 	auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.3);


// 	GaussianProcess<RBFKernel> gp;

// 	gp.fit(X_train, y_train);
// }



int main ()
{
	//toyTest();

	senoidTest();



	return 0;
}