#include "BayesianLR.h"
//#include "../LinearRegression/LinearRegression.h"

#include <gnuplot-iostream.h>

#include "../../Distributions/Normal/Normal.h"




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



auto gaussianBasis (double x, const vector<double>& mus, double var = 0.1)
{
	Vec z(mus.size());

	FOR(i, mus.size())
		z[i] = exp(-pow(x - mus[i], 2) / var);

	return z;
}

auto gaussianBasis (const Mat& X, const vector<double>& mus, double var = 0.1)
{
	Mat Z(X.rows(), mus.size());

	FOR(i, X.rows())
		Z.row(i) = gaussianBasis(double(X(i, 0)), mus, var);

	return Z;
}



int main ()
{
	Gnuplot gp;


	int N = 5;

	Mat A(N, 1);

	Vec b(N);

	double varG = 0.02;


	auto ab = genPoints(senoid, 0.0, 1.0, N, 0.01, 1);
	auto lins = genLine(senoid, 0.0, 1.0, 1000);


	FOR(i, ab.size())
		A(i, 0) = ab[i].F, b(i) = ab[i].S;

	vector<double> mus(9);
	double aux = 0.1;

	FOR(i, mus.size())
		mus[i] = aux, aux += 0.1;


	// auto gl = genLine([&](double x){ return exp(-pow(x - 0.5, 2) / 0.01); }, 0.0, 1.0, 1000);

	// plot(gl); exit(0);


	A = gaussianBasis(A, mus, varG);


	BayesianLR lr;
	//LinearRegression lr;


	lr.learn(A, b);

	// DB(b.transpose() << "\n\n");
	// DB(lr(A).transpose() << "\n");
	// exit(0);


	int pts = 1000;
	double l = 0.0, u = 1.0, d = (u - l) / pts;

	vector<pd> v(pts);

	FOR(i, pts)
	{
		Vec vx = gaussianBasis(l + i * d, mus, varG);

		//DB(vx.transpose() << "      " << lr(vx));

		v[i] = pd(l + i * d, lr(vx));
	}


	gp << "set xrange[-0.1:1.1]\nset yrange[-1.1:1.1]\n";

	gp << "plot '-' with lines, "
	   << "'-' with lines, "
	   << "'-'\n";

	gp.send1d(lins);
	gp.send1d(v);
	gp.send1d(ab);

	cin.get();


	// auto v = genPoints(senoid, 0.0, 1.0, 1000, 0.01);

	// plot(v, pd(-0.2, 1.2), pd(-1.2, 1.2), false);




	// Mat A(3, 2);
	// Vec b(3);

	// A << 1, 2, 2, 4, 3, 7;
	// b << 10, 2, 1;


	// BayesianLR lr(1e-1, 0.0);

	// lr.learn(A, b);


	// DB(lr(A) << "\n\n");

	// DB(lr.AXw.transpose() << "    " << lr.bias << "\n\n");


	return 0;
}