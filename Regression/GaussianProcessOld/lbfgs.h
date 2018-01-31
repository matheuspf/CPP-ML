#ifndef ML_GRADIENT_DESCENT_H
#define ML_GRADIENT_DESCENT_H

#include <bits/stdc++.h>

#include <type_traits>
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>



template <class F>
double lineSearch (const Eigen::MatrixXd& x, F f, const Eigen::MatrixXd& grad, double alpha = 0.5, double beta = 0.01, double minStep = 1e-6)
{
	if(grad.norm() < minStep)
		return 0.0;


	double f0 = f(x);

	//double delta = beta * grad.norm();
	double delta = grad.norm();

	double step = 1.0;

	while(step >= minStep)
	{
		double fy = f(x - step * grad);

		if(fy < f0 - (step / 2.0) * delta)
			break;

		step *= alpha;
	}

	return step;
}




// double interpolate (double a, double b)
// {
// 	return (a + b) / 2.0;
// }


inline double interpolate (double a, double fa, double ga, double b, double fb)
{
	return - (ga * b * b) / (2.0 * (fb - fa - ga * b));
}


// inline double interpolate (double a, double fa, double ga, double b, double fb)
// {
// 	return (a + b) / 2.0;
// }



template <class F, class G>
double zoom (double alo, double ahi, const Eigen::VectorXd& x, const Eigen::VectorXd& d, F f, G g, 
			 double f0, double g0, double c1 = 1e-4, double c2 = 0.9, double aMax = 5.0, int maxIter = 10)
{
	double aj;

	double flo = f(x + alo * d);
	double fhi = f(x + ahi * d);
	double glo = g(x + alo * d).transpose() * d;


	while(maxIter--)
	{
		//DB(alo << "   " << flo << "   " << glo << "   " << ahi << "   " << fhi);

		aj = interpolate(alo, flo, glo, ahi, fhi);

		//DB(aj);

		double fj = f(x + aj * d);

		if(fj > f0 + c1 * aj * g0 || fj >= flo)
		{
			ahi = aj;
			fhi = fj;
		}

		else
		{
			double gj = g(x + aj * d).transpose() * d;

			if(std::abs(gj) <= -c2 * g0)
				return aj;

			if(gj * (ahi - alo) >= 0)
				ahi = alo;

			alo = aj;
			flo = fj;
			glo = gj;
		}
	}

	return aj;
}



template <class F, class G>
double strongWolfeLS (const Eigen::VectorXd& x, const Eigen::VectorXd& d, F f, G g, 
					  double c1 = 1e-4, double c2 = 0.9, double aMax = 5.0, int maxIter = 10)
{
	double ap = 0.0, ai = 1.0;

	double f0 = f(x);
	double g0 = g(x).transpose() * d;

	double fp = f0;
	double fMax = f(x + aMax * d);

	
	//double fm = f(x + aMax * d);
	//double gm = g(x + aMax * d).transpose() * d;


	for(int i = 1; i <= maxIter; ++i)
	{
		double fi = f(x + ai * d);

		if(fi > f0 + c1 * ai * g0 || (i > 1 && fi >= fp))
			return std::max(zoom(ap, ai, x, d, f, g, f0, g0, c1, c2, aMax), 1e-5);

		double gi = g(x + ai * d).transpose() * d;

		if(std::abs(gi) <= -c2 * g0)
			return std::max(ai, 1e-5);

		if(gi >= 0)
			return std::max(zoom(ai, ap, x, d, f, g, f0, g0, c1, c2, aMax), 1e-5);

		ap = ai;
		fp = fi;

		ai = interpolate(ai, fi, gi, aMax, fMax);
		//ai = interpolate(0.0, f0, g0, ai, fi);
	}

	return std::max(ai, 1e-5);
}



template <class Function, class Gradient>
void gradientDescent (Function function, Gradient gradient, Eigen::VectorXd& theta, double alpha = 0.1, int iterations = 100, double error = 1e-8)
{
	Eigen::VectorXd direction = -gradient(theta);

	while(iterations-- && direction.norm() > error)
	{
		std::cout << iterations << "    " << function(theta) << "\n";

		//alpha = lineSearch(theta, direction);

		alpha = strongWolfeLS(theta, direction, function, gradient);
		
		theta = theta + alpha * direction;

		direction = -gradient(theta);
	}
}






Eigen::VectorXd lbfgsDirection (Eigen::VectorXd& grad, const Eigen::VectorXd& s, const Eigen::VectorXd& y, 
							    std::deque<Eigen::VectorXd>& sOld, std::deque<Eigen::VectorXd>& yOld, int m)
{
	
	//Eigen::VectorXd q = double(double(s.transpose() * y) / (y.transpose() * y)) * grad;
	Eigen::VectorXd q = grad;

	std::vector<double> alpha(m);


	if(sOld.size() > m)
	{
		sOld.pop_front();
		yOld.pop_front();
	}

	sOld.push_back(s);
	yOld.push_back(y);


	for(int i = sOld.size() - 1; i >= 0; --i)
	{
		alpha[i] = double(1.0 / double(yOld[i].transpose() * sOld[i])) * double(sOld[i].transpose() * q);

		q = q - alpha[i] * yOld[i];
	}

	Eigen::VectorXd r = double(double(y.transpose() * s) / (y.transpose() * y)) * q;


	for(int i = 0; i < sOld.size(); ++i)
	{
		double beta = (1.0 / double(yOld[i].transpose() * sOld[i])) * double(yOld[i].transpose() * r);

		r = r + (alpha[i] - beta) * sOld[i];
	}

	return -r;
}




template <class Gradient>
auto hessianFD (const Eigen::VectorXd& x, Gradient gradient, double h = 1e-8)
{
	Eigen::MatrixXd hes(x.rows(), x.rows());

	Eigen::VectorXd y = x;


	for(int i = 0; i < x.rows(); ++i)
	{
		for(int j = 0; j < x.rows(); ++j)
		{
			y(j) = x(j) - h;

			double gi1 = gradient(y)(i);

			y(j) = x(j) + h;

			double gi2 = gradient(y)(i);

			y(j) = x(j);
			y(i) = x(i) - h;

			double gj1 = gradient(y)(j);

			y(i) = x(i) + h;

			double gj2 = gradient(y)(j);

			y(i) = x(i);


			hes(i, j) = (gi2 - gi1) / (4.0 * h) + (gj2 - gj1) / (4.0 * h);
 		}
	}

	return hes;
}



template <class Function, class Gradient>
void lbfgs (Function function, Gradient gradient, Eigen::VectorXd& theta, int m = 20, double error = 1e-8, int iterations = 100)
{
	int N = theta.rows();

	// std::deque<Eigen::VectorXd> sOld(1, Eigen::VectorXd::Constant(N, 0.0));
	// std::deque<Eigen::VectorXd> yOld(1, Eigen::VectorXd::Constant(N, 0.0));

	std::deque<Eigen::VectorXd> sOld;
	std::deque<Eigen::VectorXd> yOld;


	//Eigen::VectorXd grad = 1e-4 * gradient(theta);
	// Eigen::VectorXd grad = 1e-4 * gradient(theta);

	// Eigen::VectorXd oldGrad = grad;

	// //Eigen::VectorXd direction = -grad;
	// Eigen::VectorXd direction = -hessianFD(theta, gradient).inverse() * grad;


	// double step = strongWolfeLS(theta, direction, function, gradient);

	// theta = theta + step * direction;


	Eigen::VectorXd grad = 1e-5 * gradient(theta);

	Eigen::VectorXd oldGrad = grad;

	Eigen::VectorXd direction = -grad;

	double step = strongWolfeLS(theta, direction, function, gradient);

	Eigen::VectorXd sk = step * direction;

	Eigen::VectorXd yk = 1e-5 * gradient(theta + sk) - grad;


	direction = -std::max(((yk.transpose().dot(sk)) / (yk.transpose().dot(yk))), 1e-4) * grad;

	step = strongWolfeLS(theta, direction, function, gradient);

	theta = theta + step * direction;

	//oldGrad = gradient(theta);


	while(iterations-- && direction.norm() >= 1e-7 && step >= 1e-7)
	{
		DB(iterations << "   " << function(theta) << "   " << direction.norm() << "     " << step);

		grad = gradient(theta);

		direction = lbfgsDirection(grad, step * direction, grad - oldGrad, sOld, yOld, m);

		step = strongWolfeLS(theta, direction, function, gradient);
		//step = 1.0;

		theta = theta + step * direction;

		oldGrad = grad;
	}
}






#endif // ML_GRADIENT_DESCENT_H
