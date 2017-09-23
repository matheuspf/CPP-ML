#ifndef ML_NORMAL_GAMMA_DISTRIBUTION_H
#define ML_NORMAL_GAMMA_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../Gamma/Gamma.h"




struct NormalGamma : public Gamma
{
	NormalGamma (double alpha_ = 1.0, double beta_ = 1.0, double gamma_ = 1.0, double delta_ = 0.0) : Gamma(alpha_, beta_)
	{
		params(alpha_, beta_, gamma_, delta_);
	}


	void params (double alpha_, double beta_ = 1.0, double gamma_ = 1.0, double delta_ = 0.0)
	{
		Gamma::params(alpha_, beta_);

		gamma = gamma_;
		delta = delta_;

		C = C * (sqrt(gamma) / sqrt(2*pi()));
	}

	auto params ()
	{
		return make_tuple(alpha, beta, gamma, delta);
	}


	void fit (const Vec& mu, const Vec& sigma, bool opt = false)
	{
		Gamma::fit(sigma, opt);

		double delta_ = mu.mean();

		double gamma_ = mu.rows() / (sigma.array() * pow(delta_ - mu.array(), 2)).sum();

		params(alpha, beta, gamma_, delta_);
	}




	auto operator () ()
	{
		double sigma = dist(gen);

		double mu = normal_distribution<double>(delta, 1.0 / (sqrt(sigma * gamma)))(gen);

		return make_tuple(mu, sigma);
	}


	double operator () (double mu, double sigma)
	{
		return (C / sqrt(sigma)) * pow(1.0 / sigma, alpha + 1) * exp(-(2 * beta + gamma * pow(delta - mu, 2)) / (2 * sigma));
	}
	


	double gamma;
	double delta;
};





#endif // ML_NORMAL_GAMMA_DISTRIBUTION_H