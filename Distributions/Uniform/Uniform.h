#ifndef ML_UNIFORM_DISTRIBUTION_H
#define ML_UNIFORM_DISTRIBUTION_H

#include "../../Modelo.h"


template <typename T = double>
struct Uniform
{
	Uniform (const T& lower_ = T{}, const T& upper_ = T{})
	{
		params(lower_, upper_);
	}


	void params (const T& lower_, const T& upper_)
	{
		lower = lower_;
		upper = upper_;

		update();
	}

	auto params ()
	{
		return make_tuple(lower, upper);
	}


	template <class U = T, enable_if_t<is_same_v<decay_t<U>, Vec>, int> = 0>
	void update ()
	{
		space = 1.0;

		for(int i = 0; i < lower.rows(); ++i)
			space *= (upper(i) - lower(i));
	}

	template <class U = T, enable_if_t<is_same_v<decay_t<U>, double>, int> = 0>
	void update ()
	{
		space = upper - lower;
	}


	template <class U = T, enable_if_t<is_same_v<decay_t<U>, Vec>, int> = 0>
	double operator () (const T& x)
	{
		return (x >= lower).all() && (x <= upper).all() ? 1.0 / space : 0.0;
	}

	template <class U = T, enable_if_t<is_same_v<decay_t<U>, double>, int> = 0>
	double operator () (const T& x)
	{
		return (x >= lower) && (x <= upper) ? 1.0 / space : 0.0;
	}




	auto mean ()
	{
		return (upper - lower) / 2;
	}

	auto mode ()
	{
		return mean();
	}







	T upper;
	T lower;

	double space;


	RandDouble randD;
};



#endif // ML_UNIFORM_DISTRIBUTION_H