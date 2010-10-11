/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __LINEARCRF_H__
#define __LINEARCRF_H__

#include <vector>
#include <string>
#include <map>
#include "LinearChain.h"
#include "Data.h"
#include "Evaluator.h"

namespace fastcrf {

using namespace std;

/** Linear-chain Conditional Random Fields.
*/
class LinearCRF : public LinearChain {
public:
	LinearCRF();
	LinearCRF(Logger *logger);

	bool test(Data *test_data, Option option);	
	bool train(Data *train_data, Data *dev_data, Option option); 
	void evaluate(Data *data, Evaluator& eval, bool standard = false);

private:
		
	// Inference
	pair<Real, Real> forward(Real &node, Real &edge);
	pair<Real, Real> backward(Real &node, Real &edge);	
	long double partition(Real &alpha);	
	
	// Parameter Estimation
	long double loglikelihood(pair<Real, Real> &alpha, Real &node, Real &edge, Example& example);	
	double* computeGradient(Data *data, Evaluator& eval);
	bool trainWithMLE(Data *train_data, Data *dev_data);
	bool trainWithPL(Data *train_data, Data *dev_data);
	
	size_t num_active_set;
	float avg_active_set;

	size_t regularize(Evaluator& eval, double sigma, bool L1);
	
};	///< LinearCRF

}

#endif
