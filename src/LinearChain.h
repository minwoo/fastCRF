/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __LINEARCHAIN_H__
#define __LINEARCHAIN_H__

#include <vector>
#include <string>
#include <map>
#include "StructuredModel.h"

namespace fastcrf {

using namespace std;

/** Linear-chain Model - An abstract class for linear-chain structured model
*/
class LinearChain : public StructuredModel {
public:
	LinearChain();	 
	~LinearChain();	

	virtual bool saveModel(const string& filename, bool isBinary = false);
	virtual bool loadModel(const string& filename, bool isBinary = false);

	//virtual bool Train(Data *trainData, Data *devData, Option option);
	//virtual bool Test(Data *testData, Option option);
	//virtual vector<string> Predict(Example example);
	
	virtual void initializeModel();
	
protected:
	size_t sizeY_;
	size_t defaultY_;

	vector<long double> remainWeight_;
	//virtual vector<size_t> Viterbi(Example example);

	
	// Inference
	Real nodeFactor(Example &example);	
	Real edgeFactor();	
	vector<size_t> viterbi(Real &node, Real &edge, bool standard = false);	

	inline size_t MAT3(size_t I, size_t X, size_t Y) {
		return ((sizeY_ * sizeY_ * (I)) + (sizeY_ * (X)) + Y);
	};
	inline size_t MAT2(size_t I, size_t X) {
		return ((sizeY_ * (I)) + X);
	};

};


}

#endif
