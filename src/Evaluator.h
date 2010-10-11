/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include <vector>
#include <string>
#include <map>
#include "Parameter.h"
#include "Data.h"

namespace fastcrf {

using namespace std;

/** Evaluator */
class Evaluator {
public:
	Evaluator();
	Evaluator(Parameter& param, bool bio = true);

	// encoding 
	void encode(Parameter& param, bool bio = true);

	// append the example
	size_t append(Parameter& param, vector<string> ref, vector<string> hyp);
	size_t append(vector<size_t> ref, vector<size_t> hyp);

	/// log-likelihood
	double subtractLL(double p);
	double addLL(double p);
	double getObjFunc();
	double getLL();
	
	// score
	void calculateF1();
	double accuracy();
	vector<double> macroF1();
	vector<double> microF1();
	
	// util
	void initialize();	
	void print(Logger *log_);
	size_t sizeClass();

private:
	// encoding
	map<string, size_t> classMap_;
	vector<string> classVec_;
	vector<size_t> bioIndex_;
	map<size_t, size_t> beginMap_;

	size_t outsideClass_;
	bool useBioTag_;
	bool useBioTagInFirst_;
	
	// for evaluation
	vector<size_t> trueClass_, guessClass_, correctClass_;
	vector<double> precision_, recall_, scoreF1_;

	size_t nCorrect_, nPoint_, nExample_, nClass_;
	size_t nTruePhrase_, nGuessPhrase_, nCorrectPhrase_;
	size_t OUT_OF_CLASS;

	// measures
	double loglikelihood;	
	double accuracy_;	
	double macroF1_,  macroPrecision_, macroRecall_;
	double microF1_, microPrecision_, microRecall_;
	
	vector<pair<size_t, pair<size_t, size_t> > > chunking(vector<size_t> example);

};


}

#endif

