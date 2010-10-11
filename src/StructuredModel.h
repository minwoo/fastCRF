/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __STRUCTUREDMODEL_H__
#define __STRUCTUREDMODEL_H__

#include <vector>
#include <string>
#include <map>
#include "Parameter.h"
#include "Data.h"
#include "Utility.h"
#include "Evaluator.h"

namespace fastcrf {

using namespace std;

typedef vector<double> Real;

enum LearnerType {MLE = 0, PL, PW, EG, SP, MIRA};
enum InferenceType {STANDARD = 0, ZEROOUT, SFB, TP1, TP2, ZERO1, ZERO2, TRP, MCMC};

class Option {
public:
	/// training option
	LearnerType learnerType_;
	LearnerType initType_;
	bool useInitializer_;
	bool useProbModel_;
	size_t maxIter_;
	size_t initIter_;
	double sigma;
	bool L1;
	/// inference option
	InferenceType inferenceType_;
	double eta;
	/// evaluation option
	string outputfilename;
	bool with_confidence;
	size_t n_best;
	
	/** Default option parameters.
	*/
	Option() {
		learnerType_ = MLE;
		initType_ = PL;
		useInitializer_ = false;
		useProbModel_ = true;
		
		maxIter_ = 1000;
		initIter_ = 100;
		sigma = 0.0;
		L1 = false;
		
		inferenceType_ = ZEROOUT;
		eta = 0.0;
		outputfilename = "";
		with_confidence = false;
		n_best = 1;		
	};		
};

/** Structured Model - An abstract class for structured model
*/
class StructuredModel {
public:
	StructuredModel() {};
	~StructuredModel() {};

	virtual bool saveModel(const string& filename, bool isBinary = false) = 0;
	virtual bool loadModel(const string& filename, bool isBinary = false) = 0;

	virtual bool test(Data *testData, Option option) = 0;
	virtual bool train(Data *trainData, Data *devData, Option option) = 0;
	//virtual vector<string> Predict(Example example) = 0;

	virtual void clear() { param_.clear(); };
	virtual void initializeModel() = 0;

	void setLogger(Logger *logger) { logger_ = logger; };
	Parameter* getParameter() { return &param_; };	

protected:
	Parameter param_;	
	
	Logger *logger_;
	Option option_;

};


}

#endif
