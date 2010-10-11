/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __PARAM_H__
#define __PARAM_H__

#include <vector>
#include <string>
#include <map>
#include "Utility.h"

namespace fastcrf {

using namespace std;

/** Structure for observation and state parameter.
*/
struct ObsParam {
	size_t y, fid;
	double fval;
};

struct StateParam {
	size_t y1, y2, fid;
	double fval;
};

/** Typedef for Map and Vec
*/
typedef map<string, size_t> Map;
typedef vector<string> Vec;

/** Parameter class.
	@class Parameter
*/
class Parameter {
public:
	Parameter();
	~Parameter();

	/// weight vector
	void initialize();
	void initializeGradient();
	void clear(bool state = false);
	size_t size(); 
	void print(Logger *log_);

	/// save and load
	bool save(ofstream& f);
	bool load(ifstream& f);

	/// Parameters 
	double* getWeight();
	double* getGradient();
	void setWeight(double* theta);

	/// Dictionary access functions
	size_t sizeFeatureVec();
	size_t sizeStateVec();
	pair<Map, Vec> getState();
	int findObs(const string& key);
	int findState(const string& key);
	size_t getDefaultState();
	string getDefaultLabel() { return stateVec_[0]; }; // stupid implm... 
	void setDefaultState(const string& label);

	/// Update and test the parameters
	size_t addNewState(const string& key);
	size_t addNewObs(const string& key);
	size_t updateParameter(size_t oid, size_t pid,  double fval = 1.0);
	void finalize();
	void makeStateIndex();
	void cutOff(double K);

	/// Parameter index
	vector<vector<pair<size_t, size_t> > > paramIndex_;
	vector<StateParam> stateIndex_;

	/// Function:Approximate inference
	void makeActiveSet(double eta = 1E-02);
	vector<StateParam> selectedStateIndex_;
	vector<StateParam> remainStateIndex_;
	void makeTP1(double K);	
	void makeTP(double K);
	void makeZero1(double K);
	void makeZero2(double K);
	
	vector<size_t> remainFeatID_;
	vector<double> remainCount_;
	
	vector<vector<size_t> > activeSet_;
	vector<vector<size_t> > activeSet2_;
	vector<size_t> allState_;
	vector<vector<size_t> > revActiveSet_;

protected:
	/// Weight
	size_t nWeight_;
	vector<double> weight_;
	vector<double> gradient_;
	vector<double> empiricalCount_;
	
	/// Dictionary
	Map featureMap_;
	Vec featureVec_;
	Map stateMap_;
	Vec stateVec_;

	/// Options
	string mEDGE;
	size_t defaultStateID;

};

} // namespace argmax

#endif

