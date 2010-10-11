/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "Parameter.h"
#include "Utility.h"

namespace fastcrf {

using namespace std;

Parameter::Parameter() {
	mEDGE = "@";
	clear();
}

Parameter::~Parameter() {
	/// memory free
	//if (weight_) 
	//	delete[] weight_;
	//if (gradient_)
	//	delete[] gradient_;
}

/** Size of weight vector.
*/
size_t Parameter::size() {
	return nWeight_;
}

/** Clear the memory.
*/
void Parameter::clear(bool state) {
	if (!state) {
		stateMap_.clear();
		stateVec_.clear();
	}
	featureMap_.clear();
	featureVec_.clear();
	//m_StateID.clear();
	empiricalCount_.clear();
	weight_.clear();
	gradient_.clear();
	paramIndex_.clear();
	nWeight_ = 0;
}

/** Initialize the weight vector.
*/
void Parameter::initialize() {
	weight_.resize(nWeight_);
	fill(weight_.begin(), weight_.end(), 0.0);
	gradient_.resize(nWeight_);
	fill(gradient_.begin(), gradient_.end(), 0.0);
}

/** Initialize the gradient vector.
*/
void Parameter::initializeGradient() {
	for (size_t i=0; i < nWeight_; i++) {
		gradient_[i] = -empiricalCount_[i];
	}
}

/** Get weight vector pointer.
*/
double* Parameter::getWeight() { 
	return &weight_[0]; 
}

void Parameter::setWeight(double* theta) {
	for (size_t i = 0; i < nWeight_; i++) {
		weight_[i] = *(theta++);
	}
}

/** Get gradient vector pointer.
	@warning	The size of gradient vector should be larger than 1.
*/
double* Parameter::getGradient() {
	return &gradient_[0]; 
}


/**	Return the size of feature vector.
*/
size_t Parameter::sizeFeatureVec() { 
	return featureVec_.size(); 
}

/**	Return the size of state vector.
*/
size_t Parameter::sizeStateVec() { 
	return stateVec_.size(); 
}

/**	Return the state map and vector.
*/
pair<Map, Vec> Parameter::getState() { 
	return make_pair(stateMap_, stateVec_); 
}

/**
*/
size_t Parameter::addNewState(const string& key) {
	size_t oid;
	if (stateMap_.find(key) == stateMap_.end()) {
		oid = stateVec_.size();
		stateMap_.insert(make_pair(key, oid));
		stateVec_.push_back(key);
	} else {
		oid = stateMap_[key];
		if (stateVec_[oid] != key) {
			cerr << "outcome id mismatch error" << endl;
			exit(1);
		}
	}
	return oid;
}

/**
*/
int Parameter::findState(const string& key) {
	int oid = -1;

	if (stateMap_.find(key) != stateMap_.end()) {
		oid = stateMap_[key];
	}

	return oid;
}

/**
*/
int Parameter::findObs(const string& key) {
	int pid = -1;

	if (featureMap_.find(key) != featureMap_.end()) {
		pid = featureMap_[key];
	}

	return pid;
}

/**
*/
size_t Parameter::addNewObs(const string& key) {
	size_t pid;

	if (featureMap_.find(key) != featureMap_.end()) {
		pid = featureMap_[key];
	} else {
		pid = featureVec_.size();
		featureMap_[key] = pid;
		featureVec_.push_back(key);
	}
	return pid;
}

/** Update the parameter.
*/
size_t Parameter::updateParameter(size_t oid, size_t pid, double fval) {
	size_t fid;
	assert(paramIndex_.size() >= pid);
	if (paramIndex_.size() == pid) {	/// New feature
		vector<pair<size_t, size_t> > param;
		fid = nWeight_;
		nWeight_++;
		empiricalCount_.push_back(fval);
		weight_.push_back(0.0);
		gradient_.push_back(0.0);
		param.push_back(make_pair(oid, fid));
		paramIndex_.push_back(param);
	} else {	 /// A parameter vector is exist 
		vector<pair<size_t, size_t> >& param = paramIndex_[pid];
		size_t i;
		for (i = 0; i < param.size(); i++) {
			if (param[i].first == oid) {
				empiricalCount_[param[i].second] += fval;
				break;
			}
		}
		if (i == param.size()) {
			fid = nWeight_;
			nWeight_++;
			empiricalCount_.push_back(fval);
			weight_.push_back(0.0);
			gradient_.push_back(0.0);
			param.push_back(make_pair(oid, fid));
	        sort(param.begin(), param.end());
		}
	}
	return nWeight_;
}

void Parameter::finalize() {
	vector<double> tmp_Count = empiricalCount_;
	fill(empiricalCount_.begin(), empiricalCount_.end(), 0.0);

    size_t fid = 0;
    for (size_t i = 0; i < paramIndex_.size(); ++i) {
        vector<pair<size_t, size_t> >& param = paramIndex_[i];
        for (size_t j = 0; j < param.size(); ++j) {
			empiricalCount_[fid] = tmp_Count[param[j].second];
            param[j].second = fid;
            fid++;
        }
    }
	assert(fid == nWeight_);
}

size_t Parameter::getDefaultState() {
	return defaultStateID;
}

void Parameter::setDefaultState(const string& label) {
	if (stateMap_.find(label) != stateMap_.end())
		defaultStateID = stateMap_[label];
}

/** Making state index.
*/
void Parameter::makeStateIndex() {
	
	/** Default active set.
	*/
	allState_.clear();
	for (size_t i = 0; i < sizeStateVec(); ++i)
		allState_.push_back(i);
	activeSet_.clear();
	revActiveSet_.clear();
	for (size_t i = 0; i < sizeStateVec(); ++i) {
		activeSet_.push_back(allState_);
		revActiveSet_.push_back(allState_);
	}

	/// Make state index
	stateIndex_.clear();
	for (size_t y1=0; y1 < sizeStateVec(); y1++) {
		//int pid = FindState(y1);
		//if (pid < 0) 
		//	continue;
		string fi = mEDGE + stateVec_[y1];
		if (featureMap_.find(fi) != featureMap_.end()) {
			size_t pid = featureMap_[fi];
			vector<pair<size_t, size_t> >& param = paramIndex_[pid];
			for (size_t i = 0; i < param.size(); i++) {
				StateParam element;
				element.y1 = y1;
				element.y2 = param[i].first;
				element.fid = param[i].second;
				element.fval = 1.0;
				stateIndex_.push_back(element);
				
				/*
				// make back pointer (t, t-1)
				vector<size_t> &backpointer = activeSet_[element.y2];
				backpointer.push_back(y1);
				vector<size_t> &backpointer2 = revActiveSet_[y1];
				backpointer2.push_back(element.y2);
				*/

			}	///< for
		} ///< if else
	} ///< for each state
}

/** Make the index for Tied Potential
*/
void Parameter::makeActiveSet(double eta) {
	
	activeSet_.clear();
	revActiveSet_.clear();
	activeSet_.resize(sizeStateVec());
	revActiveSet_.resize(sizeStateVec());

	/// Make state index
	vector<StateParam>::iterator iter = stateIndex_.begin();
	for (; iter != stateIndex_.end(); ++iter) {
		if (abs( exp(weight_[iter->fid]) - 1.0 ) > eta) {
			vector<size_t> &backpointer = activeSet_[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = revActiveSet_[iter->y1];
			backpointer2.push_back(iter->y2);
		}
	}
}

/** Make the index for Tied Potential
*/
void Parameter::makeTP1(double K) {
	
	/// Make state index
	selectedStateIndex_.clear();
	remainStateIndex_.clear();

	remainCount_.clear();
	remainFeatID_.clear();
	vector<double> remain_size;
	size_t temp_fid = updateParameter(defaultStateID, addNewObs("@REMAIN@"), 0.0); // empirical feature count is augmented
	remainFeatID_.push_back(temp_fid);
	remain_size.push_back(0.0);
	remainCount_.push_back(0.0);
	
	// redefinition for tied potential (redundant)
	activeSet_.clear();
	revActiveSet_.clear();
	activeSet_.resize(sizeStateVec());
	revActiveSet_.resize(sizeStateVec());
	
	vector<StateParam>::iterator iter = stateIndex_.begin();
	for (; iter != stateIndex_.end(); ++iter) {
		if (empiricalCount_[iter->fid] >= K) {
			selectedStateIndex_.push_back(*iter);

			// make back pointer (t, t-1)
			vector<size_t> &backpointer = activeSet_[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = revActiveSet_[iter->y1];
			backpointer2.push_back(iter->y2);

		} else {
			remainStateIndex_.push_back(*iter);
			remainCount_[0] += empiricalCount_[iter->fid];
			//remainFeatID_[element.y2] = element.fid;
			remain_size[0] += 1.0;
			empiricalCount_[remainFeatID_[0]] += empiricalCount_[iter->fid]; // empirical feature count is augmented
			//empiricalCount_[iter->fid] = 0.0;
		}
	}
	
	//for (size_t i = 0; i < SizeStateVec(); i++) {
	/*	if (remain_size[0] > 0)
			empiricalCount_[remainFeatID_[0]] /= remain_size[0];
		else
			empiricalCount_[remainFeatID_[0]] = 0.0;*/
	//}
}

/** Make the index for Tied Potential
*/
void Parameter::makeTP(double K) {
	
	/// Make state index
	selectedStateIndex_.clear();
	remainStateIndex_.clear();

	remainCount_.clear();
	remainFeatID_.clear();
	vector<double> remain_size;
	for (size_t i = 0; i < sizeStateVec(); i++) {
		size_t temp_fid = updateParameter(i, addNewObs("@REMAIN@"), 0.0); // empirical feature count is augmented
		remainFeatID_.push_back(temp_fid);
		remain_size.push_back(0.0);
		remainCount_.push_back(0.0);
	}
	
	// redefinition for tied potential (redundant)
	activeSet_.clear();
	revActiveSet_.clear();
	activeSet_.resize(sizeStateVec());
	revActiveSet_.resize(sizeStateVec());
	
	vector<StateParam>::iterator iter = stateIndex_.begin();
	for (; iter != stateIndex_.end(); ++iter) {
		if (empiricalCount_[iter->fid] >= K) {
			selectedStateIndex_.push_back(*iter);
			// make back pointer (t, t-1)
			vector<size_t> &backpointer = activeSet_[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = revActiveSet_[iter->y1];
			backpointer2.push_back(iter->y2);

		} else {
			remainStateIndex_.push_back(*iter);
			remainCount_[iter->y2] += empiricalCount_[iter->fid];
			//remainFeatID_[element.y2] = element.fid;
			remain_size[iter->y2] += 1.0;
			empiricalCount_[remainFeatID_[iter->y2]] += empiricalCount_[iter->fid]; // empirical feature count is augmented
			//empiricalCount_[iter->fid] = 0.0;
		}
	}
	
	/*for (size_t i = 0; i < SizeStateVec(); i++) {
		if (remain_size[i] > 0)
			empiricalCount_[remainFeatID_[i]] /= remain_size[i];
		else
			empiricalCount_[remainFeatID_[i]] = 0.0;
	}*/
}

/** Make the index for Tied Potential
*/
void Parameter::makeZero1(double K) {
	
	/// Make state index
	selectedStateIndex_.clear();
	remainStateIndex_.clear();

	remainCount_.clear();
	remainFeatID_.clear();
	vector<double> remain_size;
	size_t temp_fid = updateParameter(defaultStateID, addNewObs("@REMAIN@"), 0.0); // empirical feature count is augmented
	remainFeatID_.push_back(temp_fid);
	remain_size.push_back(0.0);
	remainCount_.push_back(0.0);
	
	// redefinition for tied potential (redundant)
	activeSet_.clear();
	revActiveSet_.clear();
	activeSet_.resize(sizeStateVec());
	revActiveSet_.resize(sizeStateVec());
	
	vector<StateParam>::iterator iter = stateIndex_.begin();
	for (; iter != stateIndex_.end(); ++iter) {
		//if (empiricalCount_[iter->fid] >= K) {
		if (exp(weight_[iter->fid]) > K) {
			selectedStateIndex_.push_back(*iter);

			// make back pointer (t, t-1)
			vector<size_t> &backpointer = activeSet_[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = revActiveSet_[iter->y1];
			backpointer2.push_back(iter->y2);

		} else {
			remainStateIndex_.push_back(*iter);
			remainCount_[0] += empiricalCount_[iter->fid];
			//remainFeatID_[element.y2] = element.fid;
			remain_size[0] += 1.0;
			empiricalCount_[remainFeatID_[0]] += empiricalCount_[iter->fid]; // empirical feature count is augmented
			//empiricalCount_[iter->fid] = 0.0;
		}
	}
	
	/*	if (remain_size[0] > 0)
			empiricalCount_[remainFeatID_[0]] /= remain_size[0];
		else
			empiricalCount_[remainFeatID_[0]] = 0.0;
	*/
}

/** Make the index for Tied Potential
*/
void Parameter::makeZero2(double K) {
	
	/// Make state index
	selectedStateIndex_.clear();
	remainStateIndex_.clear();

	remainCount_.clear();
	remainFeatID_.clear();
	vector<double> remain_size;
	for (size_t i = 0; i < sizeStateVec(); i++) {
		size_t temp_fid = updateParameter(i, addNewObs("@REMAIN@"), 0.0); // empirical feature count is augmented
		remainFeatID_.push_back(temp_fid);
		remain_size.push_back(0.0);
		remainCount_.push_back(0.0);
	}
	
	// redefinition for tied potential (redundant)
	activeSet_.clear();
	revActiveSet_.clear();
	activeSet_.resize(sizeStateVec());
	revActiveSet_.resize(sizeStateVec());
	
	vector<StateParam>::iterator iter = stateIndex_.begin();
	for (; iter != stateIndex_.end(); ++iter) {
		if (exp(weight_[iter->fid]) > K) {
			selectedStateIndex_.push_back(*iter);
			// make back pointer (t, t-1)
			vector<size_t> &backpointer = activeSet_[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = revActiveSet_[iter->y1];
			backpointer2.push_back(iter->y2);

		} else {
			remainStateIndex_.push_back(*iter);
			remainCount_[iter->y2] += empiricalCount_[iter->fid];
			//remainFeatID_[element.y2] = element.fid;
			remain_size[iter->y2] += 1.0;
			empiricalCount_[remainFeatID_[iter->y2]] += empiricalCount_[iter->fid]; // empirical feature count is augmented
			//empiricalCount_[iter->fid] = 0.0;
		}
	}
	
	/*
	for (size_t i = 0; i < SizeStateVec(); i++) {
		if (remain_size[i] > 0)
			empiricalCount_[remainFeatID_[i]] /= remain_size[i];
		else
			empiricalCount_[remainFeatID_[i]] = 0.0;
	}
	*/
}

/** Cut off the features
*/
void Parameter::cutOff(double K) {
	vector<double> new_count;
	
	if (K == 0)
		return;
	size_t new_n_theta = 0;
	for (size_t i = 0; i < paramIndex_.size(); i++) {	
		vector<pair<size_t, size_t> >& param = paramIndex_[i];
		vector<pair<size_t, size_t> > new_param;
		for (size_t j = 0; j < param.size(); ++j) {
			size_t oid = param[j].first;
			size_t fid = param[j].second;
			if (empiricalCount_[fid] >= K) {
				new_param.push_back(make_pair(oid, new_n_theta));
				new_count.push_back(empiricalCount_[fid]);
				new_n_theta++;
			}
		}
		paramIndex_[i] = new_param;
	}	
	empiricalCount_ = new_count;
	nWeight_ = new_n_theta;
}

/** Save the model.
	@param	f	output file stream 
	@return	success or failure
*/
bool Parameter::save(ofstream& f) {
	/// Errors	
	if (paramIndex_.size() != featureVec_.size())
		return false;

	/// state
	f << "// State ; " << stateVec_.size() << endl;
    for (size_t i = 0; i < stateVec_.size(); ++i)
        f << stateVec_[i] << endl;
	
	/// feature 
    f << "// Feature ; " << featureVec_.size() << endl;
    for (size_t i = 0; i < featureVec_.size(); ++i)
        f << featureVec_[i] << endl;
	
	/// parameter index
    f << "// Parameter ; " << paramIndex_.size() << endl;
    for (size_t i = 0; i < paramIndex_.size(); ++i) {
        vector<pair<size_t, size_t> >& param = paramIndex_[i];
        f << param.size() << ' ';
        for (size_t j = 0; j < param.size(); ++j) {
            f << param[j].first << ' ';
        }
        f << endl;
    }
    /// write the weight vector
    f   << "// Weight ; " << nWeight_ << endl;
    for (size_t i = 0; i < nWeight_; ++i) {
        f << weight_[i] << endl;	
	}

	return true;
}

/** Load the model.
	@param	filename	model file name to be loaded
	@return	success or failure
*/
bool Parameter::load(ifstream& f) {
	/// useInitializer_
	clear();
    string line;
	size_t count;

    /// state
	getline(f, line);
	vector<string> tok = Tokenize(line);
	if (tok.size() < 4) {
		cerr << "state error\n";
		return false;
	}
	count = atoi(tok[3].c_str());
    for (size_t i = 0; i < count; ++i) {
		getline(f, line);
		stateMap_[line] = i;
		stateVec_.push_back(line);
    }

    /// feature
    getline(f, line);
	tok = Tokenize(line);
	if (tok.size() < 4) {
		cerr << "feature error\n";
		return false;
	}
	count = atoi(tok[3].c_str());
    for (size_t i = 0; i < count; ++i) {
        getline(f, line);
        featureMap_[line] = i;
        featureVec_.push_back(line);
    }

	/// parameter index
    getline(f, line);
	tok = Tokenize(line);
	if (tok.size() < 4)
		return false;
	count = atoi(tok[3].c_str());
    size_t fid = 0;
    vector<pair<size_t, size_t> > param;
    for (size_t i = 0; i < count; ++i) {
        param.clear();
        getline(f, line);
        size_t oid;
        tok = Tokenize(line);
		vector<string>::iterator it = tok.begin();	
        ++it; ///< skip count which is only used in binary format
        for (; it != tok.end();) {
            oid = atoi(it->c_str()); ++it;
            param.push_back(make_pair(oid,fid++));
        }
        paramIndex_.push_back(param);
    }

	/// weight
    getline(f, line);
	tok = Tokenize(line);
	if (tok.size() < 4 || fid != atoi(tok[3].c_str()))
		return false;
	nWeight_ = fid;
	initialize();
	size_t i = 0;
	for (i = 0; i < fid; i++) {
		getline(f, line);
		assert(!line.empty());
		weight_[i] = atof(line.c_str());
	}
	assert(i == nWeight_);

	/// setting
	empiricalCount_.resize(nWeight_);
	fill(empiricalCount_.begin(), empiricalCount_.end(), 0.0);
	
	return true;
}

/** Print the information.
*/
void Parameter::print(Logger *p_logger) {
	//log_->Report("[Parameters]\n");
	p_logger->report("  # of States = \t%d\n", stateVec_.size());
	p_logger->report("  # of Features = \t%d\n", featureVec_.size());
	p_logger->report("  # of Parameters = \t%d\n\n", nWeight_);
}

}
