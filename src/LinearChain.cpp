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
#include <cstring>
#include "LinearChain.h"
#include "Evaluator.h"
#include "Utility.h"
#include "LBFGS.h"

namespace fastcrf {

using namespace std;

LinearChain::LinearChain() {
	defaultY_ = 0;
}

LinearChain::~LinearChain() {
}

/**	Calculating the node factors.
*/
Real LinearChain::nodeFactor(Example &example) {
	/** Initializing.
	*/
	double* lambda = param_.getWeight();
	size_t m_example_size = example.size() + 1;	///< sequence length
	Real node(m_example_size * sizeY_);
	fill(node.begin(), node.end(), 1.0);

	/** Calculating the node factors.
	*/
	for (size_t i = 0; i < m_example_size-1; i++) {
		
		/** \sum \lambda_k f_k
		*/	
		for (vector<pair<size_t, double> >::iterator iter = example[i].x_.begin(); iter != example[i].x_.end(); iter++) {
			vector<pair<size_t, size_t> >& param = param_.paramIndex_[iter->first];
			for (size_t j = 0; j < param.size(); ++j) {
				node[MAT2(i, param[j].first)] *= exp(lambda[param[j].second] * iter->second);
			}
		}
	} ///< for 
	
	return node;
}

/**	Calculating the edge factors.
*/
Real LinearChain::edgeFactor() {
	/** Initializing.
	*/
	// state transition is independent of time t and training set 
	double* lambda = param_.getWeight();
	Real edge(sizeY_ * sizeY_);
	fill(edge.begin(), edge.end(), 1.0);

	/** Caclucating. */
	if (option_.inferenceType_ == SFB || option_.inferenceType_ == ZEROOUT || option_.inferenceType_ == STANDARD) {
		vector<StateParam>::iterator iter = param_.stateIndex_.begin();
		for (; iter != param_.stateIndex_.end(); ++iter) {
			edge[MAT2(iter->y1,iter->y2)] *= exp(lambda[iter->fid] * iter->fval);	 
		}
	} else {
		vector<StateParam>::iterator iter = param_.selectedStateIndex_.begin();
		for (; iter != param_.selectedStateIndex_.end(); ++iter) {	
			edge[MAT2(iter->y1,iter->y2)] *= exp(lambda[iter->fid] * iter->fval);	 
		}
	}
		
	return edge;
}


/** Viterbi search to find the best probable output sequence.
*/
vector<size_t> LinearChain::viterbi(Real &node, Real &edge, bool standard) {
	/** Initializing.
	*/
	vector<vector<size_t> > psi;
    vector<vector<long double> > delta;
	size_t prev_max_j = defaultY_;
	long double prev_maxj = -100000.0;
    size_t i, j, k;
    size_t m_example_size = node.size() / sizeY_;

	/** Setting the omega.
	*/
	double omega;
	if (option_.inferenceType_ == ZEROOUT)
		omega = 1.0;
	else if (option_.inferenceType_ == TP1 || option_.inferenceType_ == ZERO1)
		omega = remainWeight_[0];
	else
		omega = 0.0;
		
	if (standard)
		omega = 0.0;
		
	/** Recursion.
	*/
    for (i = 0; i < m_example_size-1; i++) {
        vector<size_t> psi_i;
        vector<long double> delta_i;

		long double maxj = -10000.0;
		size_t max_j = 0;

        for (j = 0; j < sizeY_; j++) {
            long double max = -10000.0;
            size_t max_k = 0;
            if (i == 0) {
                max = 1.0; //m_M[MAT3(i,defaultY_,j)];
                max_k = defaultY_;
            } else {
                vector<size_t>::iterator it, end_it;
            	if (option_.inferenceType_ == SFB) {
					 it = param_.activeSet2_[i-1].begin();
					 end_it = param_.activeSet2_[i-1].end();
            	} else {
					 it = param_.activeSet_[j].begin();
					 end_it = param_.activeSet_[j].end();
            	}
            	if (option_.inferenceType_ == TP2 || option_.inferenceType_ == ZERO2)
                	omega = remainWeight_[j];
                if (standard) {
                	it = param_.allState_.begin();
                	end_it = param_.allState_.end();                
                	omega = 0.0;
                }                	
                
				for ( ; it != end_it; ++it) {
					double val = delta[i-1][*it] * edge[MAT2(*it,j)];
	                if (val > max) {
	                    max = val;
	                    max_k = *it;
	                }
	            }
				// See [Siddiqi and Moore, 2005, ICML]
				if (max < prev_maxj * omega) {
					max = prev_maxj * omega;
					max_k = prev_max_j;
				}
            } ///< for j
			
			/**
			*/
			max = max * node[MAT2(i, j)]; // / m_AlphaScale[i];
            delta_i.push_back(max);
            psi_i.push_back(max_k);

			if (max > maxj) {
				maxj = max;
				max_j = j;
			}
        } // for j

        delta.push_back(delta_i);
        psi.push_back(psi_i);

		prev_max_j = max_j;
		prev_maxj = maxj;
	
    } // for i

	/** Last state.
	*/
	vector<size_t> psi_i(sizeY_, 0);
	vector<long double> delta_i(sizeY_, -10000.0);
	long double max = -10000.0;
	size_t max_k = 0;
	for (size_t k=0; k < sizeY_; k++) {
		double val = delta[m_example_size-2][k]; 
		if (val > max) {
			max = val;
			max_k = k;
		}
	}
	//max /= m_AlphaScale[m_example_size-1];
	delta_i[defaultY_] = max;
	psi_i[defaultY_] = max_k;
	delta.push_back(delta_i);
	psi.push_back(psi_i);

	/** Back-tracking.
	*/
    vector<size_t> y_example;
    size_t prev_y = defaultY_;
    for (i = m_example_size-1; i >= 1; i--) {
        size_t y = psi[i][prev_y];
        y_example.push_back(y);
        prev_y = y;
    }
    reverse(y_example.begin(), y_example.end());
    long double prob = delta[m_example_size-1][defaultY_];

	return y_example;
}

void LinearChain::initializeModel() {
	param_.initialize();
	param_.makeStateIndex();
	sizeY_ = param_.sizeStateVec();
}

/** Save the model.
*/
bool LinearChain::saveModel(const string& filename, bool isBinary) {
	/// Checking the error
	if (filename == "")
		return false;

	timer stop_watch;
	logger_->report("[Model saving]\n");

	/// file stream
    ofstream f(filename.c_str());
    f.precision(20);
    if (!f)
        throw runtime_error("unable to open file to write");

    /// header
    f << "# MAX: A C++ Library for Structured Prediction" << endl;
	f << "# CRF Model file (text format)" << endl;
	f << "# Do not edit this file" << endl;
	f << "# " << endl << ":" << endl;
	
	bool ret = param_.save(f);
	f.close();
	logger_->report("  saving time = \t%.3f\n\n", stop_watch.elapsed());

	return ret;
}

/** Load the model.
*/
bool LinearChain::loadModel(const string& filename, bool isBinary) {
	/// Checking the error
	if (filename == "")
		return false;

	timer stop_watch;
	logger_->report("[Model loading]\n");

	/// file stream
    ifstream f(filename.c_str());
    f.precision(20);
    if (!f)
        throw runtime_error("fail to open model file");

    /// header
	size_t count = 0;
    string line;
    getline(f, line);
    while (line.empty() || line[0] == '#') {
		if (count == 1) {
			vector<string> tok = Tokenize(line);
			if (tok.size() < 2 || tok[1] != "CRF") {
				logger_->report("|Error| Invalid model files ... \n");
				return false;
			}
		}
        getline(f, line);
		count++;
	}

	bool ret = param_.load(f);
	f.close();
	param_.print(logger_);
	logger_->report("  loading time = \t%.3f\n\n", stop_watch.elapsed());
	
	/** For edge feature function.
	*/
	param_.makeStateIndex();
	sizeY_ = param_.sizeStateVec();
		
	return ret;
}

}
