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
#include "LinearCRF.h"
#include "Evaluator.h"
#include "Utility.h"
#include "LBFGS.h"

namespace fastcrf {

using namespace std;

LinearCRF::LinearCRF() {
	defaultY_ = 0;
}

LinearCRF::LinearCRF(Logger *logger) {
	setLogger(logger);
	defaultY_ = 0;
}


/**	Forward Recursion.
	Computing and storing the alpha value.
*/
pair<Real, Real> LinearCRF::forward(Real &node, Real &edge) {
	/** Initializing.
	*/
	size_t m_example_size = node.size() / sizeY_;
	Real alpha(m_example_size * sizeY_);
	fill(alpha.begin(), alpha.end(), 0.0);
	Real m_AlphaScale(m_example_size);
	fill(m_AlphaScale.begin(), m_AlphaScale.end(), 1.0);
	long double sum = 0.0;
	
	vector<size_t> prune;
	if(option_.inferenceType_ == SFB)
		param_.activeSet2_.clear();
	
	/** Setting the omega.
	*/
	double omega = 1.0;
	if (option_.inferenceType_ == ZEROOUT)
		omega = 1.0;
	else if (option_.inferenceType_ == TP1 || option_.inferenceType_ == ZERO1)
		omega = remainWeight_[0];
	else
		omega = 0.0;
			
	/** Initial state.
	*/
	for (size_t j = 0; j < sizeY_; j++) {
		alpha[MAT2(0, j)] += node[MAT2(0, j)] * 1.0; 
		sum += alpha[MAT2(0, j)];
		prune.push_back(j);		
	}
	for (size_t j = 0; j < sizeY_; j++) 
		alpha[MAT2(0, j)] /= sum;
	m_AlphaScale[0] = sum;
	if (option_.inferenceType_ == SFB)
		param_.activeSet2_.push_back(prune);
	
	/** Recursion. 
	*/
    for (size_t i = 1; i < m_example_size-1; i++) {
		sum = 0.0;
		
		/** \alpha_t = \sum_y \phi_{y_t,x} * \phi_{y_t-1,y_t} * \alpha_{t-1}
		*/
        for (size_t j = 0; j < sizeY_; j++) {
			size_t k, index = MAT2(i, j);
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
			//vector<size_t>::iterator it = param_.activeSet_[j].begin();
			for ( ; it != end_it; ++it) {
                alpha[index] += alpha[MAT2(i-1, *it)] * (edge[MAT2(*it,j)] - omega);
			}
			
			alpha[index] = node[index] * (alpha[index] + omega);
			sum += alpha[index];
        }
        
        /** Scaling factor.
        */
		for (size_t j = 0; j < sizeY_; j++) 
			alpha[MAT2(i, j)] /= sum;
		m_AlphaScale[i] = sum;
		
		/** For dynamic setting for active set.
			See [Chris Pal et al., 2006, ICASSP]
		*/
		if (option_.inferenceType_ == SFB) {
			// Compute the new dense belief
			// gamma = alpha * beta  (beta = 1 for all variables)
			// Compress into a sparse belief; Sort the elements of the beilef vector and truncate after logZ(I) exceeds -eta
			prune.clear();
			for (size_t j = 0; j < sizeY_; j++) {
				if (alpha[MAT2(i, j)] > option_.eta)
					prune.push_back(j);
			}
			param_.activeSet2_.push_back(prune);
		}
		
    } ///< for i
	
	/** Final state.
	*/
	for (size_t k = 0; k < sizeY_; k++) {
		alpha[MAT2(m_example_size-1, defaultY_)] += alpha[MAT2(m_example_size-2, k)]; 
	}
	m_AlphaScale[m_example_size-1] = alpha[MAT2(m_example_size-1, defaultY_)];
	
	if (option_.inferenceType_ == SFB) {
		prune.clear();
		prune.push_back(defaultY_);
		param_.activeSet2_.push_back(prune);
	}
	
	return make_pair(alpha, m_AlphaScale);
}

/**	Backward Recursion.
	Computing and storing the beta value.
*/
pair<Real, Real> LinearCRF::backward(Real &node, Real &edge) {
	/** Initialing.
	*/
	size_t m_example_size = node.size() / sizeY_;
	Real beta(m_example_size * sizeY_);
	fill(beta.begin(), beta.end(), 0.0);
	Real m_BetaScale(m_example_size);
	fill(m_BetaScale.begin(), m_BetaScale.end(), 1.0);
	long double sum = 0.0;

	/** Setting the omega.
	*/
	double omega;
	if (option_.inferenceType_ == ZEROOUT)
		omega = 1.0;
	else if (option_.inferenceType_ == TP1 || option_.inferenceType_ == ZERO1)
		omega = remainWeight_[0];
	else
		omega = 0.0;
		
	/** Final state.
	*/
	beta[MAT2(m_example_size-1, defaultY_)] = 1.0;
	for (size_t k = 0; k < sizeY_; k++) {
		beta[MAT2(m_example_size-2, k)] += 1.0;
		sum += beta[MAT2(m_example_size-2, k)];
	}
	for (size_t k = 0; k < sizeY_; k++) 
		beta[MAT2(m_example_size-2, k)] /= sum;
	m_BetaScale[m_example_size-2] = sum;

	/** Recursion.
	*/
    for (int i = m_example_size-2; i >= 1; i--) {
		sum = 0.0;
		long double constant = 0.0;
		for (size_t k = 0; k < sizeY_; k++) {
			if (option_.inferenceType_ == TP2 || option_.inferenceType_ == ZERO2) 
				omega = remainWeight_[k];
			constant += node[MAT2(i,k)] * beta[MAT2(i, k)] * omega;
		}

		/** \beta_t = \sum_y \phi_{y_t,x} * \phi_{y_t+1,y_t} * \beta_{t+1}
		*/
		vector<size_t>::iterator it2, end_it2;
		if (option_.inferenceType_ == SFB) {
			it2 = param_.activeSet2_[i-1].begin();
			end_it2 = param_.activeSet2_[i-1].end();
		} else {
			it2 = param_.allState_.begin();
			end_it2 = param_.allState_.end();
		}
		for (; it2 != end_it2; ++it2) {
			size_t j = *it2;
			size_t index = MAT2(i-1, j);
			size_t index2, k;
			vector<size_t>::iterator it, end_it;
			if (option_.inferenceType_ == SFB) {
				it = param_.activeSet2_[i].begin();
				end_it = param_.activeSet2_[i].end();
			} else {
				it = param_.revActiveSet_[j].begin();
				end_it = param_.revActiveSet_[j].end();
			}
			
			for ( ; it != end_it; ++it) {
				if (option_.inferenceType_ == TP2 || option_.inferenceType_ == ZERO2)
					omega = remainWeight_[*it];
				index2 = MAT2(i, *it);					
			    beta[index] += node[index2] * (edge[MAT2(j, *it)] - omega) * beta[index2];
			}			
			beta[MAT2(i-1, j)] += constant;
			sum += beta[index];
        } // for j
		for (size_t j = 0; j < sizeY_; j++) 
			beta[MAT2(i-1, j)] /= sum;
		m_BetaScale[i-1] = sum;
    } // for i
    
    return make_pair(beta, m_BetaScale);
}

/**	Partition function (Z).
	@return normalizing constant 
*/
long double LinearCRF::partition(Real &alpha) {
	size_t m_example_size = alpha.size() / sizeY_;
    return alpha[MAT2(m_example_size-1, defaultY_)];
}

/** Calculate prob. of y* sequence.
*/
long double LinearCRF::loglikelihood(pair<Real, Real> &alpha, Real &node, Real &edge, Example& example) {
	/* Initialing.
	*/
	long double z = partition(alpha.first);
    long double example_prob = 1.0;
	long double tran = 1.0;
    size_t prev_y = defaultY_;
    size_t y;
    size_t m_example_size = example.size() + 1;
    
    /** Calculating.
    */
    for (size_t i = 0; i < m_example_size; i++) {
        if (i < m_example_size-1) {
            y = example[i].y_;
			if (i > 0)
				tran = edge[MAT2(prev_y, y)];
			example_prob *= node[MAT2(i,y)] * tran;
        } else {
            y = defaultY_;
        }

        prev_y = y;
		example_prob /= alpha.second[i];
    }
    if (example_prob == 0.0) {
        cerr << "example_prob==0 ";
    }
    example_prob = example_prob / z;
	
	/** Returning the prob.
	*/
    return example_prob;
}

/** Computing the gradient vector.
*/
double* LinearCRF::computeGradient(Data *data, Evaluator& eval) {
	/** Initializing.
	*/
	double* lambda = param_.getWeight();
	double* gradient = param_.getGradient();
	avg_active_set = 0.0;
	
	Real edge = edgeFactor();	
	/** Repeat: for each sequence.
	*/
    for (vector<Example>::iterator sit = data->begin(); sit != data->end(); ++sit) {
		vector<size_t> reference, hypothesis;

		/** Forward-Backward recursion.
		*/
		Real node = nodeFactor(*sit);
		pair<Real, Real> alphas = forward(node, edge);	///< Forward
		pair<Real, Real> betas = backward(node, edge);	///< Backward
		Real &alpha = alphas.first, &alpha_scale = alphas.second;
		Real &beta = betas.first, &beta_scale = betas.second;
		long double zval = partition(alpha);

		num_active_set = 0;
		
		if (option_.inferenceType_ == SFB) {		
			for (size_t i = 0; i < param_.activeSet2_.size(); i++)
				num_active_set += param_.activeSet2_[i].size();	
			avg_active_set += (float)num_active_set / sit->size();
		} else {
			for (size_t i = 0; i < param_.activeSet_.size(); i++)
				num_active_set += param_.activeSet_[i].size();
			avg_active_set += (float)num_active_set / sizeY_;
		}
		
		/** Evaluation.
		*/
		vector<size_t> y_example = viterbi(node, edge);

		long double y_example_prob = loglikelihood(alphas, node, edge, *sit);
        if (!finite((double)y_example_prob)) {
            cerr << "calculateLikelihood:" << y_example_prob << endl;
        }
		
		/** Scaling.
		*/
		vector<long double> prod_m_AlphaScale, prod_m_BetaScale;
		prod_m_AlphaScale.clear();
		prod_m_BetaScale.clear();
		long double prod = 1.0;
		for (int a = sit->size(); a >= 0; a--) {
			prod *= alpha_scale[a];
			prod_m_AlphaScale.push_back(prod);
		}
		reverse(prod_m_AlphaScale.begin(), prod_m_AlphaScale.end());
		prod = 1.0;
		for (int a = sit->size(); a >= 0; a--) {
			prod *= beta_scale[a];
			prod_m_BetaScale.push_back(prod);
		}
		reverse(prod_m_BetaScale.begin(), prod_m_BetaScale.end());
		
		/** Repeat: for each data point.
		*/
		size_t prev_outcome = 0;
		size_t i = 0;
		for (Example::iterator it = sit->begin(); it != sit->end(); ++it, ++i) {	
			size_t outcome = it->y_;
			reference.push_back(it->y_);
			hypothesis.push_back(y_example[i]);

			/** Calculating the expectation.
				E[~p] - E[p]
			*/
			long double m_AlphaScale_factor = prod_m_BetaScale[i] / prod_m_AlphaScale[i+1];
			long double m_AlphaScale_factor2 = prod_m_BetaScale[i] / prod_m_AlphaScale[i];
			
			/** Node factor.
			*/
			vector<pair<size_t, double> >::iterator iter = it->x_.begin();
			for (; iter != it->x_.end(); iter++) {
				vector<pair<size_t, size_t> >& param = param_.paramIndex_[iter->first];
				for (size_t j = 0; j < param.size(); ++j) {
					long double prob =  alpha[MAT2(i, param[j].first)] * beta[MAT2(i, param[j].first)] / zval;
					prob *= m_AlphaScale_factor;
					//prob *= m_AlphaScale[i];
					gradient[param[j].second] += prob * iter->second *sit->count;
				}
			}
			
			/** Edge factor.
			*/
			if (i > 0) {
				if (option_.inferenceType_ == TP2 || option_.inferenceType_ == TP1 || option_.inferenceType_ == ZERO1 || option_.inferenceType_ == ZERO2) {
					vector<StateParam>::iterator iter = param_.selectedStateIndex_.begin();
					for (; iter != param_.selectedStateIndex_.end(); ++iter) {
						long double a_y = alpha[MAT2(i-1, iter->y1)];
						long double b_y = beta[MAT2(i, iter->y2)];
						long double m_yy = node[MAT2(i, iter->y2)] * edge[MAT2(iter->y1, iter->y2)];
						long double prob = a_y * b_y * m_yy / zval;
						prob *= m_AlphaScale_factor2;
						gradient[iter->fid] += prob * iter->fval * sit->count;
					}
					// remaining 
					iter = param_.remainStateIndex_.begin();
					for (; iter != param_.remainStateIndex_.end(); ++iter) {
						long double a_y = alpha[MAT2(i-1, iter->y1)];
						long double b_y = beta[MAT2(i, iter->y2)];
						long double m_yy = node[MAT2(i,iter->y2)];
						if (option_.inferenceType_ == TP2 || option_.inferenceType_ == ZERO2)
							m_yy *= remainWeight_[iter->y2];
						else
							m_yy *= remainWeight_[0];
						long double prob = a_y * b_y * m_yy / zval;
						prob *= m_AlphaScale_factor2;
						if (option_.inferenceType_ == TP2 ||option_.inferenceType_ == ZERO2)
							gradient[param_.remainFeatID_[iter->y2]] += prob * iter->fval * sit->count;
						else
							gradient[param_.remainFeatID_[0]] += prob * iter->fval * sit->count;
						gradient[iter->fid] += prob * iter->fval * sit->count;
					} 				
				} 
				else {
					vector<StateParam>::iterator iter = param_.stateIndex_.begin();
					for (; iter != param_.stateIndex_.end(); ++iter) {
						long double a_y = alpha[MAT2(i-1, iter->y1)];
						long double b_y = beta[MAT2(i, iter->y2)];
						long double m_yy = node[MAT2(i,iter->y2)] * edge[MAT2(iter->y1,iter->y2)];
						long double prob = a_y * b_y * m_yy / zval;
						prob *= m_AlphaScale_factor2;
						gradient[iter->fid] += prob * iter->fval * sit->count;
					}
				}
			}

			prev_outcome = outcome;
	
		} ///< for sequence

		for (size_t c = 0; c < sit->count; c++) {
			eval.addLL(y_example_prob);	/// loglikelihood
			eval.append(reference, hypothesis);	/// evaluation (accuracy and f1 score)
		}
		
	} ///< for train_data

	avg_active_set /= (float)data->size();
	
	return gradient;
}

/** Evaluating.
*/
void LinearCRF::evaluate(Data *data, Evaluator& eval, bool standard) {
	/** Initializing.
	*/
	eval.initialize();	///< evaluator intialization
	Real edge = edgeFactor();	
		
	ofstream out;
	vector<string> state_vec;
	if (option_.outputfilename != "") {
		out.open(option_.outputfilename.c_str());
		out.precision(20);
		state_vec = param_.getState().second;
	}	
	
	/** For each sequence
	*/
    for (vector<Example>::iterator sit = data->begin(); sit != data->end(); ++sit) {
		vector<size_t> reference, hypothesis;

		/** Forward recursion.
		*/
		Real node = nodeFactor(*sit);
		if (option_.inferenceType_ == SFB) 
			pair<Real, Real> alphas = forward(node, edge);	///< Forward
		//Real &alpha = alphas.first, &alpha_scale = alphas.second;		
		//long double zval = Partition(alpha);
		
		/** Evaluating.
		*/
		vector<size_t> y_example = viterbi(node, edge, standard);

		/** For each data point.
		*/
		size_t i = 0;
		for (Example::iterator it = sit->begin(); it != sit->end(); ++it, ++i) {
			size_t outcome = it->y_;
			reference.push_back(it->y_);
			hypothesis.push_back(y_example[i]);
			
			/** For testing mode.
			*/
			if (option_.outputfilename != "") {
				out << state_vec[y_example[i]] << endl; 
			}
						
		} ///< for sequence
		if (option_.outputfilename != "")
			out << endl;

		for (size_t c = 0; c < sit->count; c++) {
			eval.append(reference, hypothesis);	/// evaluation (accuracy and f1 score)
		}
	} ///< for data

}

/** Training with maximum likelihood estimation.
*/
bool LinearCRF::trainWithMLE(Data *train_data, Data *dev_data) {
	/** Initializing.
	*/
	double* lambda = param_.getWeight();
	double* gradient = param_.getGradient();
	Evaluator train_eval(param_);	///< Evaluator
	Evaluator dev_eval(param_);	
	LBFGS lbfgs;	///< LBFGS optimizer
	double eta = 1E-05;
	timer timer_for_training;
	
	/** Logging and reporting.
	*/
	train_data->print(logger_);
	param_.print(logger_);
	logger_->report("[Parameter estimation]\n");
	logger_->report("  Method = \t\tLBFGS\n");
	logger_->report("  Regularization = \t%s\n", (option_.sigma ? (option_.L1 ? "L1":"L2") : "none"));
	logger_->report("  Penalty value = \t%.2f\n\n", option_.sigma);
	logger_->report("[Inference]\n");
	logger_->report("  Method = \t\t%s\n\n", (option_.inferenceType_ == ZEROOUT ? "ZeroOut" : 
											option_.inferenceType_ == TP1 ? "TP constant" : 
											option_.inferenceType_ == TP2 ? "TP variable" :
											option_.inferenceType_ == SFB ? "Sparse FB" :
											option_.inferenceType_ == ZERO1 ? "ZeroOut constant" :
											option_.inferenceType_ == ZERO2 ? "ZeroOut variable" :
											"Standard"));
	logger_->report("[Iterations]\n");
	logger_->report("%4s %15s %8s %8s %8s %8s\n", "iter", "loglikelihood", "acc", "micro-f1", "macro-f1", "sec");
	
	double old_obj = 1e+37;
	int converge = 0;

	/** Fast inference using Tied Potential
	*/
	if (option_.inferenceType_ == TP1) {
		param_.makeTP1(option_.eta);
		remainWeight_.clear();
		remainWeight_.push_back(exp(lambda[param_.remainFeatID_[0]]));
	} else if (option_.inferenceType_ == TP2) {
		param_.makeTP(option_.eta);
		remainWeight_.clear();
		for (size_t i = 0; i < sizeY_; i++) {
			remainWeight_.push_back(exp(lambda[param_.remainFeatID_[i]]));
		}		
	} else if (option_.inferenceType_ == SFB) {
		param_.makeActiveSet(0.0);
	}
	
	/** Training iteration.
	*/
    for (size_t niter = 0 ;niter < (int)option_.maxIter_; ++niter) {
		/** Initializing.		
		*/
		param_.initializeGradient();	///< gradient vector initialization
		train_eval.initialize();				///< evaluator intialization
        timer timer_for_iter;	///< elapsed time for one iteration
	
		/** Fast inference using Zero Out
		*/		
		if (option_.inferenceType_ == ZEROOUT) {
			param_.makeActiveSet(option_.eta);
		}
		else if (option_.inferenceType_ == ZERO1) {
			param_.makeZero1(option_.eta);
			remainWeight_.clear();
			remainWeight_.push_back(exp(lambda[param_.remainFeatID_[0]]));
		} else if (option_.inferenceType_ == ZERO2) {
			param_.makeZero2(option_.eta);
			remainWeight_.clear();
			for (size_t i = 0; i < sizeY_; i++) {
				remainWeight_.push_back(exp(lambda[param_.remainFeatID_[i]]));
			}		
		}

		/** Computing the Gradient. */
		gradient = computeGradient(train_data, train_eval);
		
		/** Regularization. */
		size_t n_nonzero = 0;
		if (option_.sigma) {
			n_nonzero = regularize(train_eval, option_.sigma, option_.L1);
		}
		
		/** Termination condition */
		double diff = (niter == 0 ? 1.0 : abs(old_obj - train_eval.getObjFunc()) / old_obj);
		if (diff < eta) 
			converge++;
		else
			converge = 0;
		old_obj = train_eval.getObjFunc();
		if (converge == 3)
			break;

		/** Optimization */
		int ret = lbfgs.optimize(param_.size(), lambda, train_eval.getObjFunc(), gradient, option_.L1, option_.sigma);
		if (ret < 0)
			return false;
		else if (ret == 0)
			return true;
		
		/** Logging and Reporting. */
		train_eval.calculateF1();
		if (dev_data->size() > 0) {
			/** Evaluating the dev set. */
			timer timer_for_eval;
			dev_eval.initialize();			
			evaluate(dev_data, dev_eval, true);
			dev_eval.calculateF1();
			logger_->report("%4d %15E %8.3f %8.3f %8.3f %8.3f  |  %8.3f %8.3f %8.3f %8.3f\n", 
				niter, train_eval.getLL(), 
				train_eval.accuracy(), train_eval.microF1()[2], train_eval.macroF1()[2], timer_for_iter.elapsed() - timer_for_eval.elapsed(), 
				dev_eval.accuracy(), dev_eval.microF1()[2], dev_eval.macroF1()[2], avg_active_set);
		} else {
			logger_->report("%4d %15E %8.3f %8.3f %8.3f %8.3f\n", niter, train_eval.getLL(),
				train_eval.accuracy(), train_eval.microF1()[2], train_eval.macroF1()[2], timer_for_iter.elapsed());
		}
		
	} ///< for iter
	logger_->report("total training time = %8.3f\n\n", timer_for_training.elapsed());
	
	timer stop_watch;	
	evaluate(dev_data, dev_eval);
	dev_eval.calculateF1();
	logger_->report("  # of data = \t\t%d\n", dev_data->size());
	logger_->report("  # of point = \t\t%d\n", dev_data->sizeElement());
	logger_->report("  testing time = \t%.3f\n\n", stop_watch.elapsed());
	logger_->report("  Acc = \t\t%8.3f\n", dev_eval.accuracy());
	logger_->report("  MicroF1 = \t\t%8.3f\n", dev_eval.microF1()[2]);
	logger_->report("  MacroF1 = \t\t%8.3f\n", dev_eval.macroF1()[2]);
	dev_eval.print(logger_);	
	
	return true;
}

/** Training with pseudo-likelihood estimation.
*/
bool LinearCRF::trainWithPL(Data *train_data, Data *dev_data) {
	double* lambda = param_.getWeight();
	double* gradient = param_.getGradient();
	LBFGS lbfgs;	///< LBFGS optimizer
	Evaluator train_eval(param_);	///< Evaluator
	Evaluator dev_eval(param_);	
	double old_obj = 1e+37;
	int converge = 0;
	double eta = 1E-05;
	timer timer_for_training;

	/** Logging and reporting.
	*/
	param_.print(logger_);
	logger_->report("[Parameter estimation]\n");
	logger_->report("  Method = \t\tPL\n");
	logger_->report("  Regularization = \t%s\n", (option_.sigma ? (option_.L1 ? "L1":"L2") : "none"));
	logger_->report("  Penalty value = \t%.2f\n\n", option_.sigma);
	logger_->report("[Iterations]\n");
	logger_->report("%4s %15s %8s %8s %8s %8s\n", "iter", "loglikelihood", "acc", "micro-f1", "macro-f1", "sec");
	
	/** Training iteration.
	*/
	size_t maxIter_ = (option_.initType_ == 1 ? option_.initIter_ : option_.maxIter_);
    for (size_t niter = 0 ;niter < (int)maxIter_; ++niter) {
		/** Initializing.
		*/
		param_.initializeGradient();	///< gradient vector initialization
		train_eval.initialize();	///< evaluator intialization
		dev_eval.initialize();	///< evaluator intialization
        timer timer_for_iter;	///< elapsed time for one iteration
		
		/** For pseudo-likelihood.
		*/				
		for (vector<StateParam>::iterator iter = param_.stateIndex_.begin(); 
			iter != param_.stateIndex_.end(); ++iter) {		
			gradient[iter->fid] *= 2.0;
		}
			
		/** Repeat: for each sequence.
		*/
        for (vector<Example>::iterator sit = train_data->begin(); sit != train_data->end(); ++sit) {
			size_t prev_outcome = defaultY_;
			size_t next_outcome = defaultY_;
			vector<size_t> reference, hypothesis;
			
			/** Repeat: for each data point.
			*/
			size_t i = 0;
			for (Example::iterator it = sit->begin(); it != sit->end(); ++it, ++i) {	 
				if (i < sit->size() - 1)
					next_outcome = sit->at(i+1).y_;
				else
					next_outcome = defaultY_;
				
				/** Evaluation for pseudo-likelihood.
				*/
				size_t max_outcome = 0;
				Real prob(sizeY_);
				fill(prob.begin(), prob.end(), 0.0);
				
				/** Node factors.
				*/
				for (vector<pair<size_t, double> >::iterator iter = it->x_.begin(); 
						iter != it->x_.end(); ++iter) {
					vector<pair<size_t, size_t> >& param = param_.paramIndex_[iter->first];
					for (size_t j = 0; j < param.size(); ++j) {
						prob[param[j].first] += lambda[param[j].second] * iter->second;
					}
				}
				/** Edge factors.
				*/
				if (i > 0) {
					for (vector<StateParam>::iterator iter = param_.stateIndex_.begin(); 
							iter != param_.stateIndex_.end(); ++iter) {
						/// y_{t-1} edge factor.
						if (iter->y1 == prev_outcome)
							prob[iter->y2] += lambda[iter->fid] * iter->fval;
					}
				}
				if (i < sit->size() -1) {
					for (vector<StateParam>::iterator iter = param_.stateIndex_.begin(); 
							iter != param_.stateIndex_.end(); ++iter) {
						/// y_{t+1} edge factor.
						if (iter->y2 == next_outcome)
							prob[iter->y1] += lambda[iter->fid] * iter->fval;
					}
				}
				
				/** Nomalizing.
				*/
				double sum = 0.0;
				double max = 0.0;
				for (size_t j=0; j < param_.sizeStateVec(); j++) {
					prob[j] = exp(prob[j]); 
					sum += prob[j];
					if (prob[j] > max) {
						max = prob[j];
						max_outcome = j;
					}
				}
				for (size_t j=0; j < param_.sizeStateVec(); j++) {
					prob[j] /= sum;
				}

				reference.push_back(it->y_);
				hypothesis.push_back(max_outcome);
				
				/** Calculate the expectation.
				*/
				for (vector<pair<size_t, double> >::iterator iter = it->x_.begin(); 
						iter != it->x_.end(); iter++) {
					vector<pair<size_t, size_t> >& param = param_.paramIndex_[iter->first];
					for (size_t j = 0; j < param.size(); ++j) {
						gradient[param[j].second] += prob[param[j].first] * iter->second * sit->count;
					}
				}
				if (i > 0) {
					for (vector<StateParam>::iterator iter = param_.stateIndex_.begin(); 
						iter != param_.stateIndex_.end(); ++iter) {
						if (iter->y1 == prev_outcome)
							gradient[iter->fid] += prob[iter->y2] * iter->fval * sit->count;
					}
				}
				if (i < sit->size() -1) {
					for (vector<StateParam>::iterator iter = param_.stateIndex_.begin(); 
						iter != param_.stateIndex_.end(); ++iter) {
						if (iter->y2 == next_outcome)
							gradient[iter->fid] += prob[iter->y1] * iter->fval * sit->count;
					}
				}
		
				/** loglikelihood.
				*/
				for (size_t c = 0; c < sit->count; c++) {
					train_eval.addLL(prob[it->y_]);	
				}

				prev_outcome = it->y_;

			} ///< for example
			
			/** Evaluation.
			*/
			for (size_t c = 0; c < sit->count; c++) {
				train_eval.append(reference, hypothesis);
			}
			
		} ///< for train_data
		
		/** Evaluating the dev set.
		*/
		//Evaluate(dev_data, dev_eval);
		
		/** Regularization.
		*/
		size_t n_nonzero = 0;
		if (option_.sigma) 
			n_nonzero = regularize(train_eval, option_.sigma, option_.L1);
		
		/** Termination condition.
		*/
		double diff = (niter == 0 ? 1.0 : abs(old_obj - train_eval.getObjFunc()) / old_obj);
		if (diff < eta) 
			converge++;
		else
			converge = 0;
		old_obj = train_eval.getObjFunc();
		if (converge == 3)
			break;
		
		/** Optimization.
		*/
		int ret = lbfgs.optimize(param_.size(), lambda, train_eval.getObjFunc(), gradient, option_.L1, option_.sigma);
		if (ret < 0)
			return false;
		else if (ret == 0)
			return true;

		/** Logging and reporting.
		*/
		train_eval.calculateF1();
		/*
		if (dev_data->size() > 0) {
			dev_eval.CalculateF1();
			logger_->Report("%4d %15E %8.3f %8.3f %8.3f %8.3f  |  %8.3f %8.3f %8.3f\n", 
				niter, train_eval.GetLL(), 
				train_eval.Accuracy(), train_eval.MicroF1()[2], train_eval.MacroF1()[2], timer_for_iter.elapsed(), 
				dev_eval.Accuracy(), dev_eval.MicroF1()[2], dev_eval.MacroF1()[2]);
		} else {
		*/
			logger_->report("%4d %15E %8.3f %8.3f %8.3f %8.3f\n", niter, train_eval.getLL(),
				train_eval.accuracy(), train_eval.microF1()[2], train_eval.macroF1()[2], timer_for_iter.elapsed());
		//}

	} ///< for iter
	
	logger_->report("total training time = %8.3f\n", timer_for_training.elapsed());
	
	return true;
}

/** Training.
*/
bool LinearCRF::train(Data *train_data, Data *dev_data, Option option) { 
	option_ = option;

	if (option_.useInitializer_) {
		trainWithPL(train_data, dev_data);
	}	
	return trainWithMLE(train_data, dev_data); 
}

/** Testing.
*/
bool LinearCRF::test(Data* test_data, Option option) {
	/** Initializing.
	*/
	double* lambda = param_.getWeight();	
	Evaluator test_eval(param_);
	test_eval.initialize();
	timer stop_watch;
	option_ = option;
	
	logger_->report("[Testing begins ...]\n");
	
	/** Evaluating the test data.
	*/
	if (option_.inferenceType_ == ZEROOUT) {
		param_.makeActiveSet(option_.eta);	
	} else if (option_.inferenceType_ == ZERO1) {
		param_.makeZero1(option_.eta);
		remainWeight_.clear();
		remainWeight_.push_back(exp(lambda[param_.remainFeatID_[0]]));
	} else if (option_.inferenceType_ == ZERO2) {
		param_.makeZero2(option_.eta);
		remainWeight_.clear();
		for (size_t i = 0; i < sizeY_; i++) {
			remainWeight_.push_back(exp(lambda[param_.remainFeatID_[i]]));
		}		
	}
			
	evaluate(test_data, test_eval);

	/** Logging and reporting.
	*/
	test_eval.calculateF1();
	logger_->report("  # of data = \t\t%d\n", test_data->size());
	logger_->report("  # of point = \t\t%d\n", test_data->sizeElement());
	logger_->report("  testing time = \t%.3f\n\n", stop_watch.elapsed());
	logger_->report("  Acc = \t\t%8.3f\n", test_eval.accuracy());
	logger_->report("  MicroF1 = \t\t%8.3f\n", test_eval.microF1()[2]);
	logger_->report("  MacroF1 = \t\t%8.3f\n", test_eval.macroF1()[2]);
	
	test_eval.print(logger_);
}

/** Regularization.
	@param eval	A evaluator
	@param sigma	A hyper-parameter for regularization
	@param L1	L1 or L2 method
*/
size_t LinearCRF::regularize(Evaluator& eval, double sigma, bool L1) {
	/** Initializing.
	*/
	double* lambda = param_.getWeight();
	double* gradient = param_.getGradient();
	size_t n_nonzero = 0;
	
	/** L1-Regularization.
	*/
	if (L1) { 
		for (size_t i = 0; i < param_.size(); ++i) {
			eval.subtractLL(abs(lambda[i] / sigma));
			if (lambda[i] != 0.0) 
				n_nonzero++;
		}
	}
	/** L2-Regularization.
	*/
	else {	
		n_nonzero = param_.size();
		for (size_t i = 0; i < param_.size(); ++i) {
			gradient[i] += lambda[i] / sigma;
			eval.subtractLL((lambda[i] * lambda[i]) / (2.0 * sigma));
		}
	}
    
    /** Return the number of non-zero parameter.
    	In practice, L1-regularization could reduce the active parameter size.
    	See Gao et al., ACL 2007.
    */
    return n_nonzero;
}

}
