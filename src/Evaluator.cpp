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
#include <numeric>
#include "Evaluator.h"
#include "Utility.h"

namespace fastcrf {

using namespace std;

Evaluator::Evaluator() {
	initialize();
}

Evaluator::Evaluator(Parameter& param, bool bio) {
	useBioTagInFirst_ = true;
	initialize();
	encode(param, bio);
}

void Evaluator::initialize() {
	/// initialize
	nCorrect_ = 0;
	nPoint_ = 0;
	nExample_ = 0;
	nClass_ = classVec_.size();
	accuracy_ = 0.0;
	loglikelihood = 0.0;
	nTruePhrase_ = 0;
	nGuessPhrase_ = 0;
	nCorrectPhrase_ = 0;
	
	/// F1 score
	if (nClass_ > 0) {
		trueClass_.resize(nClass_, 0);
		fill(trueClass_.begin(), trueClass_.end(), 0);
		guessClass_.resize(nClass_, 0);
		fill(guessClass_.begin(), guessClass_.end(), 0);
		correctClass_.resize(nClass_, 0);
		fill(correctClass_.begin(), correctClass_.end(), 0);
		precision_.resize(nClass_, 0.0);
		fill(precision_.begin(), precision_.end(), 0.0);
		recall_.resize(nClass_, 0.0);
		fill(recall_.begin(), recall_.end(), 0.0);
		scoreF1_.resize(nClass_, 0.0);
		fill(scoreF1_.begin(), scoreF1_.end(), 0.0);
	}
}

/** Encode the class information
*/
void Evaluator::encode(Parameter& param, bool bio) {
	map<string, size_t> stateMap_ = param.getState().first;
	vector<string> stateVec_ = param.getState().second;
	outsideClass_ = param.getDefaultState();
	
	if (!bio) {	/// does not use BIO encoding scheme
		classMap_ = stateMap_;
		classVec_ = stateVec_;
		for (size_t i = 0; i < classVec_.size(); ++i) {
			//if (i != outsideClass_)
				bioIndex_.push_back(i);
				beginMap_.insert(make_pair(i, 1));				
		}
		useBioTag_ = false;
		
	} else {	 /// use BIO encoding scheme
		size_t index1 = 0, index2 = 1;
		if (useBioTagInFirst_) {
			index1 = 1;
			index2 = 0;
		}
		
		vector<string>::iterator it = stateVec_.begin();
		for (; it != stateVec_.end(); ++it) {
			//if (*it == "O")
			//	continue;
			vector<string> tok = Tokenize(*it, "-");
			if (tok.size() > 1 && (tok[index2] == "B" || tok[index2] == "I")) {
				if (classMap_.find(tok[index1]) == classMap_.end()) {
					classMap_.insert(make_pair(tok[index1], classVec_.size()));
					bioIndex_.push_back(classVec_.size());
					classVec_.push_back(tok[index1]);
				} else 
					bioIndex_.push_back(classMap_[tok[index1]]); 
					
				if (tok[index2] == "B")
					beginMap_.insert(make_pair(stateMap_[*it], 1));
			} else {
				classMap_.insert(make_pair(*it, classVec_.size()));
				bioIndex_.push_back(classVec_.size());
				classVec_.push_back(*it);
				beginMap_.insert(make_pair(stateMap_[*it], 1));				
			}
		} // for
		useBioTag_ = true;
	}	// if else
	
	/** Out of class 
	*/
	classMap_.insert(make_pair("!OUT_OF_CLASS!", classVec_.size()));
	OUT_OF_CLASS = classVec_.size();
	bioIndex_.push_back(classVec_.size());	
	classVec_.push_back("!OUT_OF_CLASS!");
}

/** Chunking the sequence.
	Using BIO encoding, this function does chunking for a given sequence.
*/
vector<pair<size_t, pair<size_t, size_t> > > Evaluator::chunking(vector<size_t> example) {
	vector<pair<size_t, pair<size_t, size_t> > > phrase;	///< return

	size_t label = outsideClass_, spos = 0, epos = 0;
	bool isinphrase = false, isempty = true;
	
	for (size_t i = 0; i < example.size(); i++) {
		/* Outside class */
		if (example[i] == outsideClass_) {	
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			isempty = true;
			isinphrase = false;
			continue;
		} 

		/* B-X class */
		if (beginMap_.find(example[i]) != beginMap_.end()) {
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			label = bioIndex_[example[i]];
			spos = i; epos = i;
			isempty = false;
			isinphrase = true;
			continue;
		}	
		
		/* I-X class within chunk */
		if (isinphrase) {	
			/* I-X class but boundary */
			if (label != bioIndex_[example[i]]) {	 
				if (!isempty) 
					phrase.push_back(make_pair(label, make_pair(spos, epos)));
				label = bioIndex_[example[i]];
				spos = i; epos = i;
			/* I-X class within chunk */
			} else 
				epos = i;		
			isempty = false;
		/* I-X class but chunk boundary */
		} else {
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			label = bioIndex_[example[i]];
			spos = i; epos = i;
			isempty = false;
			isinphrase = true;
		}
	}
	if (!isempty)
		phrase.push_back(make_pair(label, make_pair(spos, epos)));

	return phrase;
}

size_t Evaluator::append(Parameter& param, vector<string> ref, vector<string> hyp) {
	vector<size_t> ref_d, hyp_d;
	map<string, size_t> stateMap_ = param.getState().first;
	
	assert(ref.size() == hyp.size());
	for (size_t i = 0; i < ref.size(); i++) {
		if (stateMap_.find(ref[i]) != stateMap_.end()) 
			ref_d.push_back(stateMap_[ref[i]]);
		else 
			ref_d.push_back(OUT_OF_CLASS);
				
		if (stateMap_.find(hyp[i]) != stateMap_.end()) 
			hyp_d.push_back(stateMap_[hyp[i]]);
		else 
			hyp_d.push_back(OUT_OF_CLASS);
	}
	
	assert(ref_d.size() == hyp_d.size());
	return append(ref_d, hyp_d);
}

/** Append the reference and hypothesis.
*/
size_t Evaluator::append(vector<size_t> ref, vector<size_t> hyp) {

	// accuracy_
	assert(ref.size() == hyp.size());
	for (size_t i = 0; i < ref.size(); i++) {
		if (ref[i] == hyp[i]) 
			nCorrect_ ++;
		nPoint_ ++;
	}
	nExample_ ++;
	
	/// f1-score
	size_t g_index, t_index;
	if (useBioTag_) { /// bio enconding
		vector<pair<size_t, pair<size_t, size_t> > > ref_class, hyp_class;
		ref_class = chunking(ref);
		hyp_class = chunking(hyp);
		/// for reference
		vector<pair<size_t, pair<size_t, size_t> > >::iterator rit = ref_class.begin();
		for (; rit != ref_class.end(); ++rit )  {
			assert(rit->first < trueClass_.size());
			trueClass_[rit->first] ++;
			nTruePhrase_ ++;
		}
		
		/// for hypothesis
		vector<pair<size_t, pair<size_t, size_t> > >::iterator hit = hyp_class.begin();
		for (; hit != hyp_class.end(); ++hit ) {
			assert(hit->first < guessClass_.size());		
			guessClass_[hit->first] ++;
			nGuessPhrase_ ++;
		}
		/// correct 
		rit = ref_class.begin();
		hit = hyp_class.begin();
		for (; rit != ref_class.end() && hit != hyp_class.end(); ) {
			g_index = hit->first;
			t_index = rit->first;
			if (rit->second.first == hit->second.first && rit->second.second == hit->second.second) {
				if (g_index == t_index) {
					correctClass_[t_index] ++;
					nCorrectPhrase_ ++;
				}
				++rit; ++hit;
			} else if (rit->second.second < hit->second.first) {
				++rit;
			} else if (rit->second.first > hit->second.second) {
				++hit;
			} else {
				++rit; ++hit;
			}		
		}
	} else { ///< no bio encoding
		for (size_t i = 0; i < ref.size(); i++) {
			g_index = hyp[i];
			t_index = ref[i];
			guessClass_[g_index] ++;
			nCorrectPhrase_ ++;
			trueClass_[t_index] ++;
			nTruePhrase_ ++;			
			
			if (g_index == t_index) {
				correctClass_[t_index] ++;
				nCorrectPhrase_ ++;
			}
		}
	}
	
	return nExample_;
}

void Evaluator::calculateF1() {
	/// per-class f1-scores
	size_t num_data = 0;
	microPrecision_ = 0.0;
	microRecall_ = 0.0;
	microF1_ = 0.0;
	macroPrecision_ = 0.0;
	macroRecall_ = 0.0;
	macroF1_ = 0.0;
	for (size_t i = 0; i < classVec_.size(); i++) {
		if (guessClass_[i] == 0 || correctClass_[i] == 0)
			precision_[i] = 0.0;
		else 
			precision_[i] = (double)correctClass_[i] * 100.0 / guessClass_[i];
		if (trueClass_[i] == 0 || correctClass_[i] == 0)
			recall_[i] = 0.0;
		else
			recall_[i] = (double)correctClass_[i] * 100.0 / trueClass_[i];
		if ((precision_[i] + recall_[i]) == 0.0)
			scoreF1_[i] = 0.0;
		else
			scoreF1_[i] = 2.0 * (precision_[i] * recall_[i]) / (precision_[i] + recall_[i]);
		/*
		if (i != OUT_OF_CLASS && i != outsideClass_) {
			microPrecision_ += precision_[i] * trueClass_[i];
			microRecall_ += recall_[i] * trueClass_[i];
			num_data += trueClass_[i];
		}
		*/
	}
	macroPrecision_ = accumulate(precision_.begin(), precision_.end(), 0.0) / nClass_;
	macroRecall_ = accumulate(recall_.begin(), recall_.end(), 0.0) / nClass_;
	if ((macroPrecision_ + macroRecall_) != 0.0)
		macroF1_ = 2.0 * (macroPrecision_ * macroRecall_) / (macroPrecision_ + macroRecall_);
	/*
	microPrecision_ /= num_data;
	microRecall_ /= num_data;
	*/
	
	if (nGuessPhrase_ > 0)
		microPrecision_ = 100.0 * (double)nCorrectPhrase_ / (double)nGuessPhrase_;
	if (nTruePhrase_ > 0)
		microRecall_ = 100.0 * (double)nCorrectPhrase_ / (double)nTruePhrase_;
	if ((microPrecision_ + microRecall_) != 0.0)
		microF1_ = 2.0 * (microPrecision_ * microRecall_) / (microPrecision_ + microRecall_);
			
}

double Evaluator::addLL(double p) {
	double t;
	if (p == 0.0)
		t = LOG_ZERO;
	else
		t = (double)log(p);
	if (finite(t))
		loglikelihood -= t;
	else
		loglikelihood -= LOG_ZERO;

	return loglikelihood;
}

double Evaluator::subtractLL(double p) {
	loglikelihood += p;
	return loglikelihood;
}

double Evaluator::getObjFunc() {
	return loglikelihood;
}

double Evaluator::getLL() {
	return -loglikelihood;
}

double Evaluator::accuracy() {
	accuracy_ = nCorrect_ / (double)nPoint_ * 100.0;
	return accuracy_;
}

vector<double> Evaluator::macroF1() {
	vector<double> ret;	 ///< return vector
	ret.push_back(macroPrecision_);
	ret.push_back(macroRecall_);
	ret.push_back(macroF1_);
	return ret;
}

vector<double> Evaluator::microF1() {
	vector<double> ret;	 ///< return vector
	ret.push_back(microPrecision_);
	ret.push_back(microRecall_);
	ret.push_back(microF1_);
	return ret;
}

size_t Evaluator::sizeClass() {
	return nClass_;
}

void Evaluator::print(Logger *logger) {
	logger->report("[Result]\n");
	logger->report("Accuracy: %6.2f%%: prec: %6.2f%%; rec: %6.2f%%; F1: %6.2f\n", 
		accuracy(), microPrecision_, microRecall_, microF1_);
	for (size_t i = 0; i < nClass_; i++) {
		if (i != classMap_["O"] && i != OUT_OF_CLASS)
			logger->report("%17s: prec: %6.2f%%; rec: %6.2f%%; F1: %6.2f\n", 
				classVec_[i].c_str(), precision_[i], recall_[i], scoreF1_[i]);
	}
}

}
