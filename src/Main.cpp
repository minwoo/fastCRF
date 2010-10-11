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
#include "LinearCRF.h"

using namespace std;

/** Command line tool */
int main(int argc, char** argv) {
	////////////////////////////////////////////////////////////////
	///	 Model
	////////////////////////////////////////////////////////////////
	fastcrf::StructuredModel *model;
	fastcrf::Data *train_data, *dev_data, *test_data;

	////////////////////////////////////////////////////////////////
	///	 Parameters
	////////////////////////////////////////////////////////////////
	vector<string> model_file, train_file, dev_file, test_file, output_file;
	string template_file = "";
	string initialize_method, estimation_method;
	size_t maxIter, cutoff;
	double l1_prior, l2_prior;
	enum {MaxEnt = 0, LinearCRF} model_type;
	bool train_mode = false;
	bool confidence = false;
	bool first_is_true_label = false, with_label = true;
	string outside_label = "O";
	fastcrf::Option option;

	////////////////////////////////////////////////////////////////
	///	 Reading the configuration file
	////////////////////////////////////////////////////////////////
	char config_filename[128];
	if (argc > 1) {
		strcpy(config_filename, (char*)argv[1]);
	} else {
		cout << HEADER;
		cout << "[Usage] argmax config_file \n\n";
		exit(1);		
	}
	
	fastcrf::Configurator config(config_filename);

	////////////////////////////////////////////////////////////////
	///	 logger_
	////////////////////////////////////////////////////////////////
	fastcrf::Logger *log = NULL;
	if (config.isValid("log_file")) {
		size_t log_mode = 2;
		if (config.isValid("log_mode"))
			log_mode = atoi(config.get("log_mode").c_str());
		log = new fastcrf::Logger(config.get("log_file"), log_mode);
		log->report(HEADER);
		log->report("[Configurating]\n");
		log->report(" Configuration File = %s\n\n", config.getFilename().c_str());
	}
	
	////////////////////////////////////////////////////////////////
	///	 Selecting the model
	////////////////////////////////////////////////////////////////
	if (config.isValid("model_type")) {
		string type_str = config.get("model_type");
		if (1) {
			model_type = LinearCRF;
			if (log != NULL)
				model = new fastcrf::LinearCRF(log);
			else
				model = new fastcrf::LinearCRF();
			train_data = new fastcrf::Data(model->getParameter());
			dev_data = new fastcrf::Data(model->getParameter());
			test_data = new fastcrf::Data(model->getParameter());
			log->report("[Model = Linear-chain CRF]\n\n");

		}
	}	

	////////////////////////////////////////////////////////////////
	///	 Data Files
	////////////////////////////////////////////////////////////////
	if (config.isValid("train_file"))
		train_file = config.gets("train_file");
	if (config.isValid("dev_file"))
		dev_file = config.gets("dev_file");
	if (config.isValid("test_file"))
		test_file = config.gets("test_file");
	if (config.isValid("template"))
		template_file = config.get("template");
	if (config.isValid("cutoff"))
		cutoff = atoi(config.get("cutoff").c_str());
	if (config.isValid("true_label")) {
		if (config.get("true_label") == "first")
			first_is_true_label = true;
		else
			first_is_true_label = false; // default is last
		with_label = true;
	} else
		with_label = false;
		
	////////////////////////////////////////////////////////////////
	///	 Model File
	////////////////////////////////////////////////////////////////
	if (config.isValid("model_file")) {
		model_file = config.gets("model_file");
	}

	////////////////////////////////////////////////////////////////
	///	 Pruning
	////////////////////////////////////////////////////////////////
	if (config.isValid("prune")) {
		double prune = atof(config.get("prune").c_str());
		//model->setPrune(prune);
	}
	
	////////////////////////////////////////////////////////////////
	///	 Inference Type
	////////////////////////////////////////////////////////////////
	if (config.isValid("inference_type")) {
		if (config.get("inference_type") == "zeroout") 
			option.inferenceType_ = fastcrf::ZEROOUT;
		else if (config.get("inference_type") == "sparsefb") 
			option.inferenceType_ = fastcrf::SFB;
		else if (config.get("inference_type") == "tp1") 
			option.inferenceType_ = fastcrf::TP1;
		else if (config.get("inference_type") == "tp2") 
			option.inferenceType_ = fastcrf::TP2;
		else if (config.get("inference_type") == "zero1") 
			option.inferenceType_ = fastcrf::ZERO1;
		else if (config.get("inference_type") == "zero2") 
			option.inferenceType_ = fastcrf::ZERO2;
		else 
			option.inferenceType_ = fastcrf::STANDARD;
	}
	if (config.isValid("eta")) {
		option.eta = atof(config.get("eta").c_str());	
	}
	
	////////////////////////////////////////////////////////////////
	///	 Initializing the parameter
	////////////////////////////////////////////////////////////////
	if (config.isValid("initialize")) {
		option.useInitializer_ = true;
		if (config.get("initialize") == "MLE") 
			option.initType_ = fastcrf::MLE;
		else 
			option.initType_ = fastcrf::PL;
		if (config.isValid("initialize_iter"))
			option.initIter_ = atoi(config.get("initialize_iter").c_str());			
	}

	////////////////////////////////////////////////////////////////
	///	 Mode
	////////////////////////////////////////////////////////////////
	if (config.isValid("mode")) 
		train_mode = (config.get("mode") == "train" ? true : false);

	////////////////////////////////////////////////////////////////
	///	 Training mode
	////////////////////////////////////////////////////////////////
	if (train_mode) { 
		assert(train_file.size() == model_file.size());
		if (train_file.size() <= 0) {
			cerr << "Invalid setting. Please see the configuration\n";
			return -1;
		}
		
		for (size_t iter = 0; iter < train_file.size(); iter++) {
			log->report("\n\nTraining File = %s\n\n", train_file[iter].data());
			model->clear();
			train_data->clear();
			train_data->read(train_file[iter], template_file, first_is_true_label);
			if (config.isValid("outside_label")) 
				model->getParameter()->setDefaultState(config.get("outside_label"));
			if (config.isValid("cutoff"))
				train_data->cutFeature(cutoff);
			model->initializeModel();	// initialize the model
			if (dev_file.size() != 0) {
				assert(train_file.size() == dev_file.size());
				dev_data->clear();
				dev_data->read(dev_file[iter], template_file, first_is_true_label, false);
			}
			
			if (config.isValid("iter"))
				option.maxIter_ = atoi(config.get("iter").c_str());
				
			string type_str = "LBFGS-L2";	///< default estimation method
			if (config.isValid("estimation")) {
				type_str = config.get("estimation");
			}

			if (type_str == "LBFGS-L1") {
				option.L1 = true;
				/// LBFGS-L1
				if (config.isValid("l1_prior"))
					option.sigma = atof(config.get("l1_prior").c_str());

			} else { 
				/// LBFGS-L2
				if (config.isValid("l2_prior"))
					option.sigma = atof(config.get("l2_prior").c_str());
			}
			
			/** training */
			if (!model->train(train_data, dev_data, option)) {
				cerr << "training terminates with error\n\n";
			
				return -1;
			}

			if (config.isValid("model_file")) 
				model->saveModel(model_file[iter]);

		} // iteration

	} 
	////////////////////////////////////////////////////////////////
	///	 Testing mode
	////////////////////////////////////////////////////////////////	
	else { 
		assert(test_file.size() == model_file.size());
		if (model_file.size() == 0 || test_file.size() == 0) {
			cerr << "Invalid setting. Please see the configuration\n";
			return -1;
		}
		
		if (config.isValid("output_file")) {
			output_file = config.gets("output_file");
			assert(test_file.size() == output_file.size());
			if (config.isValid("confidence"))
				option.with_confidence = (config.get("confidence") == "true" ? true : false);
		}

		for (size_t iter = 0; iter < test_file.size(); iter++) {
			log->report("\n\nTest File = %s\n\n", test_file[iter].data());
			model->clear();
			if (!model->loadModel(model_file[iter])) {
				cerr << "Model loading error\n";
				return -1;
			}
			if (config.isValid("outside_label")) 
				model->getParameter()->setDefaultState(config.get("outside_label"));
			test_data->clear();
			test_data->read(test_file[iter], template_file, first_is_true_label, false);
			if (config.isValid("output_file")) {
				option.outputfilename = output_file[iter];
				model->test(test_data, option);
			} else
				model->test(test_data, option);
		}
	}

}
