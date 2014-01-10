/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "Data.h"
#include "Utility.h"

using namespace std;

namespace fastcrf {

Data::Data(Parameter *param, bool ignoreState) {
	param_ = param;
	ignoreState_ = ignoreState;
	sizeElement_ = 0;
}

void Data::append(Example example) { 
	push_back(example); 
	sizeElement_ += example.size(); 
}

size_t Data::sizeElement() { 
	return sizeElement_; 
}

void Data::cutFeature(float k) {
	param_->cutOff(k);
	param_->finalize();
}

void Data::print(Logger *logger) {
	logger->report("[Example Data]\n");
	logger->report("  # of Data = \t%d\n", size());
	logger->report("  # of Point = \t%d\n", sizeElement());
}
	
Point Data::pack(vector<string>& tokens, bool isUpdate, bool hasLabel) {
	Point event;
	vector<string>::iterator it = tokens.begin();

	// y
	if (hasLabel) {
		string fstr(it->c_str());
		vector<string> tok = Tokenize(fstr, ":");
		float fval = 1.0;
		if (tok.size() > 1) {
			fval = atof(tok[1].c_str());	
			fstr = tok[0];
		}
		if (isUpdate) { 
			event.y_ = param_->addNewState(fstr);	
		} else { 
			int class_id = param_->findState(fstr);
			if ( class_id >= 0 )
				event.y_ = class_id;
			else 
				event.y_ = param_->sizeStateVec();
		}
		event.val_ = fval;

		++it;
	} 
	
	// x
	for ( ; it != tokens.end() ; it++) {
		string fstr(it->c_str()); 
		vector<string> tok = Tokenize(fstr, ":");
		double fval = 1.0;
		if ( tok.size() > 1 ) {
			fval = atof(tok[1].c_str());	
			fstr = tok[0];
		}
		if (isUpdate) { 
			size_t pid = param_->addNewObs(fstr);
			event.x_.push_back(make_pair(pid, fval));
			/* too naive implementation */
			//for (size_t i = 0; i < param_->SizeStateVec(); i++) 
			//{
				//if (i == event.y_)
					param_->updateParameter(event.y_, pid, fval);
				//else
				//	param_->UpdateParameter(i, pid, 0.0);
			//}
		} else { 
			int pid = param_->findObs(fstr);
			if ( pid >= 0 )
				event.x_.push_back(make_pair((size_t)pid, fval));
		}
	}
	
	return event;
}

bool Data::read(const string& filename, const string& templateFilename, bool isLabelInFirst, bool isUpdate, bool hasLabel) {
	map<vector<vector<string> >, size_t> tempDataMap;	
	vector<vector<string> > tokenList;
	vector<string> templateList;	
	size_t tempDataCount = 0;

	string line;
	Example example;	
	
	// Template file loading
	if (templateFilename != "") {
		ifstream f(templateFilename.c_str());
		if ( !f )
			throw runtime_error("cannot open template file");
		while ( getline(f, line) ) {
			if ( !line.empty() ) {
				if (line[0] == '#' || line[0] == 'B')
					continue;
				templateList.push_back(line);
			}
		}
	}

	// File stream
	ifstream f(filename.c_str());
	if (!f)
		throw runtime_error("cannot open data file");
	
	// Make a state space Y
	if ( isUpdate ) {
		while ( getline(f, line) ) {
			if ( !line.empty() ) {
				vector<string> tokens = Tokenize(line);
				if (tokens.size() > 0 && hasLabel) {
					size_t index = 0;
					if (!isLabelInFirst)
						index = tokens.size()-1;
					string fstr(tokens[index]);
					vector<string> tok = Tokenize(fstr, ":");
					float fval = 1.0;
					if (tok.size() > 1) {
						fval = atof(tok[1].c_str());	///< feature value
						fstr = tok[0];
					}
					param_->addNewState(fstr);	// outcome id
				}
			}
		}
		
		f.clear();
		f.seekg(0, ios::beg);
	}

	// Read from data file
	size_t count = 0;
	string prev_label = "";	
    	
	string ofilename = filename + ".dat";
	ofstream ofs(ofilename.c_str());
    	ofs.precision(20);
	
	while ( getline(f,line) ) {
		if ( line.empty() ) {
			if (tokenList.size() > 0) {
				example.clear();
				prev_label = "";
				for (size_t i = 0; i < tokenList.size(); i++) {
					vector<string> tokens; 
					if (templateList.size() > 0) 
						tokens = applyTemplate(tokenList, i, templateList);
					else
						tokens = tokenList[i];
					
					// *****************************
					for (int x = 0; x < tokens.size(); x++)
						ofs << tokens[x] << " ";
					ofs << "\n";
	
					// Pack the event 
					Point event = pack(tokens, isUpdate, hasLabel);	
					example.push_back(event);
					
					// State transition feature
					if (isUpdate && !ignoreState_) {
						if (prev_label != "") {
							size_t pid = param_->addNewObs("@" + prev_label);
							for (size_t i = 0; i < param_->sizeStateVec(); i++) {
								if (i == event.y_)
									param_->updateParameter(event.y_, pid, event.val_);
								else
									param_->updateParameter(i, pid, 0.0);
							}
						}
						prev_label = tokens[0];
					}
				}
				ofs << "\n";
			}
			
			// processing for equivalent sentence
			if (tempDataMap.find(tokenList) == tempDataMap.end()) {
				example.count = 1.0;
				append(example);
				tempDataMap.insert(make_pair(tokenList, tempDataCount));
				tempDataCount++;				
			} else {
				at(tempDataMap[tokenList]).count += 1.0;
			}
			tokenList.clear();
			++count;
		} else {
			tokenList.push_back(Tokenize(line));
		}

	}	// end of while
	
	// Finalize the parameter isUpdate
	if (isUpdate) {
		//param_->CutOff(0);	
		param_->finalize();
	}
		
	return true;
}

vector<string> Data::applyTemplate(vector<vector<string> > tokenList, size_t curPos, vector<string> templateList) {
	vector<string> point, cur_raw = tokenList[curPos];
	int size = tokenList.size();
	
	// label
	point.push_back(cur_raw[cur_raw.size()-1]);
	
	// for each template 
	for (vector<string>::iterator it = templateList.begin();
			it != templateList.end(); it++) {
		bool valid_feature = true;
		char output[1024], pos = 0;		
		const char* ch = it->c_str();
		for (; *ch; ch++) {
			if (*ch == '%') {
				ch+=2; // %x
				if (*ch != '[')
					throw runtime_error("Invalid template file");
				ch++;
				char temp[16], cpos = 0;
				for (;*ch != ']'; ++ch) 
					temp[cpos++] = *ch;
				temp[cpos++] = '\0';
				vector<string> tok = Tokenize(temp, ",");
				assert(tok.size() == 2);
				int row = atoi(tok[0].c_str()), col = atoi(tok[1].c_str());
				
				string str;
				if ( (curPos + row) >= 0 && (curPos + row) < size ) {
					vector<string> raw = tokenList[curPos + row];
					if ( col < raw.size() )
						str = raw[col];
				} else {
					valid_feature = false;
					break;
				}
				
				strcpy(&output[pos], str.c_str());
				pos += str.size();				
			} 
			else 
				output[pos++] = *ch;
		}
		output[pos++] = '\0';
		if (valid_feature)
			point.push_back(output);
	}
	
	return point;
}


}
