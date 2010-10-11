/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __DATA_H__
#define __DATA_H__

#include <vector>
#include <string>
#include <map>
#include "Parameter.h"

namespace fastcrf {

using namespace std;

/** Point */
struct Point {
	size_t y_;
	double val_;
	vector<pair<size_t, double> > x_;
};

/** Example - A vector that contains a collection of points */
class Example : public vector<Point> {
public:
	float count;
};

/** Data - A vector that contains a collection of examples */
class Data : public vector<Example> {
public:
	Data(Parameter *param, bool ignoreState = false);

	bool read(const string& filename, const string& templateFilename = "", bool isFirst = false, bool isUpdate = true, bool hasLabel = true);

	void append(Example example);
	size_t sizeElement();
	void cutFeature(float k);
	void print(Logger *log_);
	
private:
	Parameter *param_;
	
	size_t sizeElement_;
	size_t sizeData_;
	bool ignoreState_;
	
	Point pack(vector<string>& tokens, bool isUpdate = true, bool hasLabel = true);
	vector<string> applyTemplate(vector<vector<string> > tokenList, size_t curPos, vector<string> templateList);	
};


}

#endif

