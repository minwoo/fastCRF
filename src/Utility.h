/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "fastCRF" distribution.
 * http://github.com/minwoo/fastCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __UTIL_H__
#define __UTIL_H__

#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <stdarg.h>

namespace fastcrf {

using namespace std;

#define HEADER "============================================================\n  fastCRF - A fast tool of CRF\n============================================================\n"

/// tokenizer
vector<string> Tokenize(const string& str, const string& delimiters = " \t");

/** Logger */
class Logger {
public:
	Logger();
	Logger(const string& filename, size_t level = 1);
	~Logger();

	void setLevel(size_t level);
	int report(size_t level, const char *fmt, ...);
	int report(const char *fmt, ...);

private:
	size_t level_;
	FILE *file_;
	string getTime();
};

/** Configurator */
class Configurator {
public:
	Configurator();
	Configurator(const string& filename);
	bool parse(const string& filename);
	string getFilename();
	bool isValid(const string& key);
	string get(const string& key);
	vector<string> gets(const string& key);

private:
	string filename_;
	map<string, vector<string> > config_;
};

//  boost timer  -------------------------------------------------------------------//
//  Copyright Beman Dawes 1994-99.
//  See accompanying license for terms and conditions of use.
//  See http://www.boost.org/libs/timer for documentation.
class timer {
 public:
	timer() { _start_time = clock(); } 
	void   restart() { _start_time = clock(); } 
	double elapsed() const { return  double(clock() - _start_time) / CLOCKS_PER_SEC; }
	double elapsed_max() const { return (double(numeric_limits<clock_t>::max())	- double(_start_time)) / double(CLOCKS_PER_SEC); 	}
	double elapsed_min() const  { return double(1)/double(CLOCKS_PER_SEC); }
private:
	clock_t _start_time;
}; // timer

/** finite testing function */
#if defined(_MSC_VER) || defined(__BORLANDC__)
inline int finite(double x) { return _finite(x); }
#endif

/** log zero */
const double LOG_ZERO = log(DBL_MIN);

}

#endif
