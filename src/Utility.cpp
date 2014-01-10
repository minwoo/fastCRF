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
#include <time.h>
#include <stdio.h>
#include "Utility.h"

namespace fastcrf {

using namespace std;

/** Tokenizer */
vector<string> Tokenize(const string& str, const string& delimiters) {
	vector<string> tokens;
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos  = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }

	return tokens;
} 

/** Logger */
Logger::Logger() {
	file_ = stderr;
	level_ = 1;
}

Logger::Logger(const string& filename, size_t level) {
	if (filename == "")
		file_ = stderr;
	else {
		if (!(file_ = fopen(filename.c_str(), "a+"))) 
			throw runtime_error("cannot open data file");
	}

	level_ = level;
}

Logger::~Logger() {
	if (file_)
		fclose(file_);
}

void Logger::setLevel(size_t level) {
	level_ = level;
}

string Logger::getTime() {
	time_t	unix_time;
	time(&unix_time);
	struct tm	*clock = localtime(&unix_time);

	char tmp_time[1024];
	sprintf(tmp_time, "%04d-%02d-%02d %02d:%02d:%02d", clock->tm_year+1900, clock->tm_mon+1, clock->tm_mday, clock->tm_hour, clock->tm_min, clock->tm_sec);

	return string(tmp_time);
}

int Logger::report(const char *fmt, ...) {
	int ret = 0;

	/// write current time
	if (level_ > 2) {
		fprintf(file_, "[%s] ", getTime().c_str());
		fflush(file_);
	}
	
	/// write the message
	if (level_ > 0) {
		va_list argptr;
		va_start(argptr, fmt);
		ret = vfprintf(file_, fmt, argptr);
		fflush(file_);

		/// standard out
		if (level_ > 1) {
			vfprintf(stderr, fmt, argptr);
			fflush(stderr);
		}
		va_end(argptr);
	}

	return ret;
}

int Logger::report(size_t level, const char *fmt, ...) {
	int ret = 0;

	/// write current time
	if (level > 2) {
		fprintf(file_, "[%s] ", getTime().c_str());
		fflush(file_);
	}
	
	/// write the message
	if (level > 0) {
		va_list argptr;
		va_start(argptr, fmt);
		ret = vfprintf(file_, fmt, argptr);
		fflush(file_);

		if (level > 1) {
			vfprintf(stderr, fmt, argptr);
			fflush(stderr);
		}
		va_end(argptr);
	}

	return ret;
}

/** Configurator */
Configurator::Configurator() {
}

Configurator::Configurator(const string& filename) {
	parse(filename);
}

bool Configurator::parse(const string& filename) {
	config_.clear();

	string line;
	ifstream f(filename.c_str());
	if (!f)
		throw runtime_error("cannot open data file");

	while (getline(f, line)) {
		if (!line.empty() && line[0] != '#') {
			vector<string> tokens = Tokenize(line, " =\t");
			if (tokens.size() < 2)
				throw runtime_error("invalid configuration file");
			vector<string> values;
			for (size_t i = 1; i < tokens.size(); i++) {

				if (tokens[i].find("[") != string::npos) {
					vector<string> tok = Tokenize(tokens[i], "[-]");
					if (tok.size() < 3)
						throw runtime_error("invalid configuration file");
					for (size_t i = atoi(tok[1].c_str()); i <= atoi(tok[2].c_str()); i++) {
						char temp[64];
						sprintf(temp, "%s%d", tok[0].c_str(), i);
						values.push_back(temp);
					}
				} else
					values.push_back(tokens[i]);
			}
			config_.insert(make_pair(tokens[0], values));
		}
	}
		
	filename_ = filename;
	return true;
}

string Configurator::getFilename() {
	return filename_;
}

bool Configurator::isValid(const string& key) {
	if (config_.find(key) != config_.end())
		return true;
	return false;
}

string Configurator::get(const string& key) {
	if (config_.find(key) == config_.end())
		return NULL;
	else
		return config_[key][0];
}

vector<string> Configurator::gets(const string& key) {
	vector<string> result;
	if (config_.find(key) != config_.end())
		result = config_[key];
	return result;
}


}
