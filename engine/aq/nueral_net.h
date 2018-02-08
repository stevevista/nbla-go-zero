#pragma once

#ifdef _WIN32
	#define COMPILER_MSVC
	#define NOMINMAX
#endif

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "feed_tensor.h"

#ifdef _WIN32
#include <windows.h>
#endif


// fake stubs
namespace tensorflow {
	struct Session {
		int _;
	};
}

void PolicyNet(
		tensorflow::Session* sess,
		std::vector<FeedTensor>& ft_list,
		std::vector<std::array<double,EBVCNT>>& prob_list,
		double temp=0.67,
		int sym_idx = 0);

void ValueNet(
		tensorflow::Session* sess,
		std::vector<FeedTensor>& ft_list,
		std::vector<float>& eval_list,
		int sym_idx = 0);


void register_aq_predict(IPreditModel* m);

extern int cfg_sym_idx;

