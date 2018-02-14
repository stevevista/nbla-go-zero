#pragma once

#ifdef _WIN32
	#define COMPILER_MSVC
	#define NOMINMAX
#endif

#include <vector>
#include "feed_tensor.h"
#include <sstream>


namespace tensorflow {

struct Session {};

}

void ReadPolicyProto(const std::string& sl_path);
void ReadValueProto(const std::string& vl_path);

void PolicyNet(tensorflow::Session* sess,
		std::vector<FeedTensor>& ft_list,
		std::vector<std::array<double,EBVCNT>>& prob_list,
		double temp=0.67, int sym_idx=0);

void ValueNet(tensorflow::Session* sess, std::vector<FeedTensor>& ft_list,
		std::vector<float>& eval_list, int sym_idx=0);

extern int cfg_sym_idx;
