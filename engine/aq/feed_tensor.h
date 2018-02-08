#pragma once


#include "board.h"
#include "../nn.h"


struct FeedTensor {
	
	lightmodel::zero_model::feature feature;
		int color; //turn of white -> 0, black -> 1.
	
		FeedTensor();
		void Clear();
		void Set(Board& b, int nv);
};

