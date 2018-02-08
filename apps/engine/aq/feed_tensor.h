#pragma once


#include "board.h"
#include "../tensor.h"


struct FeedTensor {
	
		InputFeature feature;
		int color; //turn of white -> 0, black -> 1.
	
		FeedTensor();
		void Clear();
		void Set(Board& b, int nv);
};

