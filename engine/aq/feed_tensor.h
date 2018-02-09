#pragma once


#include "board.h"
#include <array>

struct FeedTensor {
	
	using Feature = std::array<std::array<float, 361>, 18>;
	Feature feature;
	int color; //turn of white -> 0, black -> 1.
	
	FeedTensor();
	void Clear();
	void Set(Board& b, int nv);
};

