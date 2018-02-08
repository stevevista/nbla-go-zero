#pragma once
#include <array>
#include <vector>
#include <iostream>
#include <bitset>


using BoardPlane = std::bitset<361>;
using InputFeature = std::array<BoardPlane, 18>;
using InputTensor = std::vector<InputFeature>;

class IPreditModel {
public:
	virtual ~IPreditModel() {}
	virtual std::vector<std::vector<float>> predict_distribution(const InputTensor& planes, float temprature) = 0;
	virtual std::vector<float> predict_winrate(const InputTensor& planes) = 0;	
};

