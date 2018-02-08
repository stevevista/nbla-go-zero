#pragma once

#include "../tensor.h"

class ILeelaModel {
public:
	virtual ~ILeelaModel() {}
    virtual std::pair<std::vector<float>, float> predict(const InputFeature& features, float temperature) = 0;
};

