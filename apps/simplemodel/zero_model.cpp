#include "zero_model.hpp"
#include <cstring>
#include <cassert>
#include <unordered_map>
#include <list>
#include "model.h"


using namespace dlib;


class ZeroPredictModel::ModelImpl {
public:
    zero_model model;
};


ZeroPredictModel::ZeroPredictModel()
: impl_(new ModelImpl)
{}

bool ZeroPredictModel::load_weights(const std::string& path) {
    return impl_->model.load_weights(path);
}


std::pair<std::vector<float>, float> ZeroPredictModel::predict(const InputFeature& features, float temperature) {
    
    return impl_->model.predict(features, temperature);
}

vector<ZeroPredictModel::distribution_t> ZeroPredictModel::predict_distribution(const InputTensor& planes, float temprature) {

    return impl_->model.predict_policy(planes, temprature);
}

vector<float> ZeroPredictModel::predict_winrate(const InputTensor& planes) {

    return impl_->model.predict_value(planes);
}
