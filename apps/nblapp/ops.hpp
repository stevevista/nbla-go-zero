#pragma once

#include <nblapp/variable.hpp>

namespace nblapp {

using std::string;

enum PAD_TYPE {
    PAD_SAME,
    PAD_VALID
};

namespace nn {

NPP_API Variable sink(const vector<Variable>& vars, bool one_input_grad);

NPP_API Variable add(const Variable& x, double scalar);
NPP_API Variable add(const Variable& a, const Variable& b, bool inplace=true);
NPP_API Variable sub(double scalar, const Variable& x);
NPP_API Variable sub(const Variable& a, const Variable& b);
NPP_API Variable mul(const Variable& a, const Variable& b);
NPP_API Variable mul(const Variable& x, double scalar);
NPP_API Variable div(const Variable& a, const Variable& b);
NPP_API Variable div(const Variable& x, double scalar);
NPP_API Variable div(double scalar, const Variable& x);
NPP_API Variable log(const Variable& x);

NPP_API Variable relu(const Variable& x, const bool inplace=true);
NPP_API Variable tanh(const Variable& x);
NPP_API Variable sigmoid(const Variable& x);
NPP_API Variable softmax(const Variable& x, int axis=-1);

NPP_API Variable avgpool(const Variable& x, const std::vector<int>& kernel, const std::vector<int>& stride);
NPP_API Variable maxpool(const Variable& x, const std::vector<int>& kernel, const std::vector<int>& stride);

NPP_API Variable reduce_mean(const Variable& x, vector<int> axes={}, bool keep_dim=false);

NPP_API Variable cross_entropy(const Variable& x, const Variable& y, vector<int> axes={});
NPP_API Variable softmax_cross_entropy(const Variable& x, const Variable& y, int axis=-1);
NPP_API Variable sigmoid_cross_entropy(const Variable& x, const Variable& y);
NPP_API Variable squared_error(const Variable& x, const Variable& y);


NPP_API Variable fully_connected(const string& name, const Variable& x, 
        const int outputs, 
        bool has_bias = true, int axis = 1);

NPP_API Variable conv2d(const string& name, const Variable& x, 
        const int filters, const std::vector<int>& kernel, const std::vector<int>& stride, const PAD_TYPE pad_type = PAD_SAME, 
        bool has_bias = false,
        int axis = 1);

NPP_API Variable batchnorm(const string& name, const Variable& x, bool batch_stat,  
                int axis=1,
                float decay_rate=0.9, float eps=1e-05);


}





template <typename T>
inline Variable operator+(const Variable& a, T b) {
    return nn::add(a, (float)b);
}
template <typename T>
inline Variable operator+(T b, const Variable& a) {
    return nn::add(a, (float)b);
}
inline Variable operator+(const Variable& a, const Variable& b) {
    return nn::add(a, b, false);
}
template <typename T>
inline Variable operator-(const Variable& a, T b) {
    return nn::add(a, -((float)b));
}
template <typename T>
inline Variable operator-(T a, const Variable& b) {
    return nn::sub((float)a, b);
}
inline Variable operator-(const Variable& a, const Variable& b) {
    return nn::sub(a, b);
}
template <typename T>
inline Variable operator*(const Variable& a, T b) {
    return nn::mul(a, (float)b);
}
template <typename T>
inline Variable operator*(T b, const Variable& a) {
    return nn::mul(a, (float)b);
}
inline Variable operator*(const Variable& a, const Variable& b) {
    return nn::mul(a, b);
}
template <typename T>
inline Variable operator/(const Variable& a, T b) {
    return nn::mul(a, 1.0f/(float)b);
}
template <typename T>
inline Variable operator/(T a, const Variable& b) {
    return nn::div((float)a, b);
}
inline Variable operator/(const Variable& a, const Variable& b) {
    return nn::div(a, b);
}

}
