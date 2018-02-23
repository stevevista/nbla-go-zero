
#include "ops.hpp"
#include "parameter.hpp"
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/function/rand.hpp>
#include <nbla/function/randn.hpp>
#include <nbla/function/sink.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/add_scalar.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/function/r_sub_scalar.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/function/div2.hpp>
#include <nbla/function/r_div_scalar.hpp>
#include <nbla/function/log.hpp>
#include <nbla/function/relu.hpp>
#include <nbla/function/leaky_relu.hpp>
#include <nbla/function/sigmoid.hpp>
#include <nbla/function/tanh.hpp>
#include <nbla/function/softmax.hpp>
#include <nbla/function/average_pooling.hpp>
#include <nbla/function/max_pooling.hpp>
#include <nbla/function/mean.hpp>
#include <nbla/function/softmax_cross_entropy.hpp>
#include <nbla/function/sigmoid_cross_entropy.hpp>
#include <nbla/function/squared_error.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/identity.hpp>
#include <cassert>


namespace nblapp {

using std::make_shared;
using nbla::CgFunction;
using nbla::connect;


extern nbla::Context current_ctx;


static std::string join(const string& name, const string& sub) {
    return name.empty() ? sub : name + "/" + sub;
}



namespace nn {

Variable sink(const vector<Variable>& vars, bool one_input_grad) {
    auto fn = make_shared<CgFunction>(create_Sink(current_ctx, one_input_grad));
    vector<CgVariablePtr> _vars;
    for (auto v : vars) _vars.push_back(v.ptr());
    return connect(fn, _vars)[0];
}

Variable identity(const Variable& x) {
    auto fn = make_shared<CgFunction>(create_Identity(current_ctx));
    return connect(fn, {x.ptr()})[0];
}

Variable add(const Variable& x, double scalar) {
    auto fn = make_shared<CgFunction>(create_AddScalar(current_ctx, scalar));
    return connect(fn, {x.ptr()})[0];
}

Variable add(const Variable& a, const Variable& b, bool inplace) {
    auto fn = make_shared<CgFunction>(create_Add2(current_ctx, inplace));
    return connect(fn, {a.ptr(), b.ptr()})[0];
}

Variable sub(double scalar, const Variable& x) {
    auto fn = make_shared<CgFunction>(create_RSubScalar(current_ctx, scalar));
    return connect(fn, {x.ptr()})[0];
}

Variable sub(const Variable& a, const Variable& b) {
    auto fn = make_shared<CgFunction>(create_Sub2(current_ctx));
    return connect(fn, {a.ptr(), b.ptr()})[0];
}


Variable mul(const Variable& x, double scalar) {
    auto fn = make_shared<CgFunction>(create_MulScalar(current_ctx, scalar));
    return connect(fn, {x.ptr()})[0];
}

Variable mul(const Variable& a, const Variable& b) {
    auto fn = make_shared<CgFunction>(create_Mul2(current_ctx));
    return connect(fn, {a.ptr(), b.ptr()})[0];
}

Variable div(const Variable& a, const Variable& b) {
    auto fn = make_shared<CgFunction>(create_Div2(current_ctx));
    return connect(fn, {a.ptr(), b.ptr()})[0];
}

Variable div(double scalar, const Variable& x) {
    auto fn = make_shared<CgFunction>(create_RDivScalar(current_ctx, scalar));
    return connect(fn, {x.ptr()})[0];
}

Variable log(const Variable& x) {
    auto fn = make_shared<CgFunction>(create_Log(current_ctx));
    return connect(fn, {x.ptr()})[0];
}

Variable relu(const Variable& x, const bool inplace) {
        auto fn = make_shared<CgFunction>(create_ReLU(current_ctx, inplace));
        return connect(fn, {x.ptr()})[0];
}

Variable tanh(const Variable& x) {
    auto fn = make_shared<CgFunction>(create_Tanh(current_ctx));
    return connect(fn, {x.ptr()})[0];
}

Variable sigmoid(const Variable& x) {
    auto fn = make_shared<CgFunction>(create_Sigmoid(current_ctx));
    return connect(fn, {x.ptr()})[0];
}

Variable softmax(const Variable& x, int axis) {
    
    if (axis == -1) {
        axis = x.ndim() - 1;
    }
    auto fn = make_shared<CgFunction>(create_Softmax(current_ctx, axis));
    return connect(fn, {x.ptr()})[0];
}


Variable reduce_mean(const Variable& x, vector<int> axes, bool keep_dim) {
    
    if(axes.empty()) {
        for (int axis=0; axis < x.ndim(); axis++)
            axes.push_back(axis);
    }
    
    auto fn = make_shared<CgFunction>(create_Mean(current_ctx, axes, keep_dim)); 
    return connect(fn, {x.ptr()})[0];
}

Variable avgpool(const Variable& x, const std::vector<int>& kernel, const std::vector<int>& stride) {
    auto fn = make_shared<CgFunction>(create_AveragePooling(current_ctx, kernel, stride, true, {0,0}, true)); 
    return connect(fn, {x.ptr()})[0];
}

Variable maxpool(const Variable& x, const std::vector<int>& kernel, const std::vector<int>& stride) {
    auto fn = make_shared<CgFunction>(create_MaxPooling(current_ctx, kernel, stride, true, {0,0})); 
    return connect(fn, {x.ptr()})[0];
}


Variable reduce_sum(const Variable& x, vector<int> axes, bool keep_dim) {
    
    if(axes.empty()) {
        for (int axis=0; axis < x.ndim(); axis++)
            axes.push_back(axis);
    }
    
    auto fn = make_shared<CgFunction>(create_Sum(current_ctx, axes, keep_dim)); 
    return connect(fn, {x.ptr()})[0];
}

Variable cross_entropy(const Variable& x, const Variable& y, vector<int> axes) {
    if (axes.empty())
        axes.push_back(x.ndim()-1);
    return -1 * reduce_sum(y * log(x), axes, false);
}

Variable softmax_cross_entropy(const Variable& x, const Variable& y, int axis) {
    if (x.shape() == y.shape()) {
        if (axis == -1) {
            axis = x.ndim() - 1;
        }
        return cross_entropy(softmax(x, axis), y, {axis});
    }
    auto fn = make_shared<CgFunction>(create_SoftmaxCrossEntropy(current_ctx, 1));
    return connect(fn, {x.ptr(), y.ptr()})[0];
}

Variable sigmoid_cross_entropy(const Variable& x, const Variable& y) {
    auto fn = make_shared<CgFunction>(create_SigmoidCrossEntropy(current_ctx));
    return connect(fn, {x.ptr(), y.ptr()})[0];
}

Variable squared_error(const Variable& x, const Variable& y) {
    auto fn = make_shared<CgFunction>(create_SquaredError(current_ctx));
    return connect(fn, {x.ptr(), y.ptr()})[0];
}

Variable fully_connected(const string& name, const Variable& x, 
        const int outputs, bool has_bias, int axis) {

    const auto in_shape = x.shape();
    assert(in_shape.size() > axis);
    int inputs = 1;
    for (int i=axis; i<in_shape.size(); ++i)
        inputs *= in_shape[i];

    auto fn = make_shared<CgFunction>(create_Affine(current_ctx, axis));

    auto d = std::sqrt(6. / (inputs + outputs));
    auto W = ParameterScope::get_or_create_uniform(join(name, "fc/W"), {inputs, outputs}, -d, d);
    
    vector<CgVariablePtr> params = {x.ptr(), W.ptr()};
    if (has_bias) {
        auto b = ParameterScope::get_or_create_constant(join(name, "fc/b"), {outputs}, 0);
        params.push_back(b.ptr());
    }

    return connect(fn, params)[0];
}


Variable conv2d(const string& name, const Variable& x, 
        const int filters, const std::vector<int>& kernel, const std::vector<int>& stride, const PAD_TYPE pad_type, 
        bool has_bias,
        int axis) {

    const int in_k = x.shape()[axis];

    std::vector<int> pad(2, 0);
    if (pad_type == PAD_SAME) {
        int pad_h = kernel[0]/2;
        int pad_w = kernel[1]/2;
        pad = {pad_h, pad_w};
    }

    auto fn = make_shared<CgFunction>(create_Convolution(current_ctx, axis, pad, stride, /*dilation*/{1, 1}, /*group*/1));

    auto d = std::sqrt(6. / ((kernel[0]*kernel[1]) * in_k + filters));
    auto W = ParameterScope::get_or_create_uniform(join(name, "conv/W"), {filters, in_k, kernel[0], kernel[1]}, -d, d);

    vector<CgVariablePtr> params = {x.ptr(), W.ptr()};
    if (has_bias) {
        auto b = ParameterScope::get_or_create_constant(join(name, "conv/b"), {filters}, 0);
        params.push_back(b.ptr());
    }

    return connect(fn, params)[0];
}

Variable batchnorm(const string& name, const Variable& x, bool batch_stat,  
    int axis,
    float decay_rate, float eps) {

    auto shape = x.shape();
    for (auto i=0; i<shape.size(); i++) {
        if (i != axis) shape[i] = 1;
    }
    auto fn = make_shared<CgFunction>(create_BatchNormalization(current_ctx, {axis}, decay_rate, eps, batch_stat));
    auto b = ParameterScope::get_or_create_constant(join(name, "/bn/b"), shape, 0);
    auto g = ParameterScope::get_or_create_constant(join(name, "/bn/g"), shape, 1);
    auto m = ParameterScope::get_or_create_constant(join(name, "/bn/m"), shape, 0);
    auto v = ParameterScope::get_or_create_constant(join(name, "/bn/v"), shape, 0);
    return connect(fn, {x.ptr(), b.ptr(), g.ptr(), m.ptr(), v.ptr()})[0];
}


} // namespace nn


}

