
#include "variable.hpp"
#include "solver.hpp"
#include <nbla/computation_graph/variable.hpp>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/function/constant.hpp>
#include <nbla/function/rand.hpp>
#include <nbla/function/randn.hpp>
#include <iostream>

namespace nblapp {

using std::make_shared;

using nbla::Shape_t;
using nbla::CgVariable;
using nbla::NdArray;
using nbla::CgFunction;


extern nbla::Context current_ctx;

Variable::Variable()
{}

Variable::Variable(CgVariablePtr ptr)
: var_(ptr)
{}

CgVariablePtr Variable::ptr() const { return var_; }


Variable::Variable(const vector<int>& shape, const bool need_grad)
{
    Shape_t _shape(shape.size());
    std::copy(shape.begin(), shape.end(), _shape.begin());
    var_ = make_shared<CgVariable>(_shape, need_grad);
}

Variable::Variable(const Variable& other)
: var_(other.var_) {

}

Variable::Variable(Variable&& other)
: var_(std::move(other.var_)) {

}

Variable& Variable::operator =(const Variable& other) {
    var_ = other.var_;
    return *this;
}

Variable& Variable::operator =(Variable&& other) {
    var_ = std::move(other.var_);
    return *this;
}



Variable Variable::unlinked() const {
    auto clone = Variable{};
    clone.var_ = make_shared<CgVariable>(var_->variable()->view());
    return clone;
}

Variable::operator bool() const {
    return var_.operator bool();
}


int Variable::ndim() const {
    return var_->variable()->ndim();
}

int Variable::dim(int i) const {
    if (i < 0) i = ndim() + i;
    return var_->variable()->shape()[i];
}

int Variable::size() const {
    return var_->variable()->size(-1);
}

vector<int> Variable::shape() const {
    const auto& s = var_->variable()->shape();
    vector<int> out(s.size());
    std::copy(s.begin(), s.end(), out.begin());
    return out;
}


void Variable::set_need_grad(bool need) {
    var_->variable()->set_need_grad(need);
}

bool Variable::need_grad() const {
    return var_->variable()->need_grad();
}

void Variable::set_persistent(bool persist) {
    var_->set_persistent(persist);
}

void Variable::reshape(const vector<int>& shape) {
    vector<int64_t> _shape(shape.size());
    std::copy(shape.begin(), shape.end(), _shape.begin());
    var_->variable()->reshape(_shape, false);
}


Variable Variable::constant(const vector<int>& shape, float fill) {
    auto fn = make_shared<CgFunction>(create_Constant(current_ctx, fill, shape));
    auto out = Variable(shape, false);
    nbla::connect(fn, {}, {out.ptr()}, {}, true);
    return out.unlinked();
}

Variable Variable::uniform(const vector<int>& shape, float low, float high) {
    auto fn = make_shared<CgFunction>(create_Rand(current_ctx, low, high, shape, -1));
    auto out = Variable(shape, false);
    nbla::connect(fn, {}, {out.ptr()}, {}, true);
    return out.unlinked();
}

Variable Variable::normal(const vector<int>& shape, float mu, float sigma) {
    auto fn = make_shared<CgFunction>(create_Randn(current_ctx, mu, sigma, shape, -1));
    auto out = Variable(shape, false);
    nbla::connect(fn, {}, {out.ptr()}, {}, true);
    return out.unlinked();
}


template<>
float* Variable::data<float>() {
    nbla::Context ctx_cpu;
    return var_->variable()->cast_data_and_get_pointer<float>(ctx_cpu);
}

template<>
int* Variable::data<int>() {
    nbla::Context ctx_cpu;
    return var_->variable()->cast_data_and_get_pointer<int>(ctx_cpu);
}


template<>
const float* Variable::data<float>() const {
    nbla::Context ctx_cpu;
    return var_->variable()->get_data_pointer<float>(ctx_cpu);
}

template<>
const int* Variable::data<int>() const {
    nbla::Context ctx_cpu;
    return var_->variable()->get_data_pointer<int>(ctx_cpu);
}


void Variable::forward(bool clear_buffer, bool clear_no_need_grad) {
    var_->forward(clear_buffer, clear_no_need_grad);
}

void Variable::backward(bool clear_buffer) {
    if (!bwd_grad_) {
        bwd_grad_ = make_shared<NdArray>(var_->variable()->shape());
        bwd_grad_->fill(1);
    }
    var_->backward(bwd_grad_, clear_buffer);
}

void Variable::train_batch(BaseSolver& solver, float weight_decay, bool clear_buffer, bool clear_no_need_grad) {

    solver.zero_grad();
    forward(false, clear_no_need_grad);
    backward(clear_buffer);
    solver.update(weight_decay);
}


void Variable::fill(float v) {
    var_->variable()->data()->fill(v);
}

}

