#include "solver.hpp"
#include "parameter.hpp"
#include <nbla/solver/adam.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/solver/sgd.hpp>
#include <nbla/solver/momentum.hpp>
#include <nbla/solver/nesterov.hpp>

namespace nblapp {
 
extern nbla::Context current_ctx;

void BaseSolver::set_parameters(const vector<pair<string, Variable>>& params) {

    if (ptr_) {
        vector<pair<string, nbla::VariablePtr>> _params;
        for (const auto& kv : params) 
            _params.push_back({kv.first, kv.second.ptr()->variable()});

        ptr_->set_parameters(_params);
    }
}

void BaseSolver::set_parameters(const string& scope) {
    ParameterScope _(scope);
    set_parameters(ParameterScope::get_parameters());
}

void BaseSolver::zero_grad() {
    ptr_->zero_grad();
}

void BaseSolver::update(float weight_decay) {
    if (weight_decay != 0)
        ptr_->weight_decay(weight_decay);
    ptr_->update();
}

void BaseSolver::set_learning_rate(float lr) {
    ptr_->set_learning_rate(lr);
}


AdamSolver::AdamSolver(float alpha, float beta1, float beta2, float eps) {
    ptr_ = nbla::create_AdamSolver(current_ctx, alpha, beta1, beta2, eps);
}

SgdSolver::SgdSolver(float learning_rate)
{
    ptr_ = nbla::create_SgdSolver(current_ctx, learning_rate);
}

MomentumSolver::MomentumSolver(float learning_rate, float moment)
{
    ptr_ = nbla::create_MomentumSolver(current_ctx, learning_rate, moment);
}

NesterovSolver::NesterovSolver(float learning_rate, float moment)
{
    ptr_ = nbla::create_NesterovSolver(current_ctx, learning_rate, moment);
}


}
