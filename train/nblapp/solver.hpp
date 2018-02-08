#pragma once

#include "variable.hpp"
#include <memory>
#include <vector>

namespace nbla {

class Solver;
typedef std::shared_ptr<Solver> SolverPtr;

}

namespace nblapp {

using std::vector;
using std::pair;
using std::string;
using nbla::SolverPtr;

class NPP_API BaseSolver {
public:
    void set_parameters(const vector<pair<string, Variable>>& params);
    void set_parameters(const string& scope);
    void zero_grad();
    void update(float weight_decay = 0);
    void set_learning_rate(float lr);

protected:
    SolverPtr ptr_;
};

class NPP_API AdamSolver : public BaseSolver {
public:
    AdamSolver(float alpha=0.001, float beta1=0.9, float beta2=0.999, float eps=1e-05);
};

class NPP_API SgdSolver : public BaseSolver {
public:
    SgdSolver(float learning_rate);
};

class NPP_API MomentumSolver : public BaseSolver {
public:
    MomentumSolver(float learning_rate, float moment = 0.9);
};



}

