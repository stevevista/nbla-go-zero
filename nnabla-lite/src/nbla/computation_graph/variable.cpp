// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>

#include <memory>
#include <set>
#include <unordered_set>
// #include <iostream>  // TODO: remove

namespace nbla {

using std::make_shared;
using std::set;
using std::unordered_set;

CgVariable::CgVariable() {
  var_ = make_shared<Variable>(Shape_t{});
}

CgVariable::CgVariable(Shape_t shape) {
  var_ = make_shared<Variable>(shape);
}

CgVariable::CgVariable(VariablePtr var) { var_ = var; }

// Return if all variables are consumed.
inline bool remember_unseen_or_check_all_consumed_forward(
    unordered_set<CgVariablePtr> &vseen, CgVariablePtr v) {
  // Clear if no branch
  if (v->function_reference_count() < 2)
    return true;
  // Check if the variable is seen in this forward propagation
  auto it = vseen.find(v);
  if (it == vseen.end()) {
    // Initialize the variable consumption count (set one is consume).
    v->consume(true);
    vseen.insert(v);
    return false;
  }
  int c = v->consume();
  if (c == v->function_reference_count()) {
    vseen.erase(it); // For better search performance of another.
    return true;
  }
  return false;
}

void forward_recursive(CgFunctionPtr cg_f, unordered_set<CgFunctionPtr> &fseen,
                       unordered_set<CgVariablePtr> &vseen,
                       int function_reference_count, bool clear_buffer) {
  if (!cg_f)
    return;
  // Seen set should be added only if the function output is branching.
  if (function_reference_count > 1) {
    fseen.insert(cg_f);
  }
  // Recursively call predecessors' forward functions.
  for (auto cg_v : cg_f->inputs()) {
    auto parent_f = cg_v->parent();
    if (fseen.find(parent_f) != fseen.end())
      continue;
    forward_recursive(parent_f, fseen, vseen, cg_v->function_reference_count(),
                      clear_buffer);
  }
  // Execute forward.
  auto output_shared = cg_f->function_output_shared();
  // std::cout << "Call forward of " << cg_f->function()->name() << "."
  //           << std::endl;
  cg_f->function()->forward(cg_f->function_inputs(), output_shared.get());

  // Clear unnecessary variable buffers
  if (clear_buffer) {
    for (int i = 0; i < cg_f->num_inputs(); i++) {
      auto vi = cg_f->inputs()[i];
      if (vi->rank() == 0 || vi->persistent() ||
          (i==0 && cg_f->function()->inplace_data())) {
        // Root variable, or persistent flag is set by user.
        continue;
      }
      if (remember_unseen_or_check_all_consumed_forward(vseen, vi)) {
        vi->variable()->array()->clear();
      }
    }
  }
}

void CgVariable::forward(bool clear_buffer) {
  unordered_set<CgFunctionPtr> fseen; // Seen functions.
  unordered_set<CgVariablePtr> vseen; // Seen variables.
  forward_recursive(parent_, fseen, vseen, function_reference_count_,
                    clear_buffer);
}


}
