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

#include <nbla/computation_graph/computation_graph.hpp>

#include <memory>

namespace nbla {


using std::make_shared;

// Just a helper function.
static inline const char *b2str(bool b) { return b ? "true" : "false"; }


static void connect_core(CgFunctionPtr cg_f,
                                   const vector<CgVariablePtr> &inputs,
                                   CgVariablePtr output) {


  vector<Variable *> finputs(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    finputs[i] = inputs[i]->variable().get();
  }

  // Setup function.
  auto f = cg_f->function();
  f->setup(finputs, output->variable().get());

  // Check if in-place is properly used.
  if (!inputs[0]->allow_inplace_data()) {
    const bool inplace_level = f->inplace_data();
    if (inplace_level) {
      NBLA_CHECK(inplace_level < Function::INPLACE, error_code::value,
               "In-place input data of '%s' (depth=%d) is "
               "prohibited by the parent function '%s'.",
               f->name().c_str(), cg_f->rank(),
               inputs[0]->parent()->function()->name().c_str());
      // Since the in-placed input's data is not modified in this function,
      // the allow-inplace flag is propagated to the output variable.
      output->set_allow_inplace_data(false);
    }
  }

  // Check if branching doesn't appear in in-placed variables.
    bool inplace = f->inplace_data();
    NBLA_CHECK(
        (!inplace) || inputs[0]->function_reference_count() < 2,
        error_code::value,
        "Branching a variable is prohibited if it is in-placed. 0-th input "
        "of `%s` (depth=%d) is inplaced (data: %s).",
        f->name().c_str(), cg_f->rank(), b2str(f->inplace_data()));
  
}

CgVariablePtr connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs) {
  cg_f->set_inputs(inputs);
  auto output = make_shared<CgVariable>();
  output->set_parent(cg_f);
  // Weak references are held inside.
  cg_f->set_output(output);
  connect_core(cg_f, inputs, output);
  return output;
}



}
