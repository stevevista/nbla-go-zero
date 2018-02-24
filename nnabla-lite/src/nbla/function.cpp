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

#include <nbla/function.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

using std::make_shared;

Function::Function(const Context &ctx) : ctx_(ctx) {}

Function::~Function() {}

void Function::setup(const Variables &inputs, Variable* output) {

  // Check if specifiedd array_class by context matches to allowed array
  // classes.
  int array_class_index =
      0; // Default array is 0-th array_class in allowed_array_classes().
  for (int i = 0; i < this->allowed_array_classes().size(); ++i) {
    if (ctx_.array_class == this->allowed_array_classes()[i]) {
      array_class_index = i;
    }
  }
  ctx_.set_array_class(this->allowed_array_classes()[array_class_index]);

  // Check number of inputs and output
  NBLA_CHECK(this->min_inputs() <= inputs.size(), error_code::value,
             "%s needs at least %d inputs (given %d). ", this->name().c_str(),
             this->min_inputs(), inputs.size());

  // Call setup implemention
  this->setup_impl(inputs, output);
}


void Function::forward(const Variables &inputs, Variable* output) {

  this->forward_impl(inputs, output);
}

Context Function::context() const { return ctx_; }
}
