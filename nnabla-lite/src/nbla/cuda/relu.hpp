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

/** ReLU
 */
#ifndef __NBLA_FUNCTION_RELU_HPP__
#define __NBLA_FUNCTION_RELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;

NBLA_REGISTER_FUNCTION_HEADER(ReLU);

/** Rectified Linear Unit (ReLU) defined as
@f[
y_i = \max (0, x_i).
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param inplace The output array is will be shared with the input array if true.
\ingroup FunctionImplGrp
 */
template <typename T> class ReLU : public BaseFunction<> {

public:
  ReLU(const Context &ctx)
      : BaseFunction(ctx) {}
  virtual ~ReLU() {}
  virtual shared_ptr<Function> copy() const {
    return create_ReLU(ctx_);
  }
  virtual int min_inputs() { return 1; }
  virtual string name() { return "ReLU"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool inplace_data() const {
    return true;
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   Variable* output);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     Variable* output);
};
}
#endif
