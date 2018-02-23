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

/** Softmax
 */
#ifndef __NBLA_FUNCTION_SOFTMAX_HPP__
#define __NBLA_FUNCTION_SOFTMAX_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::make_shared;

NBLA_REGISTER_FUNCTION_HEADER(Softmax, int);

/** Softmax normalization defined as
@f[
y_i = \frac{\exp(x_i)}{\sum_j exp(x_j)}
@f]
along dimension specified by axis.

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.
@param axis Axis normalization is taken.
\ingroup FunctionImplGrp
 */
template <typename T> class Softmax : public BaseFunction<int> {
protected:
  int axis_;
  int size0_, size1_, size2_;

public:
  Softmax(const Context &ctx, int axis)
      : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~Softmax() {}
  virtual shared_ptr<Function> copy() const {
    return create_Softmax(ctx_, axis_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Softmax"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return i == 0; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
};
}
#endif
