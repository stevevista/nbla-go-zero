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

/** Mul Scalar
 */
#ifndef __NBLA_FUNCTION_MUL_SCALAR_HPP__
#define __NBLA_FUNCTION_MUL_SCALAR_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>
#include <cmath>
#include <iostream>
namespace nbla {


NBLA_REGISTER_FUNCTION_HEADER(MulScalar); 

template <typename T>
class MulScalar : public BaseFunction<> {                     \
    public:                                                                        \
  virtual ~MulScalar() {}                                                           \
  virtual bool inplace_data() const {
    return true;
  }
  virtual string name() { return "MulScalar"; } 
  virtual int min_inputs() { return 2; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual void setup_impl(const Variables &inputs, Variable* output) {
    output->reshape(inputs[0]->shape(), true);
    output->set_array(inputs[0]->array());
  }
    MulScalar(const Context &ctx) : BaseFunction<>(ctx) {}      \
    virtual shared_ptr<Function> copy() const {                                \
      return create_MulScalar(this->ctx_);                                        \
    }
    void forward_impl(
      const Variables &inputs, Variable* output) {

    const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
    const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
    T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
    for (int idx = 0; idx < output->size(); ++idx) {
      y[idx] = x0[idx] * x1[0];
    }
  }
};


}
#endif
