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

/** Tanh
 */
#ifndef __NBLA_FUNCTION_TANH_HPP__
#define __NBLA_FUNCTION_TANH_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Tanh);

template <typename T> class Tanh : public Function { 
public:        
  virtual ~Tanh() {}                                
  virtual string name() { return "Tanh"; }              
    Tanh(const Context &ctx) : Function(ctx) {}      
    virtual shared_ptr<Function> copy() const {                
      return create_Tanh(this->ctx_);           
    }   
    virtual int min_inputs() { return 1; }
    virtual vector<string> allowed_array_classes() {
      return SingletonManager::get<Cpu>()->array_classes();
    }
    virtual void setup_impl(const Variables &inputs, Variable* output) {
      output->reshape(inputs[0]->shape(), true);
      output->set_array(inputs[0]->array());
    }

    virtual void forward_impl(const Variables &inputs, Variable* output) {
      const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
      T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
      for (int idx = 0; idx < inputs[0]->size(); ++idx) {
        y[idx] = std::tanh(x[idx]);
      }
    } 
};


}
#endif
