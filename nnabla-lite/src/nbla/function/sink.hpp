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

/** Sink
 */
#ifndef __NBLA_FUNCTION_SINK_HPP__
#define __NBLA_FUNCTION_SINK_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sink);

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename T> class Sink : public Function {
public:
  Sink(const Context &ctx)
      : Function(ctx) {}
  virtual ~Sink() {}
  virtual shared_ptr<Function> copy() const {
    return create_Sink(ctx_);
  }
  virtual int min_inputs() { return 1; }
  virtual string name() { return "Sink"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool inplace_data() const {
    return true;
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   Variable *output);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     Variable* output);
};
}
#endif
