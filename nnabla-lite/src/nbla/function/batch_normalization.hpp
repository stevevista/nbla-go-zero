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

/** Batch Normalization
 */
#ifndef __NBLA_FUNCTION_BATCHNORM_HPP__
#define __NBLA_FUNCTION_BATCHNORM_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <vector>

using std::vector;

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BatchNormalization, const vector<int> &);


template <typename T>
class BatchNormalization
    : public BaseFunction<const vector<int> &> {
protected:
  vector<int> axes_;
  float decay_rate_;
  float eps_;
  bool batch_stat_;
  int size0_, size1_, size2_, size02_, size12_;

public:
  BatchNormalization(const Context &ctx, const vector<int> axes)
      : BaseFunction(ctx, axes), axes_(axes) {}
  virtual ~BatchNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_BatchNormalization(ctx_, axes_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual int min_inputs() { return 5; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BatchNormalization"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const {
    // Gradient computation always requires output mean and var.
    return o > 0;
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void forward_impl_global(const Variables &inputs,
                                            const Variables &outputs);
};
}
#endif
