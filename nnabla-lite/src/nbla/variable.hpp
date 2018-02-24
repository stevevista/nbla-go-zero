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

#ifndef __NBLA_VARIABLE_HPP__
#define __NBLA_VARIABLE_HPP__

#include <nbla/common.hpp>
#include <nbla/nd_array.hpp>

#include <memory>

using std::shared_ptr;

namespace nbla {

/** User interface for Array and passed to Function.

Users will create arrays via Variable and pass them to
Function. Variable has two array region internally, data and grad.
Data region is used as an input and/or output of Function::forward(), while
grad region is used for storing backprop error of Function::backward().

\ingroup NNablaCoreGrp
*/
class Variable : public NdArray {

public:
  typedef shared_ptr<Variable> Ptr;

  /**
  Constructor.

  @param shape Shape.
  */
  NBLA_API Variable(const Shape_t &shape = {});

  /**
  A shortcut function to cast data and get pointer.

  @sa SyncedArray::cast() and Array::pointer().
  */
  template <typename T> T *cast_data_and_get_pointer(const Context &ctx) {
    Array *arr = array()->cast(get_dtype<T>(), ctx);
    return arr->pointer<T>();
  }

  /**
  A shortcut function to get data pointer.

  @sa SyncedArray::get() and Array::const_pointer().
  */
  template <typename T> const T *get_data_pointer(const Context &ctx) {
    const Array *arr = array()->get(get_dtype<T>(), ctx);
    return arr->const_pointer<T>();
  }

  DISABLE_COPY_AND_ASSIGN(Variable);
};

///< Shared pointer of Variable.
typedef Variable::Ptr VariablePtr;
}
#endif
