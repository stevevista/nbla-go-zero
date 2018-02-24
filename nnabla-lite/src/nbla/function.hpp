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

/** Function interface class
 */
#ifndef __NBLA_FUNCTION_HPP__
#define __NBLA_FUNCTION_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <tuple>

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::tuple;
using std::get;

/** \defgroup NNablaCoreGrp Core components of NNabla */
/*@{*/

/// Variable%s as a vector of raw pointer.
typedef vector<Variable *> Variables;

/** An interface for the units of computation.
This is extended to implement a new Function which implements forward
computation (forward() function) of the function

@f[
{\mathbf y} = f({\mathbf x})
@f]

and backward computation (backward() function)

@f[
  \Delta {\mathbf x} += \Delta {\mathbf y} \cdot \nabla_{\mathbf x} {\mathbf y}
@f]

where @f$\Delta {\mathbf x}@f$ and @f$\Delta {\mathbf y}@f$ are backpropagation
error (gradient) of the input and output variable
propagated through backward computations of descendant of computation graph, and
@f$\nabla_{\mathbf x} {\mathbf y}@f$ is a
<a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian
matrix</a>
of the function. Note that propagated error is not substituted but
accumulated to the gradient of the input variable because we assume
@f${\mathbf x}@f$ can be used more than once by other functions. In the above
 example, the number of each of input and output variable is one, but Function
 can take multiple inputs and outputs.
*/
class NBLA_API Function {
  bool used_{false};

public:
  typedef shared_ptr<Function> Ptr;

protected:
  Context ctx_;                           ///< Storing context.

public:
  // Inplace level used in inplace_data function.
  static const int NOT_INPLACE = 0;
  static const int INPLACE_NOT_MODIFY = 1;
  static const int INPLACE = 2;

  /// Copying and storing Context.
  explicit Function(const Context &ctx);
  virtual ~Function() = 0;

public:
  void forward(const Variables &inputs, Variable* output);
  
public:
  /** Setting up function.

  This must be called before the Function instance is used, and will do:

  - Determine Array class used according to the return of
    Function::allowed_array_classes and  the given Context.
  - Type and shape check.
  - Calling Function::setup_impl.
  - Pre-allocate memory to prevent locking in asynchronous execution in
    CUDA etc.

  @param inputs vector of Variable*
  @param outputs vector of Variable*
  */
  void setup(const Variables &inputs, Variable* output);


  /** Get Context used in this function.
  */
  Context context() const;


  /** Get minimum number of inputs.

  This is meant to be used in setup function with in_types which is used to get
  maximum number of inputs.
  */
  virtual int min_inputs() = 0;

  /** Get function name in string
  */
  virtual string name() = 0;

  /** Get array classes that are allowed to be specified by Context
  */
  virtual vector<string> allowed_array_classes() = 0;

  /** Get in-place-level of i-th input variable's data (see below).

      * 0 (NOT_INPLACE): Not in-placed
      * 1 (INPLACE_NOT_MODIFY): In-placed but not modified.
      * 2 (INPLACE): In-placed and modified.

      @param[in] i Input variable index.
      @retval Returns 0 by default.
      @note If a subclass uses in-place computation, the function must overwride
     this function.
   */
  virtual bool inplace_data() const { return false; }

  /** Copy another instance of Function with the same context.
  */
  virtual shared_ptr<Function> copy() const = 0;

  /** Check whether this was already used, and turn it used.
   */
  inline bool ask_if_used_and_use() {
    bool r = used_;
    used_ = true;
    return r;
  }

protected:
  /** Implementation part of setup().

  It must do:

  - Reshape output Variable%s.
  - Allocate resources used in forward/backward computation if necessary.
  - Checking shapes and dtypes etc.
  @sa setup() for parameters
  */
  virtual void setup_impl(const Variables &inputs,
                          Variable* output) = 0;

  /** Implementation part of forward().

  It must do:

  - Take data in inputs and store results into data in outputs.

  @sa setup() arguments.
  */
  virtual void forward_impl(const Variables &inputs,
                            Variable* output) = 0;


  DISABLE_COPY_AND_ASSIGN(Function);
};

/** Base function.

    Keep arguments.
 */
template <typename... Args> class BaseFunction : public Function {
protected:
  tuple<typename std::remove_reference<Args>::type...> args_;

public:
  BaseFunction(const Context &ctx, Args... args)
      : Function(ctx), args_(args...) {}
};
/*@}*/

typedef Function::Ptr FunctionPtr;

/** \defgroup FunctionImplGrp Function list */
/*@{*/
/*@}*/
}
#endif
