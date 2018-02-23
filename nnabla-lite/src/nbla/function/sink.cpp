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
#include <nbla/array.hpp>
#include <nbla/function/sink.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Sink, bool);

template <typename T>
void Sink<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  outputs[0]->reshape({}, true);
}

template <typename T>
void Sink<T>::forward_impl(const Variables &inputs, const Variables &outputs) {}

// Template instanciation
template class Sink<float>;
} // namespace nbla
