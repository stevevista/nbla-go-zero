# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six.moves import range

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('BatchNormalization')


def ref_batch_normalization(x, beta, gamma, rmean, rvar, axes, decay_rate,
                            eps, batch_stat, output_stat):
    assert len(axes) == 1
    reduc_axes = list(range(x.ndim))
    del reduc_axes[axes[0]]
    reduc_axes = tuple(reduc_axes)
    m = x.size / x.shape[axes[0]]
    if batch_stat:
        mean = x.mean(reduc_axes, keepdims=True)
        var = x.var(reduc_axes, keepdims=True)
        rmean[...] = decay_rate * rmean + (1 - decay_rate) * mean
        rvar[...] = decay_rate * rvar + (1 - decay_rate) * var * m / (m - 1)
    else:
        mean = rmean
        var = rvar
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = x_hat * gamma + beta
    if output_stat:
        return y, mean, var
    return y


def create_inputs(rng, axis):
    x = rng.randn(2, 3, 4).astype(np.float32) * 2
    shape_stat = [1 for _ in range(x.ndim)]
    shape_stat[axis] = x.shape[axis]
    beta = rng.randn(*shape_stat).astype(np.float32)
    gamma = rng.randn(*shape_stat).astype(np.float32)
    rmean = np.zeros(shape_stat, dtype=np.float32)
    rvar = np.zeros(shape_stat, dtype=np.float32)
    return x, beta, gamma, rmean, rvar


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("output_stat", [True, False])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_batch_normalization_forward_backward(seed, axis, decay_rate, eps,
                                              output_stat, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis))
    axes = [axis]
    batch_stat = True
    function_tester(rng, F.batch_normalization, ref_batch_normalization,
                    inputs,
                    func_args=[axes, decay_rate, eps, batch_stat, output_stat],
                    backward=[True, True, True, False, False],
                    ctx=ctx, func_name=func_name, dstep=1e-2, atol_b=1e-2)

    # Check if running mean and var works.
    vinputs = []
    for i in inputs:
        vinputs.append(nn.Variable(i.shape, True))
        vinputs[-1].d = i
    for i in range(5):
        inputs[0] = rng.randn(*inputs[0].shape)
        vinputs[0].d[...] = inputs[0]
        ref_y = ref_batch_normalization(
            *(inputs + [axes, decay_rate, eps, batch_stat, output_stat]))
        with nn.context_scope(ctx), nn.auto_forward():
            y = F.batch_normalization(
                *(vinputs + [axes, decay_rate, eps, batch_stat, output_stat]))
        assert np.allclose(vinputs[3].d, inputs[3])
        assert np.allclose(vinputs[4].d, inputs[4], atol=1e-3)

    # Check if global stat mode works
    batch_stat = False
    if output_stat:
        return
    ref_y = ref_batch_normalization(
        *(inputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    with nn.context_scope(ctx), nn.auto_forward():
        y = F.batch_normalization(
            *(vinputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    assert np.allclose(ref_y, y.d, atol=1e-6)
