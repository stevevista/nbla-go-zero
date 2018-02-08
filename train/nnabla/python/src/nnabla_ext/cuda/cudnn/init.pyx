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

from nnabla.logger import logger
from nnabla import add_available_context

cdef extern from "nbla/cuda/cudnn/init.hpp" namespace "nbla":
    void init_cudnn() except+


logger.info('Initializing cuDNN extension...')
try:
    init_cudnn()
    add_available_context('cuda', 'cudnn')
except:
    logger.warning(
        'Cudnn initialization failed. Please make sure that the Cudnn is correctly installed.')
