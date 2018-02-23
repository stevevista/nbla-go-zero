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

import generator_common.common as common
import utils.type_conv


def generate(info, func_name, func_name_snakecase, template):

    arg_info = common.function_arguments(info)
    func_arg_variables = ', '.join(
        ['ctx'] + ['{}'.format(n) for n in arg_info['names']])
    func_args = ', '.join(['const Context &ctx'] + ['{} {}'.format(utils.type_conv.type_from_proto[
                          t]['cpp'], n) for t, n in zip(arg_info['types'], arg_info['names'])])

    io_info = common.function_io(info)
    template_defines = ', '.join(['typename {}'.format(t)
                                  for t in io_info['templates']])
    templates = ', '.join(io_info['templates'])

    return template.format(func_name=func_name,
                           func_name_snakecase=func_name_snakecase,
                           func_name_upcase=func_name.upper(),
                           templates=templates,
                           template_defines=template_defines,
                           func_args=func_args,
                           func_arg_variables=func_arg_variables)
