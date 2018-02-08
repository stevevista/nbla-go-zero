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

from shutil import rmtree
import abc
import atexit
import h5py
import numpy
import os
import six
import tempfile

from nnabla.config import nnabla_config
from nnabla.logger import logger


class DataSource(object):
    '''
    Detailed documentation is available in :ref:`data_source_design`.
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _get_data(self, position):
        pass

    def __init__(self, shuffle=False, rng=None):
        '''
        Init method for DataSource
        '''
        logger.info('DataSource with shuffle({})'.format(shuffle))
        self._rng = rng
        if rng is None:
            self._rng = numpy.random.RandomState(313)
        self._variables = None
        self._generation = -1
        self._shuffle = shuffle
        self._position = 0
        self._size = 0
        self._closed = False
        atexit.register(self.close)

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            self._closed = True

    @property
    def variables(self):
        return self._variables

    def next(self):
        data = self._get_data(self._position)
        self._position += 1
        return data

    @property
    def position(self):
        return self._position

    @property
    def size(self):
        return self._size

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value

    @abc.abstractmethod
    def reset(self):
        self._position = 0


class DataSourceWithFileCacheError(Exception):
    pass


class DataSourceWithFileCache(DataSource):
    '''
    Detailed documentation is available in :ref:`data_source_with_file_cache_design`.
    '''

    def _save_cache_to_file(self):
        '''
        Store cache data into file.

        Data will be store with hdf5 format, placed at config..
        Cache file name format is "cache_START_END.h5"
        '''
        if self._cache_dir is None:
            raise DataSourceWithFileCacheError(
                'Use this class with "with statement" if you dont specify cache dir.')

        start_position = self.position - len(self._cache_data) + 1
        end_position = self.position
        cache_filename = os.path.join(
            self._cache_dir, 'cache_{:08d}_{:08d}.h5'.format(start_position, end_position))

        data = {n: [] for n in self._data_source.variables}
        for cd in self._cache_data:
            for i, n in enumerate(self._data_source.variables):
                if isinstance(cd[i], numpy.ndarray):
                    d = cd[i]
                else:
                    d = numpy.array(cd[i]).astype(numpy.float32)
                data[n].append(d)

        h5 = h5py.File(cache_filename, 'w')
        for k, v in data.items():
            h5.create_dataset(k, data=v)
        h5.close()

        self._cache_file_names.append(cache_filename)
        self._cache_file_order.append(len(self._cache_file_order))
        self._cache_file_data_orders.append(list(range(len(self._cache_data))))
        self._cache_data = []

    def _store_data_to_cache_buffer(self, position):
        d = self._data_source._get_data(position)
        if position == self._total_cached_size:
            self._cache_data.append(d)
            self._total_cached_size += 1
            if len(self._cache_data) >= self._cache_size or self._total_cached_size >= self.size:
                self._save_cache_to_file()
        return d

    def _get_data_from_cache_file(self, position):
        cache_file_index = self._cache_file_positions[position]
        cache_data_position = \
            self._cache_file_data_orders[cache_file_index][position -
                                                           self._cache_file_start_positions[cache_file_index]]

        if self._current_cache_file_index != cache_file_index:
            self._current_cache_file_index = cache_file_index

            h5 = h5py.File(self._cache_file_names[cache_file_index], 'r')
            self._current_cache_data = {}
            for k, v in h5.items():
                self._current_cache_data[k] = v.value
            h5.close()

        d = [self._current_cache_data[v][cache_data_position]
             for v in self.variables]
        return d

    def _get_data(self, position):
        self._position = position
        if self._generation <= 0:
            d = self._store_data_to_cache_buffer(position)
        else:
            d = self._get_data_from_cache_file(position)
        return d

    def __init__(self, data_source, cache_dir=None, shuffle=False, rng=None):
        logger.info('Using DataSourceWithFileCache')
        super(DataSourceWithFileCache, self).__init__(shuffle=shuffle, rng=rng)
        self._cache_dir = cache_dir
        self._cache_size = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_file_cache_size'))
        logger.info('Cache size is {}'.format(self._cache_size))
        self._size = data_source._size
        self._variables = data_source.variables
        self._data_source = data_source
        self._generation = -1
        self._cache_data = []
        self._total_cached_size = 0
        self._cache_file_names = []
        self._cache_file_order = []
        self._cache_file_start_positions = []
        self._cache_file_data_orders = []

        self._current_cache_file_index = -1
        self._current_cache_data = None

        self.shuffle = shuffle
        self._order = list(range(self._size))

        # __enter__
        self._tempdir_created = False
        if self._cache_dir is None:
            self._tempdir_created = True
            if nnabla_config.get('DATA_ITERATOR', 'data_source_file_cache_location') != '':
                self._cache_dir = tempfile.mkdtemp(dir=nnabla_config.get(
                    'DATA_ITERATOR', 'data_source_file_cache_location'))
            else:
                self._cache_dir = tempfile.mkdtemp()
            logger.info('Tempdir {} created.'.format(self._cache_dir))
        self._closed = False
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            if self._tempdir_created:
                # logger.info('Remove created tempdir {}'.format(self._cache_dir))
                rmtree(self._cache_dir, ignore_errors=True)
            self._data_source.close()
            self._closed = True

    def reset(self):
        if self._generation == 0:

            # Save all data into cache file(s).
            size = self._position + 1
            if size > self._data_source._size:
                size = self._data_source._size
            while self._position < self._data_source._size:
                self._store_data_to_cache_buffer(self._position)
                self._position += 1
            if len(self._cache_data) > 0:
                self._save_cache_to_file()

            # Adjust data size into reseted position. In most case it means
            # multiple of bunch(mini-batch) size.
            num_of_cache_files = int(numpy.ceil(
                float(size) / self._cache_size))
            self._cache_file_order = self._cache_file_order[
                0:num_of_cache_files]
            self._cache_file_data_orders = self._cache_file_data_orders[
                0:num_of_cache_files]
            if size % self._cache_size != 0:
                self._cache_file_data_orders[num_of_cache_files - 1] = self._cache_file_data_orders[
                    num_of_cache_files - 1][0:size % self._cache_size]

        elif self._generation > 0 and self._shuffle:
            self._cache_file_order = list(
                numpy.random.permutation(self._cache_file_order))
            for i in range(len(self._cache_file_data_orders)):
                self._cache_file_data_orders[i] = list(
                    numpy.random.permutation(self._cache_file_data_orders[i]))
            self._order = []
            for i in self._cache_file_order:
                self._order += self._cache_file_data_orders[i]

        if self._generation >= 0:
            # Create cached data position table.
            pos = 0
            self._cache_file_start_positions = list(
                range(len(self._cache_file_order)))
            self._order = list(range(len(self._order)))

            self._cache_file_positions = list(range(len(self._order)))
            count = 0
            for i, cache_file_pos in enumerate(self._cache_file_order):
                self._cache_file_start_positions[cache_file_pos] = pos
                pos += len(self._cache_file_data_orders[cache_file_pos])
                for j in self._cache_file_data_orders[cache_file_pos]:
                    p = j + (cache_file_pos * self._cache_size)
                    self._order[count] = p
                    self._cache_file_positions[count] = cache_file_pos
                    count += 1
        self._data_source.reset()
        self._position = 0
        self._generation += 1


class DataSourceWithMemoryCache(DataSource):
    '''
    Detailed documentation is available in :ref:`data_source_with_memory_cache_design`.
    '''

    def _get_data_func(self, position):
        return [numpy.array(x, dtype=numpy.float32) for x in self._data_source._get_data(position)]

    def _get_data(self, position):
        if self._on_memory:
            if self._order[position] < len(self._cache):
                data = self._cache[self._order[position]]
            else:
                data = self._get_data_func(position)
                self._cache.append(data)
        else:
            data = self._data_source._get_data(position)
        self._position = position
        return data

    def __init__(self, data_source, shuffle=False, rng=None):
        logger.info('Using DataSourceWithMemoryCache')
        super(DataSourceWithMemoryCache, self).__init__(
            shuffle=shuffle, rng=rng)
        self._buffer_max_size = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_buffer_max_size'))
        self._size = data_source._size
        self._variables = data_source.variables
        self._data_source = data_source
        self._order = list(range(self._size))

        self._on_memory = False
        self._cache = []

        data = self._get_data_func(0)
        self._data_size = 0
        for d in data:
            self._data_size += d.size * d.itemsize
        total_size = self._data_size * self._size
        if total_size < self._buffer_max_size:
            logger.info('On-memory')
            self._on_memory = True
        self._generation = -1
        self._closed = False
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            self._data_source.close()
            self._closed = True

    def reset(self):
        if self._on_memory:
            self._generation += 1
            if self._shuffle and self._generation > 0:
                self._order = list(self._rng.permutation(self._size))

            else:
                self._order = list(range(self._size))

            if self._position == 0:
                self._generation = -1
            else:
                self._data_source._position = self._position
                self._data_source.reset()
        else:
            self._data_source.reset()
            self._generation = self._data_source._generation
            self._position = self._data_source._position

        super(DataSourceWithMemoryCache, self).reset()
