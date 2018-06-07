from pandas import read_csv
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task


CHUNK_SIZE = 1000000
DS_NAME = 'petit'
PATH = '/path/to/dataset/'


class ChunkContainer(object):

    def __init__(self, name):
        self.name = name

    def path(self, i):
        return PATH + self.name + '_' + i + '.dat'

    def chunks_generator(self):
        chunk_num = 0
        while True:
            try:
                yield self.read_chunk(chunk_num)
                chunk_num += 1
            except IOError:
                return

    def read_chunk(self, i):
        return self._read_chunk_task(self.path(i))

    @task(path=FILE_IN)
    def _read_chunk_task(self, path):
        return read_csv(path, sep=' ', header=None, squeeze=True)

    def get(self, sorted_indexes):
        result = []
        chunk_num = -1
        chunk = None
        for i in sorted_indexes:
            current_chunk_num = i // CHUNK_SIZE
            if current_chunk_num > chunk_num:
                chunk_num = current_chunk_num
                chunk = self.read_chunk(current_chunk_num)
            result.append(chunk[i % CHUNK_SIZE])
        return result


class Y(ChunkContainer):

    def __init__(self):
        ChunkContainer.__init__(self, 'y')

    @classmethod
    def get_class_values(cls, sorted_sample):
        class_values = set()
        for sorted_sample_chunk in sorted_sample:
            class_values.union(cls.get(sorted_sample_chunk))
        return list(class_values)

    @task(chunk_name=FILE_IN)
    def read_y_chunk(self, chunk_name):
        return read_csv(PATH + chunk_name, sep=' ', header=None, squeeze=True)
