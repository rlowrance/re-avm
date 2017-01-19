'''maintain a pickled cache file on disk'''
import cPickle as pickle
import os
import pdb
import time
import unittest

if False:
    # example
    class Cache(object):
        pass

    def read_data(dictionary):
        'return the data; it will be pickled and written to the file at path_to_cache'
        return None

    c = Cache(verbose=True)
    path_to_cache = os.path.join('a', 'b', 'c')
    dictionary = {'arg1': 123}
    returned_value_from_read_data = c.read(read_data, path_to_cache, dictionary)


class Cache(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def read(self, read_data_function, path_to_cache, dictionary):
        'return whatever read_data_function(**kwds) returns'
        start_time = time.time()
        if os.path.exists(path_to_cache):
            with open(path_to_cache, 'r') as f:
                cache = pickle.load(f)
            if self.verbose:
                print 'read cache; elapsed wall clock time', time.time() - start_time
        else:
            cache = read_data_function(dictionary)
            if self.verbose:
                print 'read underlying data; elapsed wall clock time', time.time() - start_time
            start_time = time.time()
            with open(path_to_cache, 'w') as f:
                pickle.dump(cache, f)
            if self.verbose:
                print 'write cache: elapsed wall clock time', time.time() - start_time
        return cache


class CacheTest(unittest.TestCase):
    def test_1(self):
        read_data_result = 'my data'
        dictionary = {'abc': 123}

        class Reader(object):
            def __init__(self):
                self.invocations = 0

            def read(self):
                self.invocations += 1
                return read_data_result

        reader = Reader()

        def read_data(dictionary):
            self.assertEqual(dictionary['abc'], 123)
            return reader.read()

        verbose = False
        dir_temp = os.getenv('temp')  # for now, just support Windows
        path_to_cache = os.path.join(dir_temp, 'Cache-test.pickle')

        if os.path.isfile(path_to_cache):
            os.remove(path_to_cache)
        self.assertFalse(os.path.isfile(path_to_cache))

        c = Cache(verbose=verbose)

        cached_data_1 = c.read(read_data, path_to_cache, dictionary)
        self.assertEqual(read_data_result, cached_data_1)
        self.assertTrue(os.path.isfile(path_to_cache))
        self.assertEqual(reader.invocations, 1)

        cached_data_2 = c.read(read_data, path_to_cache, dictionary)
        self.assertEqual(read_data_result, cached_data_2)
        self.assertTrue(os.path.isfile(path_to_cache))
        self.assertEqual(reader.invocations, 1)

        self.assertEqual(cached_data_1, cached_data_2)

        # remove cache file
        os.remove(path_to_cache)
        self.assertFalse(os.path.isfile(path_to_cache))

if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid linter warnings about imports not used
        pdb
