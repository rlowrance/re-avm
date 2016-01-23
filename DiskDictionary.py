import os
import pdb
import cPickle as pickle
import unittest


class DiskDictionary(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.f = None

    def append(self, key, value):
        'write key and value to disk'
        if self.f is None:
            self.f = open(self.filepath, 'wb')
        pickle.dump(key, self.f)
        pickle.dump(value, self.f)

    def close(self):
        if self.f is not None:
            self.f.close()

    def items(self):
        'generate (key, value) pairs that are in the disk file'
        if self.f is not None:
            raise RuntimeError('backing file already opened')
        self.f = open(self.filepath, 'rb')
        while True:
            try:
                key = pickle.load(self.f)
                value = pickle.load(self.f)
                yield key, value
            except EOFError:
                self.f.close()
                break  # fall through and yield StopIteration
        self.f.close()
        self.f = None

    def file_exists(self):
        'return True iff backing file exists at the provided path'
        return os.path.exists(self.filepath)

    def keyset(self):
        result = set()
        if self.file_exists():
            for k, v in self.items():
                assert k not in result, 'duplicate key: %s' % str(k)
                result.add(k)
        return result

    # implement with DiskDictionary(path) as dd:
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.f is not None:
            self.f.close()


class Test(unittest.TestCase):
    def test1(self):
        path = '/tmp/DiskDictionary'
        os.system('rm %s' % path)
        dd = DiskDictionary('/tmp/DiskDictionary')
        dd.append('key1', ['value1a', 27])
        dd.close()
        dd = DiskDictionary(path)
        found = 0
        for k, v in dd.items():
            found += 1

            self.assertTrue(isinstance(k, str))
            self.assertEqual(k, 'key1')

            self.assertTrue(isinstance(v, list))
            self.assertEqual(v[0], 'value1a')
            self.assertEqual(v[1], 27)
        self.assertEqual(found, 1)
        dd.close()

    def test_keyset(self):
        path = '/tmp/DiskDictionary'
        with DiskDictionary(path) as dd:
            dd.append('key1', ['value1', 1])
            dd.append('key2', ['valye2', 2])
        with DiskDictionary(path) as dd:
            keyset = dd.keyset()
            self.assertEqual(len(keyset), 2)

    def test_file_exists(self):
        path = '/tmp/blah blah'
        with DiskDictionary(path) as dd:
            self.assertFalse(dd.file_exists())


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
