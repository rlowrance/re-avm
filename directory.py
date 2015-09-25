def directory(name):
    '''Return path to specified directory in file system.

    Parameters
    ----------
    name : string
      name of the directory, one of cells, input, log, working

    Returns
    -------
    string: path to the named directory, ending with a "/"
    '''

    root = '../'  # root of this project
    if name == 'cells' or name == 'cv-cell' or name == 'cv-cell-natural':
        return root + 'data/working/cv-cell/'
    elif name == 'cv-cell-rescaled':
        return root + 'data/working/cv-cell-rescaled/'
    elif name == 'input':
        # input directory is shared across projects
        return '../../../real-estate-log-angeles/'
    elif name == 'log':
        return root + 'data/working/log/'
    elif name == 'working':
        return root + 'data/working/'
    elif name == 'src':
        return root + 'src/'
    else:
        raise ValueError(name)

if __name__ == '__main__':
    import unittest
    # import pdb

    class TestDirectory(unittest.TestCase):
        def is_ok(self, cell_name):
            # pdb.set_trace()
            s = directory(cell_name)
            if False:
                print cell_name, s
            self.assertTrue(isinstance(s, str))
            self.assertTrue(s.endswith('/'))

        def test_cells(self):
            self.is_ok('cells')

        def test_input(self):
            self.is_ok('input')

        def test_log(self):
            self.is_ok('log')

        def test_working(self):
            self.is_ok('working')

    unittest.main()
