'hold all the knowledge about how the file system is layed out'

import os
import pdb


class Path(object):
    def __init__(self, dir_input='~/Dropbox/real-estate-los-angeles/'):
        assert dir_input[-1] == '/', dir_input + ' does not end in /'
        self._dir_input = os.path.expanduser(dir_input)
        self._dir_working = '../data/working/'  # relative to src directory
        self._dir_src = '../src/'

    def dir_input(self, file_id=None):
        def file_name(file_id):
            return file_id.split('-')[1]

        if file_id is None:
            return self.dir_input
        if file_id in ('deeds-CAC06037F1.zip',
                       'deeds-CAC06037F2.zip',
                       'deeds-CAC06037F3.zip',
                       'deeds-CAC06037F4.zip',
                       ):
            return self._dir_input + 'corelogic-deeds-090402_07/' + file_name(file_id)
        elif file_id in ('deeds-CAC06037F5.zip',
                         'deeds-CAC06037F6.zip',
                         'deeds-CAC06037F7.zip',
                         'deeds-CAC06037F8.zip',
                         ):
            return self._dir_input + 'corelogic-deeds-090402_09/' + file_name(file_id)
        elif file_id in ('parcels-CAC06037F1.zip',
                         'parcels-CAC06037F2.zip',
                         'parcels-CAC06037F3.zip',
                         'parcels-CAC06037F4.zip',
                         'parcels-CAC06037F5.zip',
                         'parcels-CAC06037F6.zip',
                         'parcels-CAC06037F7.zip',
                         'parcels-CAC06037F8.zip',
                         ):
            return self._dir_input + 'corelogic-taxrolls-090402_05/' + file_name(file_id)
        elif file_id == 'geocoding':
            return self._dir_input + 'geocoding.tsv'
        elif file_id == 'census':
            return self._dir_input + 'neighborhood-data/census.csv'
        else:
            print 'bad input file_id', file_id
            pdb.set_trace()

    def dir_src(self, file_id=None):
        if file_id is None:
            return self._dir_src
        else:
            print 'bad file_id', file_id
            pdb.set_trace()

    def dir_working(self, sub_dir_name=None):
        if sub_dir_name is None:
            return self._dir_working
        elif sub_dir_name == 'log':
            return self._dir_working + 'log/'
        else:
            print 'bad sub_dir_name', sub_dir_name
            pdb.set_trace()
