'hold all the knowledge about how the file system is layed out'

import os
import pdb
import sys

class Path(object):
    def __init__(self, dir_input=None):
        if dir_input is not None:
            print 'check code as API changed'
            print dir_input
            pdb.set_trace()
        if sys.platform == 'win32':
            pdb.set_trace()
            self._dir_data = os.path.join('C:\\', 'Users', 'roylo', 'Dropbox', 'data', '')
            self._dir_input = os.path.join(self._dir_data, 'real-estate-los-angeles', 'input', '')
            self._dir_working = os.path.join(self._dir_data, 'shasha', 're-avm', 'working', '')
        else:  # assume that we are running on Unix or MacOS
            self._dir_data = os.path.expanduser('~/Dropbox/data/')
            self._dir_input = self._dir_data + 'real-estate-los-angeles/input/'
            self._dir_working = self._dir_data + 'shasha/re-avm/working/'
        self._dir_src = os.path.join('..', 'src', '')

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
