# dodo.py
import os

import Path


def task_create_samples2():
    'create sample2 files from sample files in data/working/'
    working = Path.Path().dir_working()
    me = os.path.join(working, 'sample2')
    return {
        'file_dep': [
            os.path.join(working, 'samples-test.csv'),
            os.path.join(working, 'sample-train.csv'),
        ],
        'targets': [
            os.path.join(me, 'duplicates.pickle'),
            os.path.join(me, 'uniques.pickle'),
            os.path.join(me, 'all.csv'),
            os.path.join(me, 'test.csv'),
            os.path.join(me, 'train.csv'),
        ],
        'actions': [
            'python samples2.py',
        ],
    }