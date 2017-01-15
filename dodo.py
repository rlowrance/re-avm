# dodo.py
import os

import HPs
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

last_months = [
    year * 100 + month
    for year in (2006, 2007, 2008)
    for month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
]


def task_fit_train_en_LASTMONTH_all():
    targets = [
        HPs.to_str(hp) + '.pickle'
        for hp in HPs.iter_hps_en()
    ]
    for last_month in last_months:
        yield {
            'action': 'python fit.py train en %d all' % last_month,
            'targets': targets,
        }
