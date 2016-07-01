'''summarize a data frame

OUTPUT FILES
 WORKING/summarize-df-IN.csv
 WORKING/summarize-df-IN-report.csv
'''

import cPickle as pickle
import pandas as pd
import pdb
import random
import sys

import Bunch
import Logger
import ParseCommandLine
import Path
from Report import Report
import summarize


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage : python summarize_df.py --in PATH [--test]'
    print ' PATH   : path to input csv'
    print ' --test : run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch
    print argv
    if len(argv) not in (3, 4):
        usage('invalid number of arguments')

    pcl = ParseCommandLine.ParseCommandLine(argv)
    arg = Bunch.Bunch(
        base_name=argv[0].split('.')[0],
        inpath=pcl.get_arg('--in'),
        test=pcl.has_arg('--test'),
    )

    if arg.inpath is None:
        usage('missing --in')

    random_seed = 123456
    random.seed(random_seed)

    path = Path.Path()  # use the default dir_input

    debug = False

    out_file_prefix = arg.base_name + '-' + arg.inpath.split('.')[0]
    out_file_base = out_file_prefix + ('-test' if arg.test else '')

    return Bunch.Bunch(
        arg=arg,
        debug=debug,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in=path.dir_working() + arg.inpath,
        path_out_summary=path.dir_working() + out_file_base + '.csv',
        path_out_report=path.dir_working() + out_file_base + '-report.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def make_report(summary):
    r = Report()
    format_header = '%40s %8s %8s %8s %8s %8s %8s %8s'
    format_detail = '%40s %8.0f %8.0f %8.0f %8.0f %8d %8d %8.0f'
    r.append(format_header % ('numeric feature', 'min', 'median', 'mean', 'max', 'distinct', 'NaN', 'std'))
    for row_name, row_value in summary.iterrows():
        r.append(format_detail % (
            row_name,
            row_value['min'],
            row_value['50%'],
            row_value['mean'],
            row_value['max'],
            row_value['number_distinct'],
            row_value['number_nan'],
            row_value['std']))
    return r


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(base_name=control.arg.base_name)
    print control

    in_df = pd.read_csv(control.path_in,
                        nrows=1000 if control.test else None,
                        )
    summary_df = summarize.summarize(in_df)
    report_summary = make_report(summary_df)
    # TODO: print correlations of each variable with price

    print summary_df

    # write output files
    summary_df.to_csv(control.path_out_summary)

    f = open(control.path_out_report, 'wb')
    pickle.dump((report_summary, control), f)
    f.close()

    if control.test:
        print 'DISCARD OUTPUT: TESTING'

    print control
    print 'done'


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    main(sys.argv)
