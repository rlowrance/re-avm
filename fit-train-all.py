'''run many jobs to train models

INOVOCATION
 python fit-train-all.py model processes
which repeatedly runs in {processes} processes
 python fit.py train {model} LASTMONTH all
where
 model      is one of {en, gb, rf}, the model to train
 processes  is an int, the number of processes to run
'''

import argparse
import collections
import math
import multiprocessing as mp
import os
import pdb
import subprocess
import sys

import Bunch
import dirutility
import Logger
import Path
import Timer


def f(x):
    return x * x


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('model', choices=('en', 'gb', 'rf'))
    parser.add_argument('processes')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    try:
        arg.processes_int = int(arg.processes)
    except:
        print 'processes is not an int; was: %s' % arg.processes
        raise ValueError

    dir_working = Path.Path().dir_working()
    path_out_dir = (
        os.path.join(dir_working, arg.me + '-test') if arg.test else
        os.path.join(dir_working, arg.me)
    )
    dirutility.assure_exists(path_out_dir)

    return Bunch.Bunch(
        arg=arg,
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        timer=Timer.Timer(),
    )


MapperArg = collections.namedtuple('MapperArg', 'model last_month test')
MapResult = collections.namedtuple('MapResult', 'last_month error_level')


def mapper(mapper_arg):
    print 'mapper', mapper_arg
    invocation = 'python fit.py train %s %s all' % (mapper_arg.model, mapper_arg.last_month)
    if os.name == 'nt':
        # options:
        #  /BELOWNORMAL  use BELOWNORMAL priority class
        #  /LOW          use IDLE priority class
        #  /WAIT         wait for app to terminate
        #  /B            start app without opening a new command window
        # NOTE: this approach seems to start 2 processes, which is 1 too many
        window_title = str(mapper_arg.model) + str(mapper_arg.last_month)
        command = 'START "%s" /LOW /WAIT /B %s' % (window_title, invocation)
    elif os.name == 'posix':
        command = "nice 18 " + invocation  # set very low priority (19 is lowest)
    else:
        msg = 'unexpected os.name: ', os.name
        print msg
        raise RuntimeError(msg)
    print 'mapper', command
    # pdb.set_trace()
    return_code = (
        0 if mapper_arg.test else
        subprocess.call(invocation)  # fit many models
    )
    return MapResult(
        last_month=mapper_arg.last_month,
        error_level=return_code,
    )


def reducer(map_result_list):
    'reduce list[MapResult] to the maximum error level in the list'
    print 'reducer', len(map_result_list)
    max_error_level = None
    for map_result in map_result_list:
        print 'reducer', map_result
        error_level = map_result.error_level
        max_error_level = (
            error_level if max_error_level is None else
            max(max_error_level, error_level)
        )
    return max_error_level


def do_work(control):
    pool = mp.Pool(processes=control.arg.processes_int)

    last_months = [
        str(year * 100 + month)
        for year in (2005, 2006, 2007, 2008)
        for month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    ]
    last_months.extend(['200901', '200902', '200903'])
    print last_months
    mapper_arg = [
        MapperArg(model=control.arg.model, last_month=last_month, test=control.arg.test)
        for last_month in last_months
    ]
    mapper_arg = (
        [mapper_arg[0]] if control.arg.test else
        mapper_arg
    )
    print 'mapper_arg'
    print mapper_arg
    # pdb.set_trace()

    mapped = pool.map(mapper, mapper_arg)
    reduced = reducer(mapped)
    print 'max_error_level', reduced


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    main(sys.argv)
