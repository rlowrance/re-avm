'''run many jobs to fit models and predict all possible queries with them

INOVOCATION
 python fit-predict-make.py {training_data} {neighborhood} {model} {n_processes}  [--year {year}]
which concurrently run {n_processes} across relevant time periods and hp sets by running fit-predict.
 python fit-predict.py {training_data} {neighborhood} {model} YYYYMM
where
 training_data  in {train, all} specifies which data in Working/samples2 to use
 n_processes    is an int, the number of processes to run
'''

import argparse
import collections
import multiprocessing as mp
import os
import pdb
from pprint import pprint
import subprocess
import sys

import arg_type
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
    parser.add_argument('training_data', choices=arg_type.training_data_choices)
    parser.add_argument('neighborhood', type=arg_type.neighborhood)
    parser.add_argument('model', choices=['en', 'gb', 'rf'])
    parser.add_argument('n_processes', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--year', type=arg_type.year)
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

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


MapperArg = collections.namedtuple('MapperArg', 'training_data neighborhood model prediction_month test')
MapResult = collections.namedtuple('MapResult', 'mapper_arg error_level')


def mapper(mapper_arg):
    print 'mapper', mapper_arg
    invocation_args = '%s %s %s %s' % (
        mapper_arg.training_data,
        mapper_arg.neighborhood,
        mapper_arg.model,
        mapper_arg.prediction_month)
    invocation = 'python fit-predict.py ' + invocation_args
    print 'invocation', invocation
    if mapper_arg.test:
        return_code = 0
    elif os.name == 'nt':
        # options:
        #  /BELOWNORMAL  use BELOWNORMAL priority class
        #  /LOW          use IDLE priority class
        #  /WAIT         wait for app to terminate
        #  /B            start app without opening a new command window
        # NOTE: this approach seems to start 2 processes
        command = 'START "%s" /BELOWNORMAL /B %s' % (invocation_args, invocation)
        command_list = command.split(' ')
        print 'nt:', invocation
        # return_code = subprocess.call(command_list, shell=True)
        return_code = subprocess.call(invocation)  # it lowers its own priority
    elif os.name == 'posix':
        command = invocation 
        command_list = command.split(' ')
        print 'posix:', command_list
        return_code = subprocess.call(command_list)
    else:
        msg = 'unexpected os.name: ', os.name
        print msg
        raise RuntimeError(msg)
    return MapResult(
        mapper_arg=mapper_arg,
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
    pool = mp.Pool(processes=control.arg.n_processes)

    years = (2006, 2007, 2008, 2009) if control.arg.year is None else (control.arg.year,)
    prediction_months = [
        str(year * 100 + month)
        for year in years
        for month in ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) if year != 2009 else (1, 2, 3))
    ]
    print prediction_months
    mapper_arg = [
        MapperArg(
            training_data=control.arg.training_data,
            neighborhood=control.arg.neighborhood,
            model=control.arg.model,
            prediction_month=prediction_month,
            test=control.arg.test)
        for prediction_month in prediction_months
    ]
    mapper_arg = (
        [mapper_arg[0], mapper_arg[1]] if control.arg.test else
        mapper_arg
    )
    print 'mapper_arg'
    pprint(mapper_arg)

    mapped = pool.map(mapper, mapper_arg)
    print mapped
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
    if False:  # avoid flake8 warnings for unused imports
        pdb
        pprint
