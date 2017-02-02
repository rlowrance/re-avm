'''produce a columns table .txt file from a csv file

INVOCATION
 python columns_table.py --in PATH --out PATH --columndefs PATH --select COLUMN_NAMES  --header HEADER_lines

API: none, yet maybe as below
 make_columns_table(input_lines, selected_columns, columns_def) -> ColumnsTable

INPUTS
 in_PATH
 columnsdefs_PATH

OUTPUTS
 out_PATH
'''

from __future__ import division

import argparse
import collections
import json
import pandas as pd
import pdb
from pprint import pprint
import sys

import arg_type
from Bunch import Bunch


def make_control(argv):
    'return a Bunch'
    debug = True
    if debug:
        print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=arg_type.path_existing, required=True)
    parser.add_argument('--columndefs', type=arg_type.path_existing, required=True)
    parser.add_argument('--o', type=arg_type.path_creatable, required=True)
    parser.add_argument('--select', nargs='+', type=str, required=True)
    parser.add_argument('--header', nargs="*", type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    print arg
    arg.me = parser.prog.split('.')[0]
    arg.debug = debug

    if arg.trace:
        pdb.set_trace()

    return Bunch(
        arg=arg,
    )


def columns_table(input_lines, selected_columns, column_defs, header_lines):
    '''return List[str]
    input_lines: Iterable[Dict[str] Any] repeatedly yield dictionary of values, keys are column names
    selected_colums: List(str), names of columns in order
    columns_def: Dict[str, Dict], keys are column identifies, values are how to print
    header_lines: Iterable[str]
    NOTE: designed to be invocable from another program
    '''
    lines = []

    def append(line):
        lines.append(line + '\n')

    # print header_lines
    if header_lines is None or len(header_lines) == 0:
        pass
    else:
        for header in header_lines:
            append(header)
        append(' ')

    # determine number of label lines
    max_column_label_lines = 0
    for column_name in selected_columns:
        max_column_label_lines = max(max_column_label_lines, len(column_defs[column_name]['labels']))

    # prepend ' ' to labels that are shorter than the the max
    restated_labels = {}
    for column_name in selected_columns:
        labels = column_defs[column_name]['labels']
        while len(labels) < max_column_label_lines:
            labels.insert(0, ' ')
        restated_labels[column_name] = labels

    # print the label lines
    for label_row_index in xrange(max_column_label_lines):
        line = ''
        for i, column_name in enumerate(selected_columns):
            label = restated_labels[column_name][label_row_index]
            width = column_defs[column_name]['width']
            formatter = '%%%ds' % width
            print column_name, label, width, formatter
            if i > 0:
                line += ' '
            line += formatter % label
        append(line)
    append(' ')

    # print the detail lines
    for detail_info in input_lines:
        print detail_info
        line = ''
        for i, column_name in enumerate(selected_columns):
            if i > 0:
                line += ' '
            assert column_name in column_defs, column_name
            line += column_defs[column_name]['format'] % detail_info[column_name]
        append(line)

    # determine width of elements in the legend lines
    column_name_width = 0
    column_legend_width = 0
    for column_name in selected_columns:
        column_name_width = max(column_name_width, len(column_name))
        column_legend_width = max(column_legend_width, len(column_defs[column_name]['legend']))

    # print the legend lines
    append(' ')
    append('Legend:')
    formatter = '%%%ds -> %%%ds' % (column_name_width, column_legend_width)
    for column_name in selected_columns:
        line = formatter % (column_name, column_defs[column_name]['legend'])
        append(line)
    return lines


def main(argv):
    'handle invocation from the command line'
    # NOTE: This code has not been debugged
    pdb.set_trace()
    control = make_control(argv)
    if control.arg.debug:
        print control
    input = pd.read_csv(
        control.arg.in_,
        nrows=10 if control.arg.test else None,
        low_memory=False,
    )

    def generate_input():
        'yield each row'
        pdb.set_trace()
        for i, series in input.iterrows():
            yield series

    with open(control.arg.columnsdefs, 'w') as f:
        pdb.set_trace()
        column_defs = json.load(f)
        pprint(column_defs)

    figure = make_columns_table(
        generate_input,
        control.arg.select,
        column_defs,  # dictionary
        control.arg.header_lines,
    )

    with open(control.path_out, 'w') as f:
        figure
    return


if __name__ == '__main__':
    if False:
        pdb
    main(sys.argv[1:])
