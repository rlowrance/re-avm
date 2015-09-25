import datetime
import sys
import pdb


from directory import directory


if False:
    pdb.set_trace()  # avoid warning message from pyflakes


class Logger(object):
    # from stack overflow: how do i duplicat sys stdout to a log file in python

    def __init__(self, logfile_path=None, logfile_mode='w', base_name=None):
        def path(s):
            return directory('log') + s + '-' + datetime.datetime.now().isoformat('T') + '.log'
        self.terminal = sys.stdout
        clean_path = logfile_path.replace(':', '-') if base_name is None else path(base_name)
        self.log = open(clean_path, logfile_mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush():
        pass

if False:
    # usage example
    sys.stdout = Logger('path/to/log/file')
    # now print statements write on both stdout and the log file
