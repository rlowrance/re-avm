import datetime
import sys
import pdb


from directory import directory

class Logger(object):
    # from stack overflow: how do i duplicat sys stdout to a log file in python

    def __init__(self, logfile, logfile_path=None, logfile_mode='w', base_name=None, add_time_stamp=False):
        if logfile_path is not None:
            raise RuntimeError('logfile_path is deprecated; use positional arg')
        if base_name is not None:
            raise RuntimeError('base_name is deprecated; use positional arg')
        if add_time_stamp:
            parts = logfile.split('.')
            timestamp =  datetime.datetime.now().isoformat('T')
            path = '.'.join(parts[:-1]) + '-' + timestamp + '.' + parts[-1]
        else:
            path = logfile
        self.terminal = sys.stdout
        self.log = open(path, logfile_mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        pass

if __name__ == '__main__':
    # unit test
    pdb.set_trace()
    sys.stdout = Logger('path/to/log/file.log')
    sys.stdout = Logger('path.to/log/file.log', add_time_stamp=True)
    # now print statements write on both stdout and the log file
