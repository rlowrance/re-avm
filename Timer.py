import atexit
import os
import pdb
import time


class Timer(object):
    def __init__(self):
        # time.clock() returns:
        #  unix ==> processor time in seconds as float (cpu time)
        #  windows ==> wall-clock seconds since first call to this function
        #  NOTE: time.clock() is deprecated in python 3.3
        self._program_start_clock = time.clock()  # processor time in seconds
        # time.time() returns:
        #  unit & windows ==> time in seconds since epoch as float
        self._program_start_time = time.time()  # time in seconds since the epoch (on Unix)
        self._program = (self._program_start_clock, self._program_start_time)
        self._lap = (self._program_start_clock, self._program_start_time)
        atexit.register(self.endlaps)

    # initial API
    def elapsed_cpu_seconds(self):
        return time.clock() - self._program_start_clock

    def elapsed_wallclock_seconds(self):
        return time.time() - self._program_start_time

    # second API (keep first for backwards compatibility)
    def clock_time(self):
        return (time.clock(), time.time())

    def lap(self, s, verbose=True):
        'return (cpu seconds, wall clock seconds) in last lap; maybe print time of current lap'
        # NOTE: Cannot use the python standard library to find the elapsed CPU time on Windows
        # instead, Windows returns the wall clock time
        
        # inspired by Paul McGuire's timing.py
        # ref: http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution

        def toStr(t):
            'convert seconds to hh:mm:ss.sss'
            # this code from Paul McGuire!
            return '%d:%02d:%02d.%03d' % reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                                                [(t * 1000,), 1000, 60, 60])

        def diff(start, now):
            return (
                toStr(now[0] - start[0]),
                toStr(now[1] - start[1])
            )

        clock_time = self.clock_time()
        cumulative_seconds = diff(self._program, clock_time)
        lap_seconds = diff(self._lap, clock_time)
        self._lap = clock_time  # reset lap time
        if verbose:
            visual_clue = '=' * 50
            print visual_clue
            print 'lap: %s' % s
            print 'cumulative %s cpu %s wallclock' % cumulative_seconds
            print 'lap        %s cpu %s wallclock' % lap_seconds
            print visual_clue
            print
        return lap_seconds

    def endlaps(self):
        self.lap('**End Program**')
